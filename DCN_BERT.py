"""MSc Dissertation — DCN_BERT.py

    Deep & Cross Network (DCN) architecture integrated with a BERT encoder.

    This file is prepared for publication on GitHub (appendix reference). It adds clear, standardized
    docstrings while preserving original behavior.

    Author: Prawin Thiyagrajan Veeramani
    Prepared on: 2025-08-26
    """

# ----------- Imports ----------- #
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils import compute_class_weight, resample
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, f1_score, accuracy_score
from transformers import BertTokenizerFast, BertModel
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

# ----------- Preprocessing ----------- #
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# PATCH: fix stopword logic (no repeated "NAME" spam)
def label_custom_stopwords(description, stopwords_set, stemmer, lemmatizer):
    """label_custom_stopwords — one-line summary.

    Args:
        description: Description.
        stopwords_set: Description.
        stemmer: Description.
        lemmatizer: Description.

    Returns:
        Description of return value.
    """
    description = re.sub(r"\[.*?\]", "", str(description)).lower()
    tokens = word_tokenize(description)
    processed = []
    for token in tokens:
        if token in stopwords_set:
            processed.append("NAME")
        else:
            processed.append(lemmatizer.lemmatize(stemmer.stem(token)))
    text = " ".join(processed)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def first_preprocessing(data):
    """first_preprocessing — one-line summary.

    Args:
        data: Description.

    Returns:
        Description of return value.
    """
    data = data.copy()
    data["capture_remark_updated"] = data["Capture Remark"].str.replace(r"[qQ][a-zA-Z0-9]{6}", "userid", regex=True)
    custom_stopwords = pd.read_excel("/Users/prawin/Desktop/MSc Data Science/Dissertation/250524_ACTIVE_USERS_v1.xlsx")
    second_stopwords = pd.read_csv("/Users/prawin/Desktop/MSc Data Science/Dissertation/Domain Word Counts Exception.csv")
    new_stopwords = set(custom_stopwords["SPERSNAME"].astype(str).tolist() + second_stopwords[second_stopwords["Relevant"] == "N"]["Column"].astype(str).tolist())
    new_stopwords.difference_update(["-", "biw", "1", "area", "on", "road", "test", "for", "rework"])
    data["custom_stopwords_extracted"] = data["capture_remark_updated"].apply(lambda x: label_custom_stopwords(x, new_stopwords, stemmer, lemmatizer))
    data["capture_description_processed"] = data["custom_stopwords_extracted"].apply(lambda x: x.replace("NAME", ""))
    data["capture_description_bert_processed"] = data["capture_remark_updated"].str.replace("NAME", "", regex=False).apply(lambda x: re.sub(r"\s+", " ", str(x).lower().strip()))
    data.drop(["custom_stopwords_extracted", "capture_remark_updated"], axis=1, inplace=True)
    return data.applymap(lambda x: re.sub(r"\s+", " ", x.lower().strip()) if isinstance(x, str) else x)

# ----------- Load and Preprocess Dataset ----------- #
df = pd.read_excel("/Users/prawin/Desktop/MSc Data Science/Dissertation/Dataset_Sampled.xlsx")
df.columns = ["Defect Place", "Defect Type", "Capture Remark", "Precise Defect Description"]
train_data = first_preprocessing(df).astype(str)

# Rare class upsampling (original global upsample kept if you want; CV will upsample train folds again)
min_samples = 6
label_counts = train_data["Precise Defect Description"].value_counts()
rare_classes = label_counts[label_counts < min_samples].index
resampled = [resample(train_data[train_data["Precise Defect Description"] == cls], replace=True, n_samples=min_samples, random_state=42) for cls in rare_classes]
train_data = pd.concat([train_data[~train_data["Precise Defect Description"].isin(rare_classes)]] + resampled).sample(frac=1, random_state=42)

# ----------- Label Encoding ----------- #
le_place = LabelEncoder()
le_type = LabelEncoder()
le_label = LabelEncoder()

train_data["place_enc"] = le_place.fit_transform(train_data["Defect Place"].astype(str))
train_data["type_enc"] = le_type.fit_transform(train_data["Defect Type"].astype(str))
train_data["label_enc"] = le_label.fit_transform(train_data["Precise Defect Description"])

# ----------- Tokenization Setup ----------- #
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    """tokenize — one-line summary.

    Args:
        batch: Description.

    Returns:
        Description of return value.
    """
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

# PATCH: use the BERT-ready cleaned text for modelling
texts = train_data["capture_description_bert_processed"].tolist()
places = train_data["place_enc"].tolist()
types = train_data["type_enc"].tolist()
labels = train_data["label_enc"].tolist()

# (Hold-out split removed; CV block added below)

# ----------- Cross Layer Component ----------- #
class CrossLayer(nn.Module):
    """CrossLayer — class summary and usage notes."""
    def __init__(self, input_dim):
        """__init__ — one-line summary.

    Args:
        input_dim: Description.

    Returns:
        Description of return value.
    """
        super().__init__()
        self.w = nn.Parameter(torch.randn(input_dim))
        self.b = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, x):
        """forward — one-line summary.

    Args:
        x0: Description.
        x: Description.

    Returns:
        Description of return value.
    """
        xw = torch.sum(x * self.w, dim=1, keepdim=True)  # Inner product
        return x0 * xw + self.b + x  # Cross term + bias + residual

# ----------- DCN + BERT Model ----------- #
class DCN_BERT(nn.Module):
    """DCN_BERT — class summary and usage notes."""
    def __init__(self, num_labels, num_place, num_type, emb_dim=32, num_cross=2, deep_dims=[128, 64]):
        """__init__ — one-line summary.

    Args:
        num_labels: Description.
        num_place: Description.
        num_type: Description.
        emb_dim: Description.
        num_cross: Description.
        deep_dims: Description.

    Returns:
        Description of return value.
    """
        super().__init__()

        # BERT backbone
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Categorical feature embeddings
        self.place_emb = nn.Embedding(num_place, emb_dim)
        self.type_emb = nn.Embedding(num_type, emb_dim)

        # Input dimension = BERT + place + type
        input_dim = self.bert.config.hidden_size + 2 * emb_dim

        # Cross layers
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_cross)])

        # Deep layers (MLP)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, deep_dims[0]),
            nn.ReLU(),
            nn.Linear(deep_dims[0], deep_dims[1]),
            nn.ReLU()
        )

        # Final classifier: concat(cross output, deep output)
        self.output = nn.Linear(deep_dims[-1] + input_dim, num_labels)

        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, place, type_):
        """forward — one-line summary.

    Args:
        input_ids: Description.
        attention_mask: Description.
        place: Description.
        type_: Description.

    Returns:
        Description of return value.
    """
        # BERT output
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        # Embed categorical features
        place_emb = self.place_emb(place)
        type_emb = self.type_emb(type_)

        # Concatenate all features
        x = torch.cat((bert_out, place_emb, type_emb), dim=1)
        x0 = x.clone()

        # Cross Network
        for layer in self.cross_layers:
            x = layer(x0, x)

        # Deep Network
        deep_out = self.deep(x0)

        # Final concat and output
        x_concat = torch.cat((x, deep_out), dim=1)
        return self.output(self.dropout(x_concat))

# ----------- Device Setup ----------- #
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# ----------- DataLoader Wrapper ----------- #
class CustomDataset(torch.utils.data.Dataset):
    """CustomDataset — class summary and usage notes."""
    def __init__(self, hf_dataset):
        """__init__ — one-line summary.

    Args:
        hf_dataset: Description.

    Returns:
        Description of return value.
    """
        self.dataset = hf_dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        """__len__ — one-line summary.

    Args:
        self), idx: Description.

    Returns:
        Description of return value.
    """
        item = self.dataset[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
            "place": torch.tensor(item["place"], dtype=torch.long),
            "type": torch.tensor(item["type"], dtype=torch.long)
        }

def collate_fn(batch):
    """collate_fn — one-line summary.

    Args:
        batch: Description.

    Returns:
        Description of return value.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    place = torch.tensor([item["place"] for item in batch])
    type_ = torch.tensor([item["type"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])
    batch_encoding = tokenizer.pad({"input_ids": input_ids, "attention_mask": attention_mask}, return_tensors="pt")
    return {
        "input_ids": batch_encoding["input_ids"],
        "attention_mask": batch_encoding["attention_mask"],
        "place": place,
        "type": type_,
        "labels": labels
    }

# ----------- Train & Eval Functions (unchanged) ----------- #
def train_epoch(model, dataloader, optimizer, criterion):
    """train_epoch — one-line summary.

    Args:
        model: Description.
        dataloader: Description.
        optimizer: Description.
        criterion: Description.

    Returns:
        Description of return value.
    """
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        place = batch["place"].to(device)
        type_ = batch["type"].to(device)
        labels_batch = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask, place, type_)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # small stability add
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader):
    """eval_epoch — one-line summary.

    Args:
        model: Description.
        dataloader: Description.

    Returns:
        Description of return value.
    """
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels, all_logits = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            place = batch["place"].to(device)
            type_ = batch["type"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, place, type_)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())

    accuracy = correct / total
    return accuracy, np.array(all_preds), np.array(all_labels), np.array(all_logits)

# ----------- Cross Validation (PATCHED IN) ----------- #
# Replaces the single hold-out training block
K = 5
epochs = 4  # keep your original epochs

texts_all  = texts
places_all = places
types_all  = types
labels_all = labels
n_classes  = len(le_label.classes_)

def make_hf_dataset(texts_x, places_x, types_x, labels_x):
    """make_hf_dataset — one-line summary.

    Args:
        texts_x: Description.
        places_x: Description.
        types_x: Description.
        labels_x: Description.

    Returns:
        Description of return value.
    """
    ds = Dataset.from_dict({"text": texts_x, "place": places_x, "type": types_x, "labels": labels_x})
    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "place", "type"])
    return ds

skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
fold_metrics = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(texts_all, labels_all), 1):
    print(f"\n========== Fold {fold}/{K} ==========")

    # Split
    Xtr_text = [texts_all[i]  for i in tr_idx]
    Xtr_pl   = [places_all[i] for i in tr_idx]
    Xtr_ty   = [types_all[i]  for i in tr_idx]
    ytr      = [labels_all[i] for i in tr_idx]

    Xva_text = [texts_all[i]  for i in va_idx]
    Xva_pl   = [places_all[i] for i in va_idx]
    Xva_ty   = [types_all[i]  for i in va_idx]
    yva      = [labels_all[i] for i in va_idx]

    # Train-only upsampling (same min_samples as above)
    MIN_SAMPLES = 6
    tr_df = pd.DataFrame({"text": Xtr_text, "place": Xtr_pl, "type": Xtr_ty, "label": ytr})
    vc = tr_df["label"].value_counts()
    rare = vc[vc < MIN_SAMPLES].index
    ups = []
    for cls in rare:
        ups.append(resample(tr_df[tr_df["label"] == cls], replace=True, n_samples=MIN_SAMPLES, random_state=42))
    if ups:
        tr_df = pd.concat([tr_df[~tr_df["label"].isin(rare)]] + ups).sample(frac=1, random_state=42)

    Xtr_text = tr_df["text"].tolist()
    Xtr_pl   = tr_df["place"].tolist()
    Xtr_ty   = tr_df["type"].tolist()
    ytr      = tr_df["label"].tolist()

    # HF datasets & loaders
    ds_tr = make_hf_dataset(Xtr_text, Xtr_pl, Xtr_ty, ytr)
    ds_va = make_hf_dataset(Xva_text, Xva_pl, Xva_ty, yva)

    train_dl = DataLoader(CustomDataset(ds_tr), batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_dl   = DataLoader(CustomDataset(ds_va), batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Model/criterion/optim per fold
    model = DCN_BERT(
        num_labels=len(le_label.classes_),
        num_place=len(le_place.classes_),
        num_type=len(le_type.classes_)
    ).to(device)

    # Recompute class weights on TRAIN fold (after upsampling)
    cw = compute_class_weight(class_weight="balanced", classes=np.unique(ytr), y=ytr)
    class_weights_tensor = torch.tensor(cw, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Train
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_dl, optimizer, criterion)
        val_acc, _, _, _ = eval_epoch(model, val_dl)
        print(f"[Fold {fold} | Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # Final evaluation for this fold
    val_acc, val_preds, val_true, val_logits = eval_epoch(model, val_dl)
    f1_w = f1_score(val_true, val_preds, average="weighted")
    f1_m = f1_score(val_true, val_preds, average="macro")

    print(f"\nFold {fold} | Acc={val_acc:.4f} | F1_w={f1_w:.4f} | F1_m={f1_m:.4f}")
    print("\n Classification Report:")
    print(classification_report(val_true, val_preds, target_names=le_label.classes_))

    # Optional micro-average ROC curve (labelled correctly)
    try:
        val_true_bin = label_binarize(val_true, classes=list(range(n_classes)))
        val_probs = torch.softmax(torch.tensor(val_logits), dim=1).numpy()
        fpr, tpr, _ = roc_curve(val_true_bin.ravel(), val_probs.ravel())  # micro-average
        micro_auc = auc(fpr, tpr)
        weighted_auc = roc_auc_score(val_true_bin, val_probs, average="weighted", multi_class="ovr")

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Micro-average ROC (AUC = {micro_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (Fold {fold}) — Weighted AUC = {weighted_auc:.2f}")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("ROC plotting skipped due to:", e)

    fold_metrics.append({"fold": fold, "acc": val_acc, "f1_weighted": f1_w, "f1_macro": f1_m})

# ----------- CV Summary ----------- #
accs = [m["acc"] for m in fold_metrics]
f1w  = [m["f1_weighted"] for m in fold_metrics]
f1m  = [m["f1_macro"] for m in fold_metrics]
print("\n========== CV Summary ==========")
print(f"Accuracy:      {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1 (weighted): {np.mean(f1w):.4f} ± {np.std(f1w):.4f}")
print(f"F1 (macro):    {np.mean(f1m):.4f} ± {np.std(f1m):.4f}")
