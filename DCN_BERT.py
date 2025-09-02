"""
MSc Dissertation — DCN_BERT.py

"""

# ===================== Imports =====================
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import product

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight, resample
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from transformers import BertTokenizerFast, BertModel
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

# ===================== Reproducibility & Device =====================
def set_seed(seed: int = 42):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ===================== Preprocessing =====================
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def label_custom_stopwords(description, stopwords_set, stemmer, lemmatizer):
    """
    Replace any token found in stopwords_set with 'NAME', otherwise lemmatize+stem.
    Cleans brackets and non-alphanumerics; collapses whitespace.
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

def first_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """
    - Mask user IDs in Capture Remark.
    - Build stopwords from Excel/CSV.
    - Generate BERT-cleaned text and a rule-based processed variant.
    """
    data = data.copy()
    data["capture_remark_updated"] = data["Capture Remark"].str.replace(
        r"[qQ][a-zA-Z0-9]{6}", "userid", regex=True
    )

    # Adjust these paths if needed
    custom_stopwords = pd.read_excel(
        "/Users/prawin/Desktop/MSc Data Science/Dissertation/250524_ACTIVE_USERS_v1.xlsx"
    )
    second_stopwords = pd.read_csv(
        "/Users/prawin/Desktop/MSc Data Science/Dissertation/Domain Word Counts Exception.csv"
    )

    new_stopwords = set(
        custom_stopwords["SPERSNAME"].astype(str).tolist()
        + second_stopwords[second_stopwords["Relevant"] == "N"]["Column"].astype(str).tolist()
    )
    new_stopwords.difference_update(["-", "biw", "1", "area", "on", "road", "test", "for", "rework"])

    data["custom_stopwords_extracted"] = data["capture_remark_updated"].apply(
        lambda x: label_custom_stopwords(x, new_stopwords, stemmer, lemmatizer)
    )
    data["capture_description_processed"] = data["custom_stopwords_extracted"].apply(
        lambda x: x.replace("NAME", "")
    )
    data["capture_description_bert_processed"] = (
        data["capture_remark_updated"]
        .str.replace("NAME", "", regex=False)
        .apply(lambda x: re.sub(r"\s+", " ", str(x).lower().strip()))
    )
    data.drop(["custom_stopwords_extracted", "capture_remark_updated"], axis=1, inplace=True)
    return data.applymap(lambda x: re.sub(r"\s+", " ", x.lower().strip()) if isinstance(x, str) else x)

# ===================== Load & Prepare Data =====================
df = pd.read_excel("/Users/prawin/Desktop/MSc Data Science/Dissertation/Dataset_Sampled.xlsx")
df.columns = ["Defect Place", "Defect Type", "Capture Remark", "Precise Defect Description"]
data_all = first_preprocessing(df).astype(str)

# Optional rare class upsampling globally (folds will upsample train again)
min_samples = 6
label_counts = data_all["Precise Defect Description"].value_counts()
rare_classes = label_counts[label_counts < min_samples].index
resampled = [
    resample(
        data_all[data_all["Precise Defect Description"] == cls],
        replace=True, n_samples=min_samples, random_state=42
    )
    for cls in rare_classes
]
data_all = pd.concat(
    [data_all[~data_all["Precise Defect Description"].isin(rare_classes)]] + resampled
).sample(frac=1, random_state=42)

# Label encoding
le_place = LabelEncoder()
le_type  = LabelEncoder()
le_label = LabelEncoder()

data_all["place_enc"] = le_place.fit_transform(data_all["Defect Place"].astype(str))
data_all["type_enc"]  = le_type.fit_transform(data_all["Defect Type"].astype(str))
data_all["label_enc"] = le_label.fit_transform(data_all["Precise Defect Description"])

# Tokenizer / text
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

texts  = data_all["capture_description_bert_processed"].tolist()
places = data_all["place_enc"].tolist()
types_ = data_all["type_enc"].tolist()
labels = data_all["label_enc"].tolist()
n_classes = len(le_label.classes_)

# ===================== Model Components =====================
class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(input_dim))
        self.b = nn.Parameter(torch.zeros(input_dim))
    def forward(self, x0, x):
        xw = torch.sum(x * self.w, dim=1, keepdim=True)
        return x0 * xw + self.b + x

class DCN_BERT(nn.Module):
    def __init__(self, num_labels, num_place, num_type, emb_dim=32, num_cross=2, deep_dims=(128, 64), dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.place_emb = nn.Embedding(num_place, emb_dim)
        self.type_emb  = nn.Embedding(num_type, emb_dim)
        input_dim = self.bert.config.hidden_size + 2 * emb_dim
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_cross)])
        self.deep = nn.Sequential(
            nn.Linear(input_dim, deep_dims[0]),
            nn.ReLU(),
            nn.Linear(deep_dims[0], deep_dims[1]),
            nn.ReLU()
        )
        self.output  = nn.Linear(deep_dims[-1] + input_dim, num_labels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input_ids, attention_mask, place, type_):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        place_emb = self.place_emb(place)
        type_emb  = self.type_emb(type_)
        x = torch.cat((bert_out, place_emb, type_emb), dim=1)
        x0 = x.clone()
        for layer in self.cross_layers:
            x = layer(x0, x)
        deep_out = self.deep(x0)
        x_concat = torch.cat((x, deep_out), dim=1)
        return self.output(self.dropout(x_concat))

# Dataset & collator
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
            "place": torch.tensor(item["place"], dtype=torch.long),
            "type":  torch.tensor(item["type"],  dtype=torch.long),
        }

def collate_fn(batch):
    input_ids      = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    place          = torch.stack([item["place"] for item in batch])
    type_          = torch.stack([item["type"]  for item in batch])
    labels_batch   = torch.tensor([item["labels"] for item in batch])
    batch_encoding = tokenizer.pad({"input_ids": input_ids, "attention_mask": attention_mask}, return_tensors="pt")
    return {
        "input_ids": batch_encoding["input_ids"],
        "attention_mask": batch_encoding["attention_mask"],
        "place": place,
        "type": type_,
        "labels": labels_batch
    }

def make_hf_dataset(texts_x, places_x, types_x, labels_x):
    ds = Dataset.from_dict({"text": texts_x, "place": places_x, "type": types_x, "labels": labels_x})
    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "place", "type"])
    return ds

# ===================== Train / Eval Utilities =====================
def train_epoch(model, dataloader, optimizer, criterion, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        place = batch["place"].to(device)
        type_ = batch["type"].to(device)
        labels_batch = batch["labels"].to(device)
        logits = model(input_ids, attention_mask, place, type_)
        loss = criterion(logits, labels_batch)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))

@torch.no_grad()
def eval_epoch(model, dataloader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        place = batch["place"].to(device)
        type_ = batch["type"].to(device)
        labels_batch = batch["labels"].to(device)
        logits = model(input_ids, attention_mask, place, type_)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels_batch).sum().item()
        total   += labels_batch.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())
    acc = correct / max(1, total)
    return acc, np.array(all_preds), np.array(all_labels)

def metrics_four(y_true, y_pred):
    # Weighted metrics (single set of 4 metrics as requested)
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"acc": acc, "precision": p, "recall": r, "f1": f}

class EarlyStopper:
    def __init__(self, patience=2, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.counter = 0
        self.best_state = None
    def step(self, metric, model):
        improved = metric > (self.best + self.min_delta)
        if improved:
            self.best = metric
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return improved, self.counter >= self.patience
    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

# ===================== Configs =====================
BASE_CFG = dict(emb_dim=32, num_cross=2, deep_dims=(128, 64), dropout=0.3, lr=2e-5, epochs=8, patience=2)
PARAM_GRID = {
    "emb_dim":   [16, 32],
    "num_cross": [1, 2],
    "deep_dims": [(128, 64), (256, 128)],
    "dropout":   [0.1, 0.3],
    "lr":        [2e-5, 3e-5],
    "epochs":    [8],      # keep constant for fairness
    "patience":  [2],      # keep constant
}
BATCH_TRAIN = 16
BATCH_EVAL  = 32

def grid_dicts(grid):
    keys = list(grid.keys())
    from itertools import product
    for values in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))

# ===================== EXACT OUTER SPLIT: 80/20 =====================
indices = np.arange(len(labels))
dev_idx, test_idx = train_test_split(
    indices, test_size=0.20, stratify=np.array(labels), random_state=42
)

def take(lst, idxs): return [lst[i] for i in idxs]

X_test_text = take(texts,  test_idx)
X_test_pl   = take(places, test_idx)
X_test_ty   = take(types_, test_idx)
y_test      = take(labels, test_idx)

dev_texts  = take(texts,  dev_idx)
dev_places = take(places, dev_idx)
dev_types  = take(types_, dev_idx)
dev_labels = take(labels, dev_idx)

# ===================== CV  =====================
def run_cv_once(config, dev_texts, dev_places, dev_types, dev_labels):
    """
    Returns:
      metrics_list: list of dicts with keys acc, precision, recall, f1 (one per fold)
      best_state:   state dict of the best single fold model by val F1 (to optionally reuse)
    """
    inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    metrics_list = []
    best_state = None
    best_f1 = -np.inf

    for tr_sub, va_sub in inner_skf.split(dev_texts, dev_labels):
        # Split fold
        X_tr_text = [dev_texts[i]  for i in tr_sub]
        X_tr_pl   = [dev_places[i] for i in tr_sub]
        X_tr_ty   = [dev_types[i]  for i in tr_sub]
        y_tr      = [dev_labels[i] for i in tr_sub]

        X_va_text = [dev_texts[i]  for i in va_sub]
        X_va_pl   = [dev_places[i] for i in va_sub]
        X_va_ty   = [dev_types[i]  for i in va_sub]
        y_va      = [dev_labels[i] for i in va_sub]

        # Train-only upsampling
        MIN_SAMPLES = 6
        tr_df = pd.DataFrame({"text": X_tr_text, "place": X_tr_pl, "type": X_tr_ty, "label": y_tr})
        vc = tr_df["label"].value_counts()
        rare = vc[vc < MIN_SAMPLES].index
        ups = [resample(tr_df[tr_df["label"] == c], replace=True, n_samples=MIN_SAMPLES, random_state=42) for c in rare]
        if ups:
            tr_df = pd.concat([tr_df[~tr_df["label"].isin(rare)]] + ups).sample(frac=1, random_state=42)
        X_tr_text = tr_df["text"].tolist(); X_tr_pl = tr_df["place"].tolist(); X_tr_ty = tr_df["type"].tolist(); y_tr = tr_df["label"].tolist()

        # Data
        ds_tr = make_hf_dataset(X_tr_text, X_tr_pl, X_tr_ty, y_tr)
        ds_va = make_hf_dataset(X_va_text, X_va_pl, X_va_ty, y_va)
        train_dl = DataLoader(CustomDataset(ds_tr), batch_size=BATCH_TRAIN, shuffle=True,  collate_fn=collate_fn)
        val_dl   = DataLoader(CustomDataset(ds_va), batch_size=BATCH_EVAL,  shuffle=False, collate_fn=collate_fn)

        # Model
        model = DCN_BERT(
            num_labels=len(le_label.classes_),
            num_place=len(le_place.classes_),
            num_type=len(le_type.classes_),
            emb_dim=config["emb_dim"],
            num_cross=config["num_cross"],
            deep_dims=config["deep_dims"],
            dropout=config["dropout"]
        ).to(device)

        cw = compute_class_weight(class_weight="balanced", classes=np.unique(y_tr), y=y_tr)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=device))
        optimizer = AdamW(model.parameters(), lr=config["lr"])

        # Train with early stopping (silent)
        stopper = EarlyStopper(patience=config["patience"], min_delta=1e-4)
        for _ in range(config["epochs"]):
            _ = train_epoch(model, train_dl, optimizer, criterion)
            _, val_preds, val_true = eval_epoch(model, val_dl)
            val_m = metrics_four(val_true, val_preds)
            stopper.step(val_m["f1"], model)
            if stopper.counter >= stopper.patience:
                break
        stopper.restore(model)

        # Final fold metrics
        _, val_preds, val_true = eval_epoch(model, val_dl)
        val_m = metrics_four(val_true, val_preds)
        metrics_list.append(val_m)

        # Track best fold state (by F1)
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_state = stopper.best_state

    return metrics_list, best_state

def summarize_metrics(metrics_list):
    arr_acc = np.array([m["acc"] for m in metrics_list])
    arr_p   = np.array([m["precision"] for m in metrics_list])
    arr_r   = np.array([m["recall"] for m in metrics_list])
    arr_f1  = np.array([m["f1"] for m in metrics_list])
    return {
        "acc_mean": arr_acc.mean(), "acc_std": arr_acc.std(),
        "p_mean":   arr_p.mean(),   "p_std":   arr_p.std(),
        "r_mean":   arr_r.mean(),   "r_std":   arr_r.std(),
        "f1_mean":  arr_f1.mean(),  "f1_std":  arr_f1.std(),
    }

# ===================== BASELINE CV =====================
baseline_metrics_list, baseline_best_state = run_cv_once(BASE_CFG, dev_texts, dev_places, dev_types, dev_labels)
baseline_cv = summarize_metrics(baseline_metrics_list)

# ===================== Final Test =====================
# Load best baseline fold model to test
baseline_model = DCN_BERT(
    num_labels=len(le_label.classes_),
    num_place=len(le_place.classes_),
    num_type=len(le_type.classes_),
    emb_dim=BASE_CFG["emb_dim"],
    num_cross=BASE_CFG["num_cross"],
    deep_dims=BASE_CFG["deep_dims"],
    dropout=BASE_CFG["dropout"],
).to(device)
if baseline_best_state is not None:
    baseline_model.load_state_dict(baseline_best_state)

ds_te = make_hf_dataset(X_test_text, X_test_pl, X_test_ty, y_test)
test_dl = DataLoader(CustomDataset(ds_te), batch_size=BATCH_EVAL, shuffle=False, collate_fn=collate_fn)
_, te_preds_base, te_true = eval_epoch(baseline_model, test_dl)
baseline_test = metrics_four(te_true, te_preds_base)

# ===================== HYPERPARAMETER TUNING CV =====================
best_cfg = None
best_cfg_stats = None
best_cfg_f1_mean = -np.inf
best_cfg_state = None

for cfg in grid_dicts(PARAM_GRID):
    metrics_list, cfg_state = run_cv_once(cfg, dev_texts, dev_places, dev_types, dev_labels)
    stats = summarize_metrics(metrics_list)
    if stats["f1_mean"] > best_cfg_f1_mean:
        best_cfg_f1_mean = stats["f1_mean"]
        best_cfg = cfg
        best_cfg_stats = stats
        best_cfg_state = cfg_state

# ===================== Tuned Final Test =====================
tuned_model = DCN_BERT(
    num_labels=len(le_label.classes_),
    num_place=len(le_place.classes_),
    num_type=len(le_type.classes_),
    emb_dim=best_cfg["emb_dim"],
    num_cross=best_cfg["num_cross"],
    deep_dims=best_cfg["deep_dims"],
    dropout=best_cfg["dropout"],
).to(device)
if best_cfg_state is not None:
    tuned_model.load_state_dict(best_cfg_state)

_, te_preds_tuned, te_true = eval_epoch(tuned_model, test_dl)
tuned_test = metrics_four(te_true, te_preds_tuned)

# ===================== PRINT =====================
# 1) Baseline CV: mean ± std
print("BASELINE — CV (on dev 80%)")
print(f"Accuracy: {baseline_cv['acc_mean']:.4f} ± {baseline_cv['acc_std']:.4f}")
print(f"Precision: {baseline_cv['p_mean']:.4f} ± {baseline_cv['p_std']:.4f}")
print(f"Recall: {baseline_cv['r_mean']:.4f} ± {baseline_cv['r_std']:.4f}")
print(f"F1: {baseline_cv['f1_mean']:.4f} ± {baseline_cv['f1_std']:.4f}")

# 2) Baseline Final Test: single scores
print("\nBASELINE — Final Test (fixed 20%)")
print(f"Accuracy: {baseline_test['acc']:.4f}")
print(f"Precision: {baseline_test['precision']:.4f}")
print(f"Recall: {baseline_test['recall']:.4f}")
print(f"F1: {baseline_test['f1']:.4f}")

# 3) Tuned CV: mean ± std (best config)
print("\nTUNED — CV (on dev 80%) [best config]")
print(f"Accuracy: {best_cfg_stats['acc_mean']:.4f} ± {best_cfg_stats['acc_std']:.4f}")
print(f"Precision: {best_cfg_stats['p_mean']:.4f} ± {best_cfg_stats['p_std']:.4f}")
print(f"Recall: {best_cfg_stats['r_mean']:.4f} ± {best_cfg_stats['r_std']:.4f}")
print(f"F1: {best_cfg_stats['f1_mean']:.4f} ± {best_cfg_stats['f1_std']:.4f}")

# 4) Tuned Final Test: single scores
print("\nTUNED — Final Test (fixed 20%)")
print(f"Accuracy: {tuned_test['acc']:.4f}")
print(f"Precision: {tuned_test['precision']:.4f}")
print(f"Recall: {tuned_test['recall']:.4f}")
print(f"F1: {tuned_test['f1']:.4f}")
