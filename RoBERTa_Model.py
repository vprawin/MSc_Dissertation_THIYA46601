"""
MSc Dissertation — RoBERTa_Model.py

RoBERTa-based model with categorical embeddings (place/type) and standardized
training/evaluation pipeline for the dissertation

Author: Prawin Thiyagrajan Veeramani
Prepared on: 2025-08-26
"""

# ==================== LIBRARIES ====================
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from transformers import RobertaTokenizerFast, RobertaModel
from datasets import Dataset

# ==================== SEED & DEVICE ====================
def set_seed(seed=42):
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

# ==================== LOAD DATA ====================
df = pd.read_excel('/Users/prawin/Desktop/MSc Data Science/Dissertation/Dataset_Sampled.xlsx')
df.columns = ['Defect Place', 'Defect Type', 'Capture Remark', 'Precise Defect Description']

# ==================== TEXT PREPROCESSING ====================
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def label_custom_stopwords(description, stopwords_set, stemmer, lemmatizer):
    """
    Replace any token found in stopwords_set with 'NAME', otherwise lemmatize+stem.
    Cleans brackets and non-alphanumerics; collapses whitespace.
    (Fixed: avoids 'NAME' spam by checking membership correctly.)
    """
    description = re.sub(r'\[.*?\]', '', str(description)).lower()
    tokens = word_tokenize(description)
    processed = []
    for token in tokens:
        if token in stopwords_set:
            processed.append('NAME')
        else:
            processed.append(lemmatizer.lemmatize(stemmer.stem(token)))
    text = ' '.join(processed)
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def first_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """
    - Mask user IDs in Capture Remark.
    - Build stopwords from Excel/CSV.
    - Generate (a) processed text w/o NAME, and (b) model-ready lowercased text.
    """
    rows = data.copy()
    rows['capture_remark_updated'] = rows['Capture Remark'].str.replace(
        r"[qQ][a-zA-Z0-9]{6}", 'userid', regex=True
    )

    custom_stopwords = pd.read_excel('/Users/prawin/Desktop/MSc Data Science/Dissertation/250524_ACTIVE_USERS_v1.xlsx')
    second_stopwords = pd.read_csv('/Users/prawin/Desktop/MSc Data Science/Dissertation/Domain Word Counts Exception.csv')

    filtered_words = second_stopwords[second_stopwords['Relevant'] == 'N']
    custom_names = custom_stopwords['SPERSNAME'].astype(str).tolist()
    additional = filtered_words['Column'].astype(str).tolist()

    new_stopwords = set(custom_names + additional)
    new_stopwords.difference_update(['-', 'biw', '1', 'area', 'on', 'road', 'test', 'for', 'rework'])

    rows['custom_stopwords_extracted'] = rows['capture_remark_updated'].apply(
        lambda x: label_custom_stopwords(x, new_stopwords, stemmer, lemmatizer)
    )
    rows['capture_description_processed'] = rows['custom_stopwords_extracted'].apply(lambda x: x.replace('NAME', ''))
    rows['capture_description_roberta_processed'] = (
        rows['capture_remark_updated'].str.replace('NAME', '', regex=False)
        .apply(lambda x: re.sub(r'\s+', ' ', str(x).lower().strip()))
    )
    rows.drop(['custom_stopwords_extracted', 'capture_remark_updated'], axis=1, inplace=True)
    rows = rows.applymap(lambda x: re.sub(r'\s+', ' ', x.lower().strip()) if isinstance(x, str) else x)
    return rows

data_all = first_preprocessing(df).astype(str)

# ==================== RARE CLASS UPSAMPLING ====================
min_samples = 6
label_counts = data_all['Precise Defect Description'].value_counts()
rare_classes = label_counts[label_counts < min_samples].index
resampled_frames = [
    resample(
        data_all[data_all['Precise Defect Description'] == cls],
        replace=True, n_samples=min_samples, random_state=42
    )
    for cls in rare_classes
]
df_majority = data_all[~data_all['Precise Defect Description'].isin(rare_classes)]
data_all = pd.concat([df_majority] + resampled_frames).sample(frac=1, random_state=42).reset_index(drop=True)

# ==================== ENCODING ====================
le_place = LabelEncoder()
le_type  = LabelEncoder()
le_label = LabelEncoder()

data_all['place_enc'] = le_place.fit_transform(data_all['Defect Place'].astype(str))
data_all['type_enc']  = le_type.fit_transform(data_all['Defect Type'].astype(str))
data_all['label_enc'] = le_label.fit_transform(data_all['Precise Defect Description'])

# Use the model-cleaned text
texts  = data_all['capture_description_roberta_processed'].tolist()
places = data_all['place_enc'].tolist()
types_ = data_all['type_enc'].tolist()
labels = data_all['label_enc'].tolist()

n_labels = len(le_label.classes_)
n_place  = len(le_place.classes_)
n_type   = len(le_type.classes_)

# ==================== TOKENIZATION ====================
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
def tokenize(batch_texts): return tokenizer(batch_texts, padding=True, truncation=True, max_length=128)

def make_hf_dataset(texts_x, places_x, types_x, labels_x):
    ds = Dataset.from_dict({'text': texts_x, 'place': places_x, 'type': types_x, 'labels': labels_x})
    ds = ds.map(lambda b: tokenize(b['text']), batched=True)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'place', 'type'])
    return ds

# ==================== MODEL ====================
class RoBERTa_Model(nn.Module):
    """
    RoBERTa classifier with additional embeddings for categorical features (place/type).
    """
    def __init__(self, roberta_model_name, num_labels, num_place, num_type, emb_dim=32, dropout=0.3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.place_emb = nn.Embedding(num_place, emb_dim)
        self.type_emb  = nn.Embedding(num_type, emb_dim)
        self.dropout = nn.Dropout(dropout)
        combined_dim = self.roberta.config.hidden_size + 2 * emb_dim
        self.classifier = nn.Linear(combined_dim, num_labels)

    def forward(self, input_ids, attention_mask, place, type_):
        # RoBERTa doesn't expose pooler_output by default; use <s> token representation
        cls_rep = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        combined = torch.cat((cls_rep, self.place_emb(place), self.type_emb(type_)), dim=1)
        return self.classifier(self.dropout(combined))

# ==================== DATA WRAPPERS ====================
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset): self.dataset = hf_dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': item['labels'],
            'place': torch.tensor(item['place'], dtype=torch.long),
            'type':  torch.tensor(item['type'],  dtype=torch.long)
        }

def collate_fn(batch):
    input_ids      = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    place          = torch.stack([item['place'] for item in batch])
    type_          = torch.stack([item['type']  for item in batch])
    labels_batch   = torch.tensor([item['labels'] for item in batch])
    enc = tokenizer.pad({'input_ids': input_ids, 'attention_mask': attention_mask}, return_tensors='pt')
    return {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'],
            'place': place, 'type': type_, 'labels': labels_batch}

# ==================== TRAIN / EVAL HELPERS ====================
def train_epoch(model, dataloader, optimizer, criterion, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        logits = model(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['place'].to(device),
            batch['type'].to(device),
        )
        loss = criterion(logits, batch['labels'].to(device))
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))

@torch.no_grad()
def eval_epoch(model, dataloader):
    model.eval()
    correct, total = 0, 0
    preds_all, labels_all = [], []
    for batch in dataloader:
        logits = model(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['place'].to(device),
            batch['type'].to(device),
        )
        labels = batch['labels'].to(device)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    acc = correct / max(1, total)
    return acc, np.array(preds_all), np.array(labels_all)

def metrics_four(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {'acc': acc, 'precision': p, 'recall': r, 'f1': f}

def make_class_weights(y_train, n_classes):
    counts = np.bincount(np.array(y_train, dtype=np.int64), minlength=n_classes)
    weights = np.zeros(n_classes, dtype=np.float32)
    nonzero = counts > 0
    if nonzero.any():
        weights[nonzero] = counts.sum() / (n_classes * counts[nonzero])
    # leave zeros for absent classes in this fold
    return torch.tensor(weights, dtype=torch.float, device=device)

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

# ==================== CONFIGS ====================
BATCH_TRAIN = 16
BATCH_EVAL  = 32

BASE_CFG = dict(
    emb_dim=32,
    dropout=0.3,
    lr=2e-5,
    epochs=8,
    patience=2
)

# Small, safe grid for tuning
PARAM_GRID = [
    dict(emb_dim=16, dropout=0.1, lr=2e-5, epochs=8, patience=2),
    dict(emb_dim=32, dropout=0.3, lr=2e-5, epochs=8, patience=2),
    dict(emb_dim=32, dropout=0.2, lr=3e-5, epochs=8, patience=2),
]

# ==================== SPLIT: 80/20 (dev/test) ====================
idx_all = np.arange(len(labels))
dev_idx, test_idx = train_test_split(idx_all, test_size=0.20, stratify=np.array(labels), random_state=42)

def take(lst, idxs): return [lst[i] for i in idxs]

X_test_text = take(texts,  test_idx)
X_test_pl   = take(places, test_idx)
X_test_ty   = take(types_, test_idx)
y_test      = take(labels, test_idx)

dev_texts  = take(texts,  dev_idx)
dev_places = take(places, dev_idx)
dev_types  = take(types_, dev_idx)
dev_labels = take(labels, dev_idx)

# ==================== CV RUN ====================
def run_cv_once(cfg):
    """
    4-fold CV on the dev set (80%): each fold corresponds to 60/20 of WHOLE for train/val.
    Returns:
      metrics_list: list of dicts (acc, precision, recall, f1) per fold (best epoch by early stopping)
      best_state:   state dict of best fold model by val F1 (for test inference)
    """
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    metrics_list = []
    best_state = None
    best_f1 = -np.inf

    for tr_idx, va_idx in skf.split(dev_texts, dev_labels):
        X_tr_text = [dev_texts[i]  for i in tr_idx]
        X_tr_pl   = [dev_places[i] for i in tr_idx]
        X_tr_ty   = [dev_types[i]  for i in tr_idx]
        y_tr      = [dev_labels[i] for i in tr_idx]

        X_va_text = [dev_texts[i]  for i in va_idx]
        X_va_pl   = [dev_places[i] for i in va_idx]
        X_va_ty   = [dev_types[i]  for i in va_idx]
        y_va      = [dev_labels[i] for i in va_idx]

        # Train-only upsampling
        MIN = 6
        df_tr = pd.DataFrame({'text': X_tr_text, 'place': X_tr_pl, 'type': X_tr_ty, 'label': y_tr})
        vc = df_tr['label'].value_counts()
        rare = vc[vc < MIN].index
        if len(rare) > 0:
            ups = [resample(df_tr[df_tr['label'] == c], replace=True, n_samples=MIN, random_state=42) for c in rare]
            df_tr = pd.concat([df_tr[~df_tr['label'].isin(rare)]] + ups).sample(frac=1, random_state=42)

        X_tr_text = df_tr['text'].tolist()
        X_tr_pl   = df_tr['place'].tolist()
        X_tr_ty   = df_tr['type'].tolist()
        y_tr      = df_tr['label'].tolist()

        ds_tr = make_hf_dataset(X_tr_text, X_tr_pl, X_tr_ty, y_tr)
        ds_va = make_hf_dataset(X_va_text, X_va_pl, X_va_ty, y_va)

        train_dl = DataLoader(CustomDataset(ds_tr), batch_size=BATCH_TRAIN, shuffle=True,  collate_fn=collate_fn)
        val_dl   = DataLoader(CustomDataset(ds_va), batch_size=BATCH_EVAL,  shuffle=False, collate_fn=collate_fn)

        model = RoBERTa_Model('roberta-base', n_labels, n_place, n_type,
                              emb_dim=cfg['emb_dim'], dropout=cfg['dropout']).to(device)

        weight_vec = make_class_weights(y_tr, n_labels)
        criterion = nn.CrossEntropyLoss(weight=weight_vec)
        optimizer = AdamW(model.parameters(), lr=cfg['lr'])

        stopper = EarlyStopper(patience=cfg['patience'], min_delta=1e-4)
        for _ in range(cfg['epochs']):
            _ = train_epoch(model, train_dl, optimizer, criterion)
            _, v_preds, v_true = eval_epoch(model, val_dl)
            v_metrics = metrics_four(v_true, v_preds)
            _, should_stop = stopper.step(v_metrics['f1'], model)
            if should_stop: break
        stopper.restore(model)

        # Take best-epoch metrics for this fold
        _, v_preds, v_true = eval_epoch(model, val_dl)
        v_metrics = metrics_four(v_true, v_preds)
        metrics_list.append(v_metrics)

        if v_metrics['f1'] > best_f1:
            best_f1 = v_metrics['f1']
            best_state = stopper.best_state

    return metrics_list, best_state

def summarize_metrics(metrics_list):
    acc = np.array([m['acc'] for m in metrics_list])
    prc = np.array([m['precision'] for m in metrics_list])
    rec = np.array([m['recall'] for m in metrics_list])
    f1  = np.array([m['f1'] for m in metrics_list])
    return {
        'acc_mean': acc.mean(), 'acc_std': acc.std(),
        'p_mean': prc.mean(),   'p_std':  prc.std(),
        'r_mean': rec.mean(),   'r_std':  rec.std(),
        'f1_mean': f1.mean(),   'f1_std': f1.std()
    }

# ==================== BASELINE CV ====================
baseline_metrics_list, baseline_best_state = run_cv_once(BASE_CFG)
baseline_cv = summarize_metrics(baseline_metrics_list)

# ==================== BASELINE TEST ====================
ds_te = make_hf_dataset(X_test_text, X_test_pl, X_test_ty, y_test)
test_dl = DataLoader(CustomDataset(ds_te), batch_size=BATCH_EVAL, shuffle=False, collate_fn=collate_fn)

baseline_model = RoBERTa_Model('roberta-base', n_labels, n_place, n_type,
                               emb_dim=BASE_CFG['emb_dim'], dropout=BASE_CFG['dropout']).to(device)
if baseline_best_state is not None:
    baseline_model.load_state_dict(baseline_best_state)

_, te_preds_base, te_true = eval_epoch(baseline_model, test_dl)
baseline_test = metrics_four(te_true, te_preds_base)

# ==================== TUNING: CV over PARAM_GRID ====================
best_cfg = None
best_cfg_stats = None
best_cfg_state = None
best_f1_mean = -np.inf

for cfg in PARAM_GRID:
    metrics_list, state = run_cv_once(cfg)
    stats = summarize_metrics(metrics_list)
    if stats['f1_mean'] > best_f1_mean:
        best_f1_mean = stats['f1_mean']
        best_cfg = cfg
        best_cfg_stats = stats
        best_cfg_state = state

# ==================== TUNED TEST (use best cfg + best fold snapshot) ====================
tuned_model = RoBERTa_Model('roberta-base', n_labels, n_place, n_type,
                            emb_dim=best_cfg['emb_dim'], dropout=best_cfg['dropout']).to(device)
if best_cfg_state is not None:
    tuned_model.load_state_dict(best_cfg_state)

_, te_preds_tuned, te_true2 = eval_epoch(tuned_model, test_dl)
tuned_test = metrics_four(te_true2, te_preds_tuned)

# ==================== PRINT EXACTLY FOUR TIMES ====================
# 1) Baseline — CV (mean ± std)
print("BASELINE — CV (dev 80%)")
print(f"Accuracy: {baseline_cv['acc_mean']:.4f} ± {baseline_cv['acc_std']:.4f}")
print(f"Precision: {baseline_cv['p_mean']:.4f} ± {baseline_cv['p_std']:.4f}")
print(f"Recall: {baseline_cv['r_mean']:.4f} ± {baseline_cv['r_std']:.4f}")
print(f"F1: {baseline_cv['f1_mean']:.4f} ± {baseline_cv['f1_std']:.4f}")

# 2) Baseline — Final Test (fixed 20%)
print("\nBASELINE — Final Test (fixed 20%)")
print(f"Accuracy: {baseline_test['acc']:.4f}")
print(f"Precision: {baseline_test['precision']:.4f}")
print(f"Recall: {baseline_test['recall']:.4f}")
print(f"F1: {baseline_test['f1']:.4f}")

# 3) Tuned — CV (mean ± std)
print("\nTUNED — CV (dev 80%) [best config]")
print(f"Accuracy: {best_cfg_stats['acc_mean']:.4f} ± {best_cfg_stats['acc_std']:.4f}")
print(f"Precision: {best_cfg_stats['p_mean']:.4f} ± {best_cfg_stats['p_std']:.4f}")
print(f"Recall: {best_cfg_stats['r_mean']:.4f} ± {best_cfg_stats['r_std']:.4f}")
print(f"F1: {best_cfg_stats['f1_mean']:.4f} ± {best_cfg_stats['f1_std']:.4f}")

# 4) Tuned — Final Test (fixed 20%)
print("\nTUNED — Final Test (fixed 20%)")
print(f"Accuracy: {tuned_test['acc']:.4f}")
print(f"Precision: {tuned_test['precision']:.4f}")
print(f"Recall: {tuned_test['recall']:.4f}")
print(f"F1: {tuned_test['f1']:.4f}")
