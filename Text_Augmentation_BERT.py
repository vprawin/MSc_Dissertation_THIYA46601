"""
MSc Dissertation — Text_Augmentation_BERT.py

Text augmentation pipeline for BERT classifier with standardized
training/evaluation protocol

Author: Prawin Thiyagrajan Veeramani
Prepared on: 2025-08-26
"""

# ==================== LIBRARIES ====================
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from collections import Counter
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import resample

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from transformers import BertTokenizerFast, BertModel

import nlpaug.augmenter.word as naw

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
df = pd.read_excel(r'/Users/prawin/Desktop/MSc Data Science/Dissertation/Dataset_Sampled.xlsx')
df.columns = ['Defect Place','Defect Type','Capture Remark','Precise Defect Description']

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
            processed.append("NAME")
        else:
            processed.append(lemmatizer.lemmatize(stemmer.stem(token)))
    text = ' '.join(processed)
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def first_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """
    - Anonymize user IDs like q123abc → 'userid'
    - Build stopwords from Excel/CSV
    - Produce: capture_description_bert_processed (lowercased, space-normalized)
    """
    rows = data.copy()
    rows['capture_remark_updated'] = rows['Capture Remark'].str.replace(
        r"[qQ][a-zA-Z0-9]{6}", 'userid', regex=True
    )

    custom_stopwords = pd.read_excel(
        r'/Users/prawin/Desktop/MSc Data Science/Dissertation/250524_ACTIVE_USERS_v1.xlsx'
    )
    second_stopwords = pd.read_csv(
        r'/Users/prawin/Desktop/MSc Data Science/Dissertation/Domain Word Counts Exception.csv'
    )
    filtered_words = second_stopwords[second_stopwords['Relevant'] == 'N']

    custom_names = custom_stopwords['SPERSNAME'].astype(str).tolist()
    additional = filtered_words['Column'].astype(str).tolist()
    new_stopwords = set(custom_names + additional)
    new_stopwords.difference_update(['-', 'biw', '1', 'area', 'on', 'road', 'test', 'for', 'rework'])

    rows['custom_stopwords_extracted'] = rows['capture_remark_updated'].apply(
        lambda x: label_custom_stopwords(x, new_stopwords, stemmer, lemmatizer)
    )
    rows['capture_description_processed'] = rows['custom_stopwords_extracted'].apply(lambda x: x.replace('NAME', ''))
    rows['capture_description_bert_processed'] = (
        rows['capture_remark_updated'].str.replace('NAME', '', regex=False)
        .apply(lambda x: re.sub(r'\s+', ' ', str(x).lower().strip()))
    )

    rows.drop(['custom_stopwords_extracted', 'capture_remark_updated'], axis=1, inplace=True)
    rows = rows.applymap(lambda x: re.sub(r'\s+', ' ', x.lower().strip()) if isinstance(x, str) else x)
    return rows

data_all = first_preprocessing(df).astype(str)

# ==================== ENCODING ====================
le_label = LabelEncoder()
data_all['label_enc'] = le_label.fit_transform(data_all['Precise Defect Description'])
texts_all = data_all['capture_description_bert_processed'].tolist()
labels_all = data_all['label_enc'].tolist()

n_classes = len(le_label.classes_)

# ==================== TOKENIZER ====================
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

def make_loader_from_texts(texts, labels, batch_size, shuffle=True):
    enc = tokenize_texts(texts)
    y = np.asarray(labels, dtype=np.int64)
    ds = list(zip(enc['input_ids'], y, enc['attention_mask']))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# ==================== MODEL ====================
class BERTClassifier(nn.Module):
    """Plain BERT classifier (text-only)."""
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.classifier(self.dropout(pooled))

# ==================== METRICS & HELPERS ====================
def metrics_four(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {'acc': acc, 'precision': p, 'recall': r, 'f1': f}

def make_class_weights(y_train, n_classes):
    """Safe per-fold class weights; zero for absent classes."""
    counts = np.bincount(np.array(y_train, dtype=np.int64), minlength=n_classes)
    weights = np.zeros(n_classes, dtype=np.float32)
    nonzero = counts > 0
    if nonzero.any():
        weights[nonzero] = counts.sum() / (n_classes * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float, device=device)

class EarlyStopper:
    def __init__(self, patience=2, min_delta=1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.best = -np.inf; self.counter = 0; self.best_state = None
    def step(self, metric, model):
        improved = metric > (self.best + self.min_delta)
        if improved:
            self.best = metric; self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return improved, self.counter >= self.patience
    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

def train_epoch(model, loader, optimizer, criterion, max_grad_norm=1.0):
    model.train(); total = 0.0
    for input_ids, labels, attention_mask in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for input_ids, labels, attention_mask in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_t.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

# ==================== AUGMENTATION (train-only) ====================
AUG = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=3)

def to_bert_text(s: str) -> str:
    s = str(s).replace('NAME', '')
    s = re.sub(r'\s+', ' ', s.lower().strip())
    return s

def augment_rare_classes(train_texts, train_labels, min_samples=6, per_source=2):
    """
    For labels with < min_samples in the *current training split*, generate `per_source`
    synonym-augmented sentences per original sample. Returns extended texts/labels.
    """
    texts = list(train_texts)
    labels = list(train_labels)

    counts = Counter(labels)
    rare = [cls for cls, c in counts.items() if c < min_samples]
    if not rare:
        return texts, labels

    # Build a DataFrame for convenience
    df_tr = pd.DataFrame({'text': texts, 'label': labels})
    aug_rows = []
    for cls in rare:
        df_cls = df_tr[df_tr['label'] == cls]
        for _, row in df_cls.iterrows():
            for _ in range(per_source):
                aug_text = to_bert_text(AUG.augment(row['text']))
                aug_rows.append({'text': aug_text, 'label': cls})

    if aug_rows:
        df_aug = pd.DataFrame(aug_rows)
        df_out = pd.concat([df_tr, df_aug], ignore_index=True).sample(frac=1, random_state=42)
        return df_out['text'].tolist(), df_out['label'].tolist()
    return texts, labels

# ==================== CONFIGS ====================
BATCH_TRAIN = 16
BATCH_EVAL  = 32

BASE_CFG = dict(
    lr=2e-5,
    dropout=0.3,
    epochs=8,
    patience=2,
    min_samples=6,     # for augmentation trigger
    per_source=2       # augmentations per rare example
)

# Small grid for tuning
PARAM_GRID = [
    dict(lr=2e-5, dropout=0.3, epochs=8, patience=2, min_samples=6, per_source=2),
    dict(lr=3e-5, dropout=0.1, epochs=8, patience=2, min_samples=6, per_source=2),
]

# ==================== SPLIT: 80/20 (dev/test) ====================
idx_all = np.arange(len(labels_all))
dev_idx, test_idx = train_test_split(idx_all, test_size=0.20, stratify=np.array(labels_all), random_state=42)

def take(lst, idxs): return [lst[i] for i in idxs]

X_test_text = take(texts_all,  test_idx)
y_test      = take(labels_all, test_idx)

dev_texts  = take(texts_all,  dev_idx)
dev_labels = take(labels_all, dev_idx)

# ==================== CV RUN ====================
def run_cv_once(cfg):
    """
    4-fold CV on dev set (80%): each fold => 60/20 of WHOLE for train/val.
    Train-only augmentation is applied inside each fold *only to the training split*.
    Returns:
      metrics_list: list of dicts (acc, precision, recall, f1) per fold
      best_state:   model state dict of best fold by val F1 (for test inference)
    """
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    metrics_list = []
    best_state = None
    best_f1 = -np.inf

    for tr_idx, va_idx in skf.split(dev_texts, dev_labels):
        X_tr = [dev_texts[i]  for i in tr_idx]
        y_tr = [dev_labels[i] for i in tr_idx]
        X_va = [dev_texts[i]  for i in va_idx]
        y_va = [dev_labels[i] for i in va_idx]

        # --- Train-only augmentation for rare classes ---
        X_tr_aug, y_tr_aug = augment_rare_classes(X_tr, y_tr, min_samples=cfg['min_samples'], per_source=cfg['per_source'])

        # Loaders
        train_loader = make_loader_from_texts(X_tr_aug, y_tr_aug, batch_size=BATCH_TRAIN, shuffle=True)
        val_loader   = make_loader_from_texts(X_va,    y_va,    batch_size=BATCH_EVAL,  shuffle=False)

        # Model / optimizer / criterion
        model = BERTClassifier(num_classes=n_classes, dropout=cfg['dropout']).to(device)
        weight_vec = make_class_weights(y_tr_aug, n_classes)
        criterion = nn.CrossEntropyLoss(weight=weight_vec)
        optimizer = AdamW(model.parameters(), lr=cfg['lr'])

        # Early stopping on val F1
        stopper = EarlyStopper(patience=cfg['patience'], min_delta=1e-4)
        for _ in range(cfg['epochs']):
            _ = train_epoch(model, train_loader, optimizer, criterion)
            val_preds, val_true = eval_epoch(model, val_loader)
            vm = metrics_four(val_true, val_preds)
            _, should_stop = stopper.step(vm['f1'], model)
            if should_stop: break
        stopper.restore(model)

        # Final fold metrics (best epoch)
        val_preds, val_true = eval_epoch(model, val_loader)
        vm = metrics_four(val_true, val_preds)
        metrics_list.append(vm)

        if vm['f1'] > best_f1:
            best_f1 = vm['f1']
            best_state = stopper.best_state

    return metrics_list, best_state

def summarize_metrics(metrics_list):
    a = np.array([m['acc'] for m in metrics_list])
    p = np.array([m['precision'] for m in metrics_list])
    r = np.array([m['recall'] for m in metrics_list])
    f = np.array([m['f1'] for m in metrics_list])
    return {
        'acc_mean': a.mean(), 'acc_std': a.std(),
        'p_mean': p.mean(),   'p_std':  p.std(),
        'r_mean': r.mean(),   'r_std':  r.std(),
        'f1_mean': f.mean(),  'f1_std': f.std(),
    }

# ==================== BASELINE CV ====================
baseline_metrics_list, baseline_best_state = run_cv_once(BASE_CFG)
baseline_cv = summarize_metrics(baseline_metrics_list)

# ==================== BASELINE TEST (use best fold snapshot) ====================
test_loader = make_loader_from_texts(X_test_text, y_test, batch_size=BATCH_EVAL, shuffle=False)
baseline_model = BERTClassifier(num_classes=n_classes, dropout=BASE_CFG['dropout']).to(device)
if baseline_best_state is not None:
    baseline_model.load_state_dict(baseline_best_state)

te_preds_base, te_true = eval_epoch(baseline_model, test_loader)
baseline_test = metrics_four(te_true, te_preds_base)

# ==================== TUNING: CV over PARAM_GRID ====================
best_cfg = None
best_cfg_stats = None
best_cfg_state = None
best_f1_mean = -np.inf

for cfg in PARAM_GRID:
    mlist, state = run_cv_once(cfg)
    stats = summarize_metrics(mlist)
    if stats['f1_mean'] > best_f1_mean:
        best_f1_mean = stats['f1_mean']
        best_cfg = cfg
        best_cfg_stats = stats
        best_cfg_state = state

# ==================== TUNED TEST (use best cfg + best fold snapshot) ====================
tuned_model = BERTClassifier(num_classes=n_classes, dropout=best_cfg['dropout']).to(device)
if best_cfg_state is not None:
    tuned_model.load_state_dict(best_cfg_state)

te_preds_tuned, te_true2 = eval_epoch(tuned_model, test_loader)
tuned_test = metrics_four(te_true2, te_preds_tuned)

# ==================== PRINT ====================
print("BASELINE — CV (dev 80%)")
print(f"Accuracy: {baseline_cv['acc_mean']:.4f} ± {baseline_cv['acc_std']:.4f}")
print(f"Precision: {baseline_cv['p_mean']:.4f} ± {baseline_cv['p_std']:.4f}")
print(f"Recall: {baseline_cv['r_mean']:.4f} ± {baseline_cv['r_std']:.4f}")
print(f"F1: {baseline_cv['f1_mean']:.4f} ± {baseline_cv['f1_std']:.4f}")

print("\nBASELINE — Final Test (fixed 20%)")
print(f"Accuracy: {baseline_test['acc']:.4f}")
print(f"Precision: {baseline_test['precision']:.4f}")
print(f"Recall: {baseline_test['recall']:.4f}")
print(f"F1: {baseline_test['f1']:.4f}")

print("\nTUNED — CV (dev 80%) [best config]")
print(f"Accuracy: {best_cfg_stats['acc_mean']:.4f} ± {best_cfg_stats['acc_std']:.4f}")
print(f"Precision: {best_cfg_stats['p_mean']:.4f} ± {best_cfg_stats['p_std']:.4f}")
print(f"Recall: {best_cfg_stats['r_mean']:.4f} ± {best_cfg_stats['r_std']:.4f}")
print(f"F1: {best_cfg_stats['f1_mean']:.4f} ± {best_cfg_stats['f1_std']:.4f}")

print("\nTUNED — Final Test (fixed 20%)")
print(f"Accuracy: {tuned_test['acc']:.4f}")
print(f"Precision: {tuned_test['precision']:.4f}")
print(f"Recall: {tuned_test['recall']:.4f}")
print(f"F1: {tuned_test['f1']:.4f}")
