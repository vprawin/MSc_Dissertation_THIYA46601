"""MSc Dissertation — Text_Augmentation_BERT.py

    Text augmentation utilities tailored for transformer-based workflows.

    This file is prepared for publication on GitHub (appendix reference). It adds clear, standardized
    docstrings while preserving original behavior.

    Author: Prawin Thiyagrajan Veeramani
    Prepared on: 2025-08-26
    """

# ----------- Standard Libraries for Data Processing & Visualization ----------- #
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# ----------- NLTK for Text Normalization ----------- #
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# ----------- Imbalanced Handling & Encoding ----------- #
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTEN
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

# ----------- Load Raw Dataset ----------- #
df = pd.read_excel(r'/Users/prawin/Desktop/MSc Data Science/Dissertation/Dataset_Sampled.xlsx')

# Rename long column names for easier reference
df.columns = [
    'Defect Place',
    'Defect Type',
    'Capture Remark',
    'Precise Defect Description'
]

# ----------- Initialize Lemmatizer and Stemmer ----------- #
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# ----------- Function to Replace Custom Stopwords and Normalize Text (PATCHED) ----------- #
def label_custom_stopwords(description, stopwords_set, stemmer, lemmatizer, similarity_threshold=0.7):
    """label_custom_stopwords — one-line summary.

    Args:
        description: Description.
        stopwords_set: Description.
        stemmer: Description.
        lemmatizer: Description.
        similarity_threshold: Description.

    Returns:
        Description of return value.
    """
    # Clean brackets and lowercase
    description = re.sub(r'\[.*?\]', '', str(description)).lower()
    # Tokenize text
    description_tokens = word_tokenize(description)
    processed_tokens = []
    for token in description_tokens:
        if token in stopwords_set:  # PATCH: membership check once
            processed_tokens.append("NAME")
        else:
            stemmed = stemmer.stem(token)
            lemmatized = lemmatizer.lemmatize(stemmed)
            processed_tokens.append(lemmatized)
    # Return cleaned string
    labeled_description = ' '.join(processed_tokens)
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9 ]', ' ', labeled_description.strip()))

# ----------- Full Preprocessing Pipeline ----------- #
def first_preprocessing(data):
    """first_preprocessing — one-line summary.

    Args:
        data: Description.

    Returns:
        Description of return value.
    """
    filtered_rows = data.copy()

    # Anonymize user IDs like q123abc → 'userid'
    filtered_rows['capture_remark_updated'] = filtered_rows['Capture Remark'].str.replace(
        r"[qQ][a-zA-Z0-9]{6}", 'userid', regex=True
    )

    # Load stopword reference files
    custom_stopwords = pd.read_excel(
        r'/Users/prawin/Desktop/MSc Data Science/Dissertation/250524_ACTIVE_USERS_v1.xlsx'
    )
    second_stopwords = pd.read_csv(
        r'/Users/prawin/Desktop/MSc Data Science/Dissertation/Domain Word Counts Exception.csv'
    )
    filtered_words = second_stopwords[second_stopwords['Relevant'] == 'N']

    # Filter valid SPERS_IDs using regex
    regex = r"[qQ][a-zA-Z0-9]{6}"
    custom_stopwords = custom_stopwords[custom_stopwords['SPERS_IDNR'].astype(str).str.match(regex, na=False)]

    # Merge user-based and domain-based stopwords
    custom_stopwords_list = custom_stopwords['SPERSNAME'].astype(str).tolist()
    additional_stopwords = filtered_words['Column'].astype(str).tolist()
    new_stopwords = set(custom_stopwords_list + additional_stopwords)

    # Remove useful domain words from stopword list
    new_stopwords.difference_update(['-', 'biw', '1', 'area', 'on', 'road', 'test', 'for', 'rework'])

    # Apply stopword replacement and remove 'NAME' placeholders
    filtered_rows['custom_stopwords_extracted'] = filtered_rows['capture_remark_updated'].apply(
        lambda x: label_custom_stopwords(x, new_stopwords, stemmer, lemmatizer)
    )
    filtered_rows['capture_description_processed'] = filtered_rows['custom_stopwords_extracted'].apply(
        lambda x: x.replace('NAME', '')
    )

    # Clean text for BERT-based processing
    filtered_rows['capture_description_bert_processed'] = filtered_rows['capture_remark_updated'].str.replace(
        'NAME', '', regex=False
    ).apply(lambda x: re.sub(r'\s+', ' ', str(x).lower().strip()) if isinstance(x, str) else x)

    # Drop intermediate columns
    filtered_rows.drop(['custom_stopwords_extracted', 'capture_remark_updated'], axis=1, inplace=True)

    # Apply final whitespace and casing cleanup
    filtered_rows = filtered_rows.applymap(
        lambda x: re.sub(r'\s+', ' ', x.lower().strip()) if isinstance(x, str) else x
    )

    return filtered_rows

# ----------- Apply Preprocessing ----------- #
train_data = first_preprocessing(df)
train_data = train_data.astype(str)  # Ensure all columns are string for NLP handling

# ------------------ Re-imports Required for Next Sections ------------------ #
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertModel
import nlpaug.augmenter.word as naw
from sklearn.preprocessing import label_binarize
from sklearn.utils import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    roc_auc_score,
    roc_curve
)
import json

# ----------- Encode Final Label ----------- #
le_label = LabelEncoder()
train_data['label_enc'] = le_label.fit_transform(train_data['Precise Defect Description'])
label_names = list(le_label.classes_)
report_labels = list(range(len(label_names)))  # PATCH: ensure labels align with target_names
print("Label classes:\n", label_names)

# ----------- Identify Rare Classes (less than 6 samples) ----------- #
min_samples = 6
label_counts = train_data['label_enc'].value_counts()
rare_classes = label_counts[label_counts < min_samples].index.tolist()

# ----------- Synonym Augmentation for Rare Classes ----------- #
aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=3)
augmented = []

for cls in rare_classes:
    df_cls = train_data[train_data['label_enc'] == cls]
    for _, row in df_cls.iterrows():
        for _ in range(2):  # Create two augmentations per row
            # AUGMENT raw remark text, then clean it consistently for BERT
            aug_text = aug.augment(str(row['Capture Remark']))
            augmented.append({
                'Defect Place': row['Defect Place'],
                'Defect Type': row['Defect Type'],
                'Capture Remark': aug_text,
                'Precise Defect Description': row['Precise Defect Description'],
                'label_enc': cls
            })

# ----------- Combine Augmented Data ----------- #
aug_df = pd.DataFrame(augmented)

# Build a unified BERT text column for BOTH original and augmented rows
def to_bert_text(s):
    """to_bert_text — one-line summary.

    Args:
        s: Description.

    Returns:
        Description of return value.
    """
    s = str(s).replace('NAME', '')
    s = re.sub(r'\s+', ' ', s.lower().strip())
    return s

train_data['bert_text'] = train_data['capture_description_bert_processed'].apply(to_bert_text)
if not aug_df.empty:
    aug_df['bert_text'] = aug_df['Capture Remark'].astype(str).apply(to_bert_text)

# Concat and shuffle
train_data_aug = pd.concat(
    [
        train_data[['Defect Place','Defect Type','Capture Remark','Precise Defect Description','label_enc','bert_text']],
        aug_df[['Defect Place','Defect Type','Capture Remark','Precise Defect Description','label_enc','bert_text']]
        if not aug_df.empty else []
    ],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

# ----------- Load BERT Tokenizer ----------- #
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize unified BERT text
train_data_aug['bert_text'] = train_data_aug['bert_text'].astype(str).fillna("")
tokens = tokenizer(
    train_data_aug['bert_text'].tolist(),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'  # Returns tensors directly
)

# Split into input IDs and attention masks
X = tokens['input_ids']
A = tokens['attention_mask']
y = np.array(train_data_aug['label_enc'].tolist())

# ----------- BERT Classifier Definition ----------- #
class BERTClassifier(nn.Module):
    """BERTClassifier — class summary and usage notes."""
    def __init__(self, num_classes):
        """__init__ — one-line summary.

    Args:
        num_classes: Description.

    Returns:
        Description of return value.
    """
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """forward — one-line summary.

    Args:
        input_ids: Description.
        attention_mask: Description.

    Returns:
        Description of return value.
    """
        pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.classifier(self.dropout(pooled))

# ----------- Device and (global) Placeholders ----------- #
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")

# ----------- DataLoader Construction ----------- #
def make_loader(X, y, A, batch_size, shuffle=True):
    """make_loader — one-line summary.

    Args:
        X: Description.
        y: Description.
        A: Description.
        batch_size: Description.
        shuffle: Description.

    Returns:
        Description of return value.
    """
    # zip order: (input_ids, labels, attention_mask)
    return DataLoader(list(zip(X, y, A)), batch_size=batch_size, shuffle=shuffle)

# ----------- Training Epoch Function ----------- #
def train_epoch(model, loader):
    """train_epoch — one-line summary.

    Args:
        model: Description.
        loader: Description.

    Returns:
        Description of return value.
    """
    model.train()
    total_loss = 0
    for input_ids, labels, attention_mask in tqdm(loader, desc="Training"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stability
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

# ----------- Evaluation Function ----------- #
def eval_epoch(model, loader):
    """eval_epoch — one-line summary.

    Args:
        model: Description.
        loader: Description.

    Returns:
        Description of return value.
    """
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for input_ids, labels, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    return all_preds, all_labels, all_logits

# ----------- Micro-Average ROC Curve (helper, optional) ----------- #
def plot_micro_roc(y_true, y_score, class_names):
    """plot_micro_roc — one-line summary.

    Args:
        y_true: Description.
        y_score: Description.
        class_names: Description.

    Returns:
        Description of return value.
    """
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fpr, tpr, _ = roc_curve(y_bin.ravel(), np.array(y_score).ravel())  # micro-average curve
    weighted_auc = roc_auc_score(y_bin, y_score, average='weighted', multi_class='ovr')
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Micro-average ROC (Weighted AUC = {weighted_auc:.2f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-average ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return weighted_auc

# ----------- Cross Validation (Stratified K-Fold) ----------- #
from sklearn.model_selection import StratifiedKFold

K = 5
epochs = 4
fold_metrics = []

skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n========== Fold {fold}/{K} ==========")

    # Split tensors
    X_train, X_val = X[tr_idx], X[va_idx]
    A_train, A_val = A[tr_idx], A[va_idx]
    y_train, y_val = y[tr_idx], y[va_idx]

    # Fresh model, optimizer, class weights per fold (set as globals for train_epoch to use)
    model = BERTClassifier(len(label_names)).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    cw = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=device))

    # DataLoaders
    train_loader = make_loader(X_train, y_train, A_train, batch_size=16, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   A_val,   batch_size=32, shuffle=False)

    # Train
    for epoch in range(epochs):
        print(f"\n--- Fold {fold} • Epoch {epoch+1}/{epochs} ---")
        train_loss = train_epoch(model, train_loader)
        preds, labels, logits = eval_epoch(model, val_loader)

        acc = accuracy_score(labels, preds)
        f1w = f1_score(labels, preds, average='weighted')
        f1m = f1_score(labels, preds, average='macro')

        print(f"Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | F1_w: {f1w:.4f} | F1_m: {f1m:.4f}")

    # Final eval for this fold
    preds, labels, logits = eval_epoch(model, val_loader)
    acc = accuracy_score(labels, preds)
    f1w = f1_score(labels, preds, average='weighted')
    f1m = f1_score(labels, preds, average='macro')

    print("\nClassification Report:")
    print(classification_report(
        labels, preds,
        labels=report_labels,        # align number of labels with target_names
        target_names=label_names,
        zero_division=0
    ))

    fold_metrics.append({'fold': fold, 'acc': acc, 'f1_weighted': f1w, 'f1_macro': f1m})

# CV summary
accs = [m['acc'] for m in fold_metrics]
f1ws = [m['f1_weighted'] for m in fold_metrics]
f1ms = [m['f1_macro'] for m in fold_metrics]

print("\n========== CV Summary ==========")
print(f"Accuracy:      {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1 (weighted): {np.mean(f1ws):.4f} ± {np.std(f1ws):.4f}")
print(f"F1 (macro):    {np.mean(f1ms):.4f} ± {np.std(f1ms):.4f}")

# (Optional) ROC for the last fold predictions
try:
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    auc_weighted = plot_micro_roc(labels, probs, label_names)
    print(f"\nLast-Fold Weighted ROC-AUC: {auc_weighted:.4f}")
except Exception as e:
    print("ROC plotting skipped due to:", e)
