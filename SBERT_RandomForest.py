"""MSc Dissertation — SBERT_RandomForest.py

    Sentence-BERT embedding pipeline with a RandomForest classifier.

    This file is prepared for publication on GitHub (appendix reference). It adds clear, standardized
    docstrings while preserving original behavior.

    Author: Prawin Thiyagrajan Veeramani
    Prepared on: 2025-08-26
    """

# ==================== IMPORTS ====================
import pandas as pd
import numpy as np
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.utils import resample, compute_class_weight
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==================== LOAD & PREPROCESS ====================
df = pd.read_excel('/Users/prawin/Desktop/MSc Data Science/Dissertation/Dataset_Sampled.xlsx')
df.columns = ['Defect Place', 'Defect Type', 'Capture Remark', 'Precise Defect Description']

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

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
    description = re.sub(r'\[.*?\]', '', str(description)).lower()
    tokens = word_tokenize(description)
    processed = []
    for token in tokens:
        for _ in stopwords_set:
            processed.append("NAME")
        processed.append(lemmatizer.lemmatize(stemmer.stem(token)))
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9 ]', ' ', ' '.join(processed).strip()))

def first_preprocessing(data):
    """first_preprocessing — one-line summary.

    Args:
        data: Description.

    Returns:
        Description of return value.
    """
    data = data.copy()
    data['capture_remark_updated'] = data['Capture Remark'].str.replace(r"[qQ][a-zA-Z0-9]{6}", 'userid', regex=True)
    custom_stopwords = pd.read_excel('/Users/prawin/Desktop/MSc Data Science/Dissertation/250524_ACTIVE_USERS_v1.xlsx')
    second_stopwords = pd.read_csv('/Users/prawin/Desktop/MSc Data Science/Dissertation/Domain Word Counts Exception.csv')
    new_stopwords = set(custom_stopwords['SPERSNAME'].tolist() + second_stopwords[second_stopwords['Relevant'] == 'N']['Column'].tolist())
    new_stopwords.difference_update(['-', 'biw', '1', 'area', 'on', 'road', 'test', 'for', 'rework'])
    data['custom_stopwords_extracted'] = data['capture_remark_updated'].apply(lambda x: label_custom_stopwords(x, new_stopwords, stemmer, lemmatizer))
    data['capture_description_processed'] = data['custom_stopwords_extracted'].apply(lambda x: x.replace('NAME', ''))
    data['capture_description_bert_processed'] = data['capture_remark_updated'].str.replace('NAME', '', regex=False).apply(lambda x: re.sub(r'\s+', ' ', str(x).lower().strip()))
    data.drop(['custom_stopwords_extracted', 'capture_remark_updated'], axis=1, inplace=True)
    return data.applymap(lambda x: re.sub(r'\s+', ' ', x.lower().strip()) if isinstance(x, str) else x)

train_data = first_preprocessing(df).astype(str)

# ==================== UPSAMPLING ====================
min_samples = 6
label_counts = train_data['Precise Defect Description'].value_counts()
rare_classes = label_counts[label_counts < min_samples].index
resampled = [resample(train_data[train_data['Precise Defect Description'] == cls], replace=True, n_samples=min_samples, random_state=42) for cls in rare_classes]
train_data = pd.concat([train_data[~train_data['Precise Defect Description'].isin(rare_classes)]] + resampled).sample(frac=1, random_state=42)

# ==================== ENCODING ====================
le_place = LabelEncoder()
le_type = LabelEncoder()
le_label = LabelEncoder()
train_data['type_enc'] = le_type.fit_transform(train_data['Defect Type'])
train_data['place_enc'] = le_place.fit_transform(train_data['Defect Place'])
train_data['label_enc'] = le_label.fit_transform(train_data['Precise Defect Description'])

# ==================== SBERT EMBEDDINGS ====================
print("Encoding remarks with SBERT...")
sbert = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = sbert.encode(train_data['Capture Remark'].tolist(), show_progress_bar=True)

# ==================== FEATURE MATRIX ====================
X_structured = train_data[['type_enc', 'place_enc']].values
X = np.hstack([X_structured, text_embeddings])
y = train_data['label_enc'].values

# ==================== CLASS WEIGHTS ====================
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# ==================== 5-FOLD CROSS-VALIDATION ====================
print("\nRunning 5-Fold Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = []
cv_auc = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"\nFold {fold+1}")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    y_prob = rf.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    y_val_bin = label_binarize(y_val, classes=np.arange(len(np.unique(y))))
    roc_auc = roc_auc_score(y_val_bin, y_prob, average='weighted', multi_class='ovr')

    print(f"Fold {fold+1} Accuracy: {acc:.4f}")
    print(f"Fold {fold+1} Weighted ROC-AUC: {roc_auc:.4f}")
    cv_acc.append(acc)
    cv_auc.append(roc_auc)

print("\nCross-Validation Summary:")
print(f"Mean Accuracy: {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")
print(f"Mean Weighted ROC-AUC: {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")
