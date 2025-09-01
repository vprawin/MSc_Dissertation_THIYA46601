"""
MSc Dissertation — SBERT_RandomForest.py

SBERT embeddings + Random Forest with 80/20 fixed test-set evaluation, mirroring Traditional_ML.py:
1) Stratified 80/20 split (20% = final test)
2) 5-fold CV on the 80% train only (baseline + tuned) → print mean ± std (Acc/Prec/Rec/F1 weighted)
3) Refit baseline pipeline on 80% train and evaluate on 20% test (single metrics + classification report)
4) Refit tuned pipeline on 80% train and evaluate on 20% test (single metrics + classification report)

Author: Prawin Thiyagrajan Veeramani
Prepared on: 2025-08-26
"""

# ==================== IMPORTS ====================
import re
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier

from sentence_transformers import SentenceTransformer

# ==================== SEED ====================
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)

# ==================== DATA LOAD ====================
df = pd.read_excel(r'/Users/prawin/Desktop/MSc Data Science/Dissertation/Dataset_Sampled.xlsx')
df.columns = ['Defect Place', 'Defect Type', 'Capture Remark', 'Precise Defect Description']

# ==================== TEXT PROCESSING ====================
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def label_custom_stopwords(description, stopwords_set, stemmer, lemmatizer, similarity_threshold=0.7):
    """(Kept as-is to align with your existing scripts.)"""
    description = re.sub(r'\[.*?\]', '', str(description)).lower()
    description_tokens = word_tokenize(description)
    processed_tokens = []
    for token in description_tokens:
        for sw in stopwords_set:
            processed_tokens.append("NAME")
        stemmed = stemmer.stem(token)
        lemmatized = lemmatizer.lemmatize(stemmed)
        processed_tokens.append(lemmatized)
    labeled_description = ' '.join(processed_tokens)
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9 ]', ' ', labeled_description.strip()))

def first_preprocessing(data):
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

    regex = r"[qQ][a-zA-Z0-9]{6}"
    custom_stopwords = custom_stopwords[custom_stopwords['SPERS_IDNR'].astype(str).str.match(regex, na=False)]
    custom_stopwords_list = custom_stopwords['SPERSNAME'].astype(str).tolist()
    additional_stopwords = filtered_words['Column'].astype(str).tolist()

    new_stopwords = set(custom_stopwords_list + additional_stopwords)
    new_stopwords.difference_update(['-', 'biw', '1', 'area', 'on', 'road', 'test', 'for', 'rework'])

    rows['custom_stopwords_extracted'] = rows['capture_remark_updated'].apply(
        lambda x: label_custom_stopwords(x, new_stopwords, stemmer, lemmatizer)
    )
    rows['capture_description_processed'] = rows['custom_stopwords_extracted'].apply(lambda x: x.replace('NAME', ''))

    rows['sbert_text'] = rows['capture_remark_updated'].str.replace('NAME', '', regex=False).apply(
        lambda x: re.sub(r'\s+', ' ', str(x).lower().strip()) if isinstance(x, str) else x
    )

    rows.drop(['custom_stopwords_extracted', 'capture_remark_updated'], axis=1, inplace=True)
    rows = rows.applymap(lambda x: re.sub(r'\s+', ' ', x.lower().strip()) if isinstance(x, str) else x)
    return rows

# ==================== PREP DATA ====================
data_all = first_preprocessing(df).astype(str)

le_place = LabelEncoder()
le_type  = LabelEncoder()
le_y     = LabelEncoder()

data_all['defect_place_encoded'] = le_place.fit_transform(data_all['Defect Place'])
data_all['defect_type_encoded']  = le_type.fit_transform(data_all['Defect Type'])
data_all['y']                    = le_y.fit_transform(data_all['Precise Defect Description'])

X_df_full = pd.DataFrame({
    'sbert_text': data_all['sbert_text'],
    'defect_place_encoded': data_all['defect_place_encoded'].astype(int),
    'defect_type_encoded':  data_all['defect_type_encoded'].astype(int),
})
y_full = data_all['y'].astype(int).to_numpy()
n_classes = len(le_y.classes_)

# ===== 80/20 fixed test split =====
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df_full, y_full, test_size=0.20, stratify=y_full, random_state=42
)

# ==================== SBERT TRANSFORMER ====================
class SBERTVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=256, normalize=False, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def fit(self, X, y=None):
        self._ensure_model()
        return self

    def transform(self, X):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            texts = pd.Series(X.squeeze()).astype(str).tolist()
        else:
            arr = np.array(X)
            texts = [str(t) for t in (arr[:, 0] if arr.ndim == 2 and arr.shape[1] == 1 else arr)]
        model = self._ensure_model()
        return model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=self.normalize)

# ==================== PREPROCESSOR & SCORING ====================
preprocess = ColumnTransformer(
    transformers=[
        ('sbert', SBERTVectorizer(), 'sbert_text'),
        ('num', 'passthrough', ['defect_place_encoded', 'defect_type_encoded']),
    ]
)

scoring = {
    'acc': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall':    make_scorer(recall_score,    average='weighted', zero_division=0),
    'f1':        make_scorer(f1_score,        average='weighted', zero_division=0),
}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def summarize(cvres):
    return {
        'acc_mean':       np.mean(cvres['test_acc']),       'acc_std':       np.std(cvres['test_acc']),
        'precision_mean': np.mean(cvres['test_precision']), 'precision_std': np.std(cvres['test_precision']),
        'recall_mean':    np.mean(cvres['test_recall']),    'recall_std':    np.std(cvres['test_recall']),
        'f1_mean':        np.mean(cvres['test_f1']),        'f1_std':        np.std(cvres['test_f1']),
    }

# ==================== RF: BASELINE (CV on 80%) ====================
rf_base_est = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1,
    bootstrap=True, class_weight='balanced_subsample', random_state=42, n_jobs=-1
)
pipe_base = Pipeline(steps=[('pre', preprocess), ('clf', rf_base_est)])
base_cv = cross_validate(pipe_base, X_train_df, y_train, cv=cv5, scoring=scoring, n_jobs=-1, verbose=0)
base_stats = summarize(base_cv)

# ==================== RF: TUNING (CV on 80%) ====================
pipe_tune = Pipeline(steps=[('pre', preprocess), ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))])
param_dist = {
    'clf__n_estimators':      np.arange(200, 1001, 100),
    'clf__max_depth':         [None, 10, 20, 30, 40],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf':  [1, 2, 4],
    'clf__bootstrap':         [True, False],
    'clf__class_weight':      [None, 'balanced', 'balanced_subsample'],
}
search = RandomizedSearchCV(
    estimator=pipe_tune,
    param_distributions=param_dist,
    n_iter=40,
    cv=cv5,
    scoring='f1_weighted',
    refit=True,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
search.fit(X_train_df, y_train)
best_pipe = search.best_estimator_

best_cv = cross_validate(best_pipe, X_train_df, y_train, cv=cv5, scoring=scoring, n_jobs=-1, verbose=0)
best_stats = summarize(best_cv)

# ==================== PRINT: CV blocks ====================
print("SBERT + RANDOM FOREST — 5-fold on 80% train (baseline)")
print(f"Accuracy:  {base_stats['acc_mean']:.4f} ± {base_stats['acc_std']:.4f}")
print(f"Precision: {base_stats['precision_mean']:.4f} ± {base_stats['precision_std']:.4f}")
print(f"Recall:    {base_stats['recall_mean']:.4f} ± {base_stats['recall_std']:.4f}")
print(f"F1:        {base_stats['f1_mean']:.4f} ± {base_stats['f1_std']:.4f}\n")

print("SBERT + RANDOM FOREST — 5-fold on 80% train (tuned)")
print(f"Accuracy:  {best_stats['acc_mean']:.4f} ± {best_stats['acc_std']:.4f}")
print(f"Precision: {best_stats['precision_mean']:.4f} ± {best_stats['precision_std']:.4f}")
print(f"Recall:    {best_stats['recall_mean']:.4f} ± {best_stats['recall_std']:.4f}")
print(f"F1:        {best_stats['f1_mean']:.4f} ± {best_stats['f1_std']:.4f}\n")

# ==================== FINAL TEST-SET EVALUATION (on fixed 20%) ====================
# --- Baseline pipeline on test ---
pipe_base.fit(X_train_df, y_train)
y_pred_test_base = pipe_base.predict(X_test_df)
print("SBERT + RANDOM FOREST — FINAL TEST (fixed 20%) — BASELINE")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_test_base):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test_base, average='weighted', zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_test_base, average='weighted', zero_division=0):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_test_base, average='weighted', zero_division=0):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_test_base, zero_division=0))

# --- Tuned pipeline on test ---
best_pipe.fit(X_train_df, y_train)
y_pred_test_tuned = best_pipe.predict(X_test_df)
print("\nSBERT + RANDOM FOREST — FINAL TEST (fixed 20%) — TUNED")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_test_tuned):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test_tuned, average='weighted', zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_test_tuned, average='weighted', zero_division=0):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_test_tuned, average='weighted', zero_division=0):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_test_tuned, zero_division=0))
