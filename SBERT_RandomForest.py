"""
MSc Dissertation — SBERT_RandomForest.py

SBERT embeddings + Random Forest, evaluated with 5-fold cross-validation exactly like
Traditional_ML.py:
- Each fold: 80% development (training) and 20% held-out test
- SBERT vectorization happens inside the pipeline (no leakage)
- Metrics: Accuracy, Precision, Recall, F1 (weighted) reported as mean ± std
- Two blocks printed: Baseline 5-fold and Tuned 5-fold

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
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
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

# ==================== TEXT PROCESSING (kept aligned with your code) ====================
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

    # SBERT-ready text (lowercased, normalized)
    rows['sbert_text'] = rows['capture_remark_updated'].str.replace('NAME', '', regex=False).apply(
        lambda x: re.sub(r'\s+', ' ', str(x).lower().strip()) if isinstance(x, str) else x
    )

    rows.drop(['custom_stopwords_extracted', 'capture_remark_updated'], axis=1, inplace=True)
    rows = rows.applymap(lambda x: re.sub(r'\s+', ' ', x.lower().strip()) if isinstance(x, str) else x)
    return rows

# ==================== PREP DATA ====================
data_all = first_preprocessing(df).astype(str)

# Encode like Traditional_ML.py
le_place = LabelEncoder()
le_type  = LabelEncoder()
le_y     = LabelEncoder()

data_all['defect_place_encoded'] = le_place.fit_transform(data_all['Defect Place'])
data_all['defect_type_encoded']  = le_type.fit_transform(data_all['Defect Type'])
data_all['y']                    = le_y.fit_transform(data_all['Precise Defect Description'])

X_df = pd.DataFrame({
    'sbert_text': data_all['sbert_text'],
    'defect_place_encoded': data_all['defect_place_encoded'].astype(int),
    'defect_type_encoded':  data_all['defect_type_encoded'].astype(int),
})
y = data_all['y'].astype(int).to_numpy()
n_classes = len(le_y.classes_)

# ==================== SBERT TRANSFORMER ====================
class SBERTVectorizer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that turns a column of text into SBERT embeddings.
    Loaded once per (cloned) instance; participates cleanly in CV/Pipelines.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=256, normalize=False, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device  # None -> auto
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def fit(self, X, y=None):
        self._ensure_model()
        return self

    def transform(self, X):
        # ColumnTransformer passes a 2D array for a single column
        if isinstance(X, (pd.Series, pd.DataFrame)):
            texts = pd.Series(X.squeeze()).astype(str).tolist()
        else:
            arr = np.array(X)
            if arr.ndim == 2 and arr.shape[1] == 1:
                texts = [str(t) for t in arr[:, 0]]
            else:
                texts = [str(t) for t in arr]
        model = self._ensure_model()
        vecs = model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=self.normalize)
        return vecs  # (n_samples, dim)

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

def print_block(title, s):
    print(title)
    print(f"Accuracy:  {s['acc_mean']:.4f} ± {s['acc_std']:.4f}")
    print(f"Precision: {s['precision_mean']:.4f} ± {s['precision_std']:.4f}")
    print(f"Recall:    {s['recall_mean']:.4f} ± {s['recall_std']:.4f}")
    print(f"F1:        {s['f1_mean']:.4f} ± {s['f1_std']:.4f}\n")

# ==================== RANDOM FOREST: BASELINE ====================
rf_base_est = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
pipe_base = Pipeline(steps=[('pre', preprocess), ('clf', rf_base_est)])

base_cv = cross_validate(pipe_base, X_df, y, cv=cv5, scoring=scoring, n_jobs=-1, verbose=0)
base_stats = summarize(base_cv)

# ==================== RANDOM FOREST: TUNING ====================
pipe_tune = Pipeline(steps=[('pre', preprocess),
                            ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))])

param_dist = {
    'clf__n_estimators':     np.arange(200, 1001, 100),
    'clf__max_depth':        [None, 10, 20, 30, 40],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf':  [1, 2, 4],
    'clf__bootstrap':        [True, False],
    'clf__class_weight':     [None, 'balanced', 'balanced_subsample'],
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
search.fit(X_df, y)
best_pipe = search.best_estimator_

best_cv = cross_validate(best_pipe, X_df, y, cv=cv5, scoring=scoring, n_jobs=-1, verbose=0)
best_stats = summarize(best_cv)

# ==================== PRINT (two blocks, aligned with Traditional_ML.py) ====================
print("SBERT + RANDOM FOREST — 5-fold (baseline)")
print(f"Accuracy:  {base_stats['acc_mean']:.4f} ± {base_stats['acc_std']:.4f}")
print(f"Precision: {base_stats['precision_mean']:.4f} ± {base_stats['precision_std']:.4f}")
print(f"Recall:    {base_stats['recall_mean']:.4f} ± {base_stats['recall_std']:.4f}")
print(f"F1:        {base_stats['f1_mean']:.4f} ± {base_stats['f1_std']:.4f}\n")

print("SBERT + RANDOM FOREST — 5-fold (tuned)")
print(f"Accuracy:  {best_stats['acc_mean']:.4f} ± {best_stats['acc_std']:.4f}")
print(f"Precision: {best_stats['precision_mean']:.4f} ± {best_stats['precision_std']:.4f}")
print(f"Recall:    {best_stats['recall_mean']:.4f} ± {best_stats['recall_std']:.4f}")
print(f"F1:        {best_stats['f1_mean']:.4f} ± {best_stats['f1_std']:.4f}\n")
