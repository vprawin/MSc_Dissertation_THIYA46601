"""
MSc Dissertation — Traditional_ML.py

Traditional machine-learning baselines (Logistic Regression, RandomForest, XGBoost)
with cross-validation and hyperparameter tuning aligned.

Author: Prawin Thiyagrajan Veeramani
Prepared on: 2025-08-26
"""

# ========== IMPORTS ==========
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTEN

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from scipy.stats import uniform
import optuna
import numpy as np

# ========== DATA LOAD ==========
df = pd.read_excel(r'/Users/prawin/Desktop/MSc Data Science/Dissertation/Dataset.xlsx')
df.columns = ['Defect Place', 'Defect Type', 'Capture Remark', 'Precise Defect Description']

# ========== TEXT PROCESSING FUNCTIONS (do not modify) ==========
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

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
    """first_preprocessing — one-line summary.

    Args:
        data: Description.

    Returns:
        Description of return value.
    """
    filtered_rows = data.copy()
    filtered_rows['capture_remark_updated'] = filtered_rows['Capture Remark'].str.replace(r"[qQ][a-zA-Z0-9]{6}", 'userid', regex=True)

    custom_stopwords = pd.read_excel(r'/Users/prawin/Desktop/MSc Data Science/Dissertation/250524_ACTIVE_USERS_v1.xlsx')
    second_stopwords = pd.read_csv(r'/Users/prawin/Desktop/MSc Data Science/Dissertation/Domain Word Counts Exception.csv')
    filtered_words = second_stopwords[second_stopwords['Relevant'] == 'N']

    regex = r"[qQ][a-zA-Z0-9]{6}"
    custom_stopwords = custom_stopwords[custom_stopwords['SPERS_IDNR'].str.match(regex)]
    custom_stopwords_list = custom_stopwords['SPERSNAME'].tolist()
    additional_stopwords = filtered_words['Column'].tolist()

    new_stopwords = set(custom_stopwords_list + additional_stopwords)
    new_stopwords.difference_update(['-', 'biw', '1', 'area', 'on', 'road', 'test', 'for', 'rework'])

    filtered_rows['custom_stopwords_extracted'] = filtered_rows['capture_remark_updated'].apply(
        lambda x: label_custom_stopwords(x, new_stopwords, stemmer, lemmatizer)
    )
    filtered_rows['capture_description_processed'] = filtered_rows['custom_stopwords_extracted'].apply(lambda x: x.replace('NAME', ''))
    filtered_rows['capture_description_bert_processed'] = filtered_rows['capture_remark_updated'].str.replace(
        'NAME', '', regex=False
    ).apply(lambda x: re.sub(r'\s+', ' ', str(x).lower().strip()) if isinstance(x, str) else x)

    filtered_rows.drop(['custom_stopwords_extracted', 'capture_remark_updated'], axis=1, inplace=True)
    filtered_rows = filtered_rows.applymap(lambda x: re.sub(r'\s+', ' ', x.lower().strip()) if isinstance(x, str) else x)
    return filtered_rows

def boost_words(text, words_to_boost, factor):
    """boost_words — one-line summary.

    Args:
        text: Description.
        words_to_boost: Description.
        factor: Description.

    Returns:
        Description of return value.
    """
    for word in words_to_boost:
        pattern = r'(?<!\w)' + re.escape(word) + r'(?!\w)'
        replacement = (word + ' ') * factor
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

# ========== DATA PROCESSING ==========
train_data = first_preprocessing(df)
train_data = train_data.astype(str)

# Encode categorical columns
label_encoder_place = LabelEncoder()
label_encoder_type = LabelEncoder()
label_encoder_pdd = LabelEncoder()

train_data['defect_place_encoded'] = label_encoder_place.fit_transform(train_data['Defect Place'])
train_data['defect_type_encoded'] = label_encoder_type.fit_transform(train_data['Defect Type'])
train_data['pdd_encoded'] = label_encoder_pdd.fit_transform(train_data['Precise Defect Description'])

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=500)
tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['capture_description_processed'])

# Feature combination
X = hstack([train_data[['defect_place_encoded', 'defect_type_encoded']].values, tfidf_matrix])
y = train_data['pdd_encoded'].values

# Oversample very rare classes (<6)
label_counts = pd.Series(y).value_counts()
rare_classes = label_counts[label_counts < 6].index.tolist()
sampling_strategy = {cls: 6 for cls in rare_classes}

ros_pre = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
X_ros, y_ros = ros_pre.fit_resample(X, y)

# SMOTEN + ROS for final balancing
pipeline = Pipeline(steps=[
    ('ros', RandomOverSampler(random_state=42)),
    ('smoten', SMOTEN(k_neighbors=5, random_state=42))
])
X_resampled, y_resampled = pipeline.fit_resample(X_ros, y_ros)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# Stratified 5-fold (fixed for all CV to align LaTeX)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ========== BASELINES (CV) ==========
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_scores = cross_val_score(log_reg, X_train, y_train, cv=cv, scoring='accuracy')

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')

xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='accuracy')

print("===== BASELINE CROSS-VALIDATION (mean ± std) =====")
print("Logistic Regression (CV) Accuracy: {:.4f} ± {:.4f}".format(log_scores.mean(), log_scores.std()))
print("Random Forest      (CV) Accuracy: {:.4f} ± {:.4f}".format(rf_scores.mean(), rf_scores.std()))
print("XGBoost           (CV) Accuracy: {:.4f} ± {:.4f}".format(xgb_scores.mean(), xgb_scores.std()))

# ========== LOGISTIC REGRESSION TUNING ==========
# Scale sparse matrix safely
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)

# Use two distributions to avoid invalid (penalty, l1_ratio) combos
logreg_param_distributions = [
    {   # L1/L2 (no l1_ratio)
        'penalty': ['l1', 'l2'],
        'C': uniform(1e-3, 10),
        'solver': ['saga'],
        'max_iter': [2000]
    },
    {   # ElasticNet (with l1_ratio in [0,1])
        'penalty': ['elasticnet'],
        'C': uniform(1e-3, 10),
        'l1_ratio': uniform(0.0, 1.0),
        'solver': ['saga'],
        'max_iter': [2000]
    }
]

search_logreg = RandomizedSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_distributions=logreg_param_distributions,
    n_iter=30,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=3,
    random_state=42
)
search_logreg.fit(X_train_scaled, y_train)
tuned_logreg = search_logreg.best_estimator_

tuned_log_scores = cross_val_score(tuned_logreg, X_train_scaled, y_train, cv=cv, scoring='accuracy')

print("\n===== LOGISTIC REGRESSION TUNING =====")
print("Best Logistic Regression Params:", search_logreg.best_params_)
print("Best Logistic Regression CV Score: {:.4f}".format(search_logreg.best_score_))
print("Tuned Logistic Regression (CV) Accuracy: {:.4f} ± {:.4f}".format(tuned_log_scores.mean(), tuned_log_scores.std()))

# ========== RANDOM FOREST TUNING (OPTUNA) ==========
def objective_rf(trial):
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int('n_estimators', 100, 300),
        max_depth=trial.suggest_int('max_depth', 5, 30),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 4),
        bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
        random_state=42,
        n_jobs=-1
    )
    return cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=30)
print("\n===== RANDOM FOREST TUNING =====")
print("Best Random Forest Params:", study_rf.best_params)

best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42, n_jobs=-1)
rf_tuned_scores = cross_val_score(best_rf, X_train, y_train, cv=cv, scoring='accuracy')
print("Tuned Random Forest (CV) Accuracy: {:.4f} ± {:.4f}".format(rf_tuned_scores.mean(), rf_tuned_scores.std()))

# ========== XGBOOST TUNING (OPTUNA) ==========
def objective_xgb(trial):
    model = xgb.XGBClassifier(
        n_estimators=trial.suggest_int('n_estimators', 100, 300),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
    return cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=30)
print("\n===== XGBOOST TUNING =====")
print("Best XGBoost Params:", study_xgb.best_params)

best_xgb = xgb.XGBClassifier(**study_xgb.best_params, use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
xgb_tuned_scores = cross_val_score(best_xgb, X_train, y_train, cv=cv, scoring='accuracy')
print("Tuned XGBoost (CV) Accuracy: {:.4f} ± {:.4f}".format(xgb_tuned_scores.mean(), xgb_tuned_scores.std()))

# ========== FINAL TEST-SET EVALUATION ==========
print("\n===== TEST-SET EVALUATION =====")
# Logistic Regression
X_test_scaled = scaler.transform(X_test)
tuned_logreg.fit(X_train_scaled, y_train)
y_pred_logreg = tuned_logreg.predict(X_test_scaled)
print("Logistic Regression Test Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_logreg)))
print("Logistic Regression Test Classification Report:\n", classification_report(y_test, y_pred_logreg))

# Random Forest
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)
print("Random Forest Test Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_rf)))
print("Random Forest Test Classification Report:\n", classification_report(y_test, y_pred_rf))

# XGBoost
best_xgb.fit(X_train, y_train)
y_pred_xgb = best_xgb.predict(X_test)
print("XGBoost Test Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_xgb)))
print("XGBoost Test Classification Report:\n", classification_report(y_test, y_pred_xgb))
