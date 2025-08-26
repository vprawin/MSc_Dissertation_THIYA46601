"""MSc Dissertation — BERT_Model_3.py

    BERT model variant #3: training script and helper functions.

    This file is prepared for publication on GitHub (appendix reference). It adds clear, standardized
    docstrings while preserving original behavior.

    Author: Prawin Thiyagrajan Veeramani
    Prepared on: 2025-08-26
    """

# ==================== LIBRARIES ====================
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertModel
from datasets import Dataset
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import math
import numpy as np

# ==================== LOAD & RENAME DATA ====================
df = pd.read_excel('/Users/prawin/Desktop/MSc Data Science/Dissertation/Dataset_Sampled.xlsx')
df.columns = ['Defect Place', 'Defect Type', 'Capture Remark', 'Precise Defect Description']

# ==================== TEXT PREPROCESSING ====================
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
    custom_stopwords = pd.read_excel('/Users/prawin/Desktop/MSc Data Science/Dissertation/250524_ACTIVE_USERS_v1.xlsx')
    second_stopwords = pd.read_csv('/Users/prawin/Desktop/MSc Data Science/Dissertation/Domain Word Counts Exception.csv')
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

train_data = first_preprocessing(df).astype(str)

# ==================== RARE CLASS UPSAMPLING ====================
min_samples = 6
label_counts = train_data['Precise Defect Description'].value_counts()
rare_classes = label_counts[label_counts < min_samples].index
resampled_frames = [resample(train_data[train_data['Precise Defect Description'] == cls],
                             replace=True, n_samples=min_samples, random_state=42)
                    for cls in rare_classes]
df_majority = train_data[~train_data['Precise Defect Description'].isin(rare_classes)]
train_data = pd.concat([df_majority] + resampled_frames).sample(frac=1, random_state=42).reset_index(drop=True)

# ==================== ENCODING ====================
texts = train_data['Capture Remark'].tolist()
place_encoded = LabelEncoder().fit_transform(train_data['Defect Place'].astype(str))
type_encoded = LabelEncoder().fit_transform(train_data['Defect Type'].astype(str))
labels_encoded = LabelEncoder().fit_transform(train_data['Precise Defect Description'])

# ==================== SPLIT ====================
# ==================== TRAIN-VALID SPLIT (updated: 5-fold outer, 60/20 inner) ====================
# We pick the FIRST fold's development split and create a 60/20 (train/val) from its 80%,
# leaving the 20% test fold aside (no other code changes needed).

texts_np        = np.array(texts)
place_enc_np    = np.array(place_encoded)
type_enc_np     = np.array(type_encoded)
labels_enc_np   = np.array(labels_encoded)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for dev_idx, test_idx in skf.split(texts_np, labels_enc_np):
    # 80% development portion for this fold
    dev_texts   = texts_np[dev_idx]
    dev_place   = place_enc_np[dev_idx]
    dev_type    = type_enc_np[dev_idx]
    dev_labels  = labels_enc_np[dev_idx]

    # Inner split on the dev set: 60% train / 20% val overall (i.e., 75/25 of dev)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, val_idx = next(sss.split(dev_texts, dev_labels))

    train_texts = dev_texts[tr_idx]
    val_texts   = dev_texts[val_idx]

    train_place = dev_place[tr_idx]
    val_place   = dev_place[val_idx]

    train_type  = dev_type[tr_idx]
    val_type    = dev_type[val_idx]

    train_labels = dev_labels[tr_idx]
    val_labels   = dev_labels[val_idx]

    # stop after preparing the first fold (keeps the rest of your code unchanged)
    break

# ==================== TOKENIZATION ====================
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
def tokenize(batch): return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)
train_dataset = Dataset.from_dict({'text': train_texts, 'place': train_place, 'type': train_type, 'labels': train_labels}).map(tokenize, batched=True)
val_dataset = Dataset.from_dict({'text': val_texts, 'place': val_place, 'type': val_type, 'labels': val_labels}).map(tokenize, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'place', 'type'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'place', 'type'])

# ==================== MODEL ====================
class BERT_Model_3(nn.Module):
    """tokenize — one-line summary.

    Args:
        batch), padding=True, truncation=True, max_length=128)
train_dataset = Dataset.from_dict({'text', 'place', 'type', 'labels', batched=True)
val_dataset = Dataset.from_dict({'text', 'place', 'type', 'labels', batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'place', 'type'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'place', 'type'])

# ==================== MODEL ===================: Description.

    Returns:
        Description of return value.
    """
    def __init__(self, bert_model_name, num_labels, num_place, num_type, emb_dim=32):
        """__init__ — one-line summary.

    Args:
        bert_model_name: Description.
        num_labels: Description.
        num_place: Description.
        num_type: Description.
        emb_dim: Description.

    Returns:
        Description of return value.
    """
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)  # Trainable
        self.place_emb = nn.Embedding(num_place, emb_dim)
        self.type_emb = nn.Embedding(num_type, emb_dim)
        self.dropout = nn.Dropout(0.3)
        combined_dim = self.bert.config.hidden_size + emb_dim * 2
        self.classifier = nn.Linear(combined_dim, num_labels)

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
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        combined = torch.cat((pooled_output, self.place_emb(place), self.type_emb(type_)), dim=1)
        return self.classifier(self.dropout(combined))

# ==================== COLLATOR & DATASET WRAPPER ====================
def collate_fn(batch):
    """collate_fn — one-line summary.

    Args:
        batch: Description.

    Returns:
        Description of return value.
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    place = torch.tensor([item['place'] for item in batch], dtype=torch.long)
    type_ = torch.tensor([item['type'] for item in batch], dtype=torch.long)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    batch_encoding = tokenizer.pad({'input_ids': input_ids, 'attention_mask': attention_mask}, return_tensors='pt')
    return {
        'input_ids': batch_encoding['input_ids'],
        'attention_mask': batch_encoding['attention_mask'],
        'place': place,
        'type': type_,
        'labels': labels
    }

class CustomDataset(torch.utils.data.Dataset):
    """CustomDataset — class summary and usage notes."""
    def __init__(self, hf_dataset): self.dataset = hf_dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        """__init__ — one-line summary.

    Args:
        hf_dataset)= hf_dataset
    def __len__(self), idx: Description.

    Returns:
        Description of return value.
    """
        item = self.dataset[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': item['labels'],
            'place': torch.tensor(item['place'], dtype=torch.long),
            'type': torch.tensor(item['type'], dtype=torch.long)
        }

train_dl = DataLoader(CustomDataset(train_dataset), batch_size=16, shuffle=True, collate_fn=collate_fn)
val_dl = DataLoader(CustomDataset(val_dataset), batch_size=32, collate_fn=collate_fn)

# ==================== TRAINING SETUP ====================
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = BERT_Model_3(
    'bert-base-uncased',
    num_labels=len(set(labels_encoded)),
    num_place=len(set(place_encoded)),
    num_type=len(set(type_encoded))
).to(device)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_encoded), y=labels_encoded)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, dataloader):
    """train_epoch — one-line summary.

    Args:
        model: Description.
        dataloader: Description.

    Returns:
        Description of return value.
    """
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        place = batch['place'].to(device)
        type_ = batch['type'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask, place, type_)
        loss = criterion(outputs, labels)
        loss.backward()
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
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            place = batch['place'].to(device)
            type_ = batch['type'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask, place, type_)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

PATIENCE = 2
# ==================== RUN TRAINING (Epochs = 4) ====================
# ==================== RUN TRAINING with EARLY STOP ====================
epochs = 4
best_val = -math.inf
best_state = None
no_improve = 0

for epoch in range(epochs):
    train_loss = train_epoch(model, train_dl)
    val_acc = eval_epoch(model, val_dl)
    print(f"[BERT_Model_1] Epoch {epoch+1}: train loss = {train_loss:.4f}, val accuracy = {val_acc:.4f}")

    # --- Early stopping on validation accuracy ---
    if val_acc > best_val:
        best_val = val_acc
        no_improve = 0
        # store a CPU copy so it’s safe to reload on any device
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping triggered (no improvement for {PATIENCE} epoch(s)).")
            break

# restore best weights before any final evaluation or saving
if best_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
print(f"Best validation accuracy: {best_val:.4f}")
