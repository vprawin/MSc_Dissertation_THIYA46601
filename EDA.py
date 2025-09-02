"""MSc Dissertation — EDA.py

    Exploratory Data Analysis utilities and plots for the dissertation dataset.
    
    Author: Prawin Thiyagrajan Veeramani
    Prepared on: 2025-08-26
    """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

from wordcloud import WordCloud, STOPWORDS

# Load dataset
df = pd.read_excel(r'/Users/prawin/Desktop/MSc Data Science/Dissertation/Dataset_Sampled.xlsx')

# Rename columns for ease
df.columns = [
    'Defect Place',
    'Defect Type',
    'Capture Remark',
    'Precise Defect Description'
]

### 1. Summary Info
print("Data Types:\n", df.dtypes)
print("\nShape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nUnique Values:\n", df.nunique())
print("\nShape:\n", df.shape)

### 2. Tokenizer
def simple_tokenize_column(column):
    """simple_tokenize_column — one-line summary.

    Args:
        column: Description.

    Returns:
        Description of return value.
    """
    tokens = []
    for text in column.dropna():
        words = re.findall(r'\b[a-zA-Z]{2,}\b', str(text).lower())
        tokens.extend(words)
    return tokens

tokens_place = simple_tokenize_column(df['Defect Place'])
tokens_type = simple_tokenize_column(df['Defect Type'])
tokens_remark = simple_tokenize_column(df['Capture Remark'])
tokens_precise = simple_tokenize_column(df['Precise Defect Description'])

### 3. Token Frequency Stats
def get_min_max_token_stats(tokens, column_name):
    """get_min_max_token_stats — one-line summary.

    Args:
        tokens: Description.
        column_name: Description.

    Returns:
        Description of return value.
    """
    counter = Counter(tokens)
    most_common = counter.most_common()
    return {
        "Column": column_name,
        "Max Token": most_common[0][0],
        "Max Count": most_common[0][1],
        "Min Token": most_common[-1][0],
        "Min Count": most_common[-1][1]
    }

stats = [
    get_min_max_token_stats(tokens_place, "Defect Place"),
    get_min_max_token_stats(tokens_type, "Defect Type"),
    get_min_max_token_stats(tokens_remark, "Capture Remarks"),
    get_min_max_token_stats(tokens_precise, "Precise Defect Description"),
]

print("\nToken Frequency Stats:")
print(pd.DataFrame(stats).to_string(index=False))

### 4. Token Count per Row
def entry_token_stats(column):
    """entry_token_stats — one-line summary.

    Args:
        column: Description.

    Returns:
        Description of return value.
    """
    token_counts = column.dropna().apply(lambda x: len(re.findall(r'\b[a-zA-Z]{2,}\b', str(x).lower())))
    return token_counts.min(), token_counts.max(), token_counts.mean()

token_range = {
    "Defect Place": entry_token_stats(df['Defect Place']),
    "Defect Type": entry_token_stats(df['Defect Type']),
    "Capture Remarks": entry_token_stats(df['Capture Remark']),
    "Precise Defect Description": entry_token_stats(df['Precise Defect Description']),
}

print("\nToken Count per Entry:")
print(pd.DataFrame(token_range, index=['Min Tokens', 'Max Tokens', 'Average Token']).T)

def count_tokens(text):
    """count_tokens — one-line summary.

    Args:
        text: Description.

    Returns:
        Description of return value.
    """
    if pd.isna(text):
        return 0
    return len(re.findall(r'\b[a-zA-Z]{2,}\b', str(text).lower()))

metrics = {}
for col in df.columns:
    metrics[col] = df[col].apply(count_tokens)

metrics_df = pd.DataFrame(metrics)

plt.figure(figsize=(10,6))
box = plt.boxplot(metrics_df.values,
                  labels=metrics_df.columns,
                  patch_artist=True,
                  showmeans=True)  # adds mean marker

# Custom colors
colors = ['lightblue','lightgreen','lightcoral','lightgoldenrodyellow']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

#plt.title("Boxplot of Token Counts per Column (Min, Max, Median, Mean)")
plt.ylabel("Token Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

### 5. Label Distribution
label_dist = df['Precise Defect Description'].value_counts().reset_index()
label_dist.columns = ['Precise Defect Description', 'Count']
label_dist['Percentage'] = 100 * label_dist['Count'] / label_dist['Count'].sum()

# Vertical bar chart
plt.figure(figsize=(max(12, 0.3 * len(label_dist)), 10))
ax = sns.barplot(data=label_dist, x='Precise Defect Description', y='Count', palette='viridis')
for i, row in label_dist.iterrows():
    ax.text(i, row['Count'] + 5, int(row['Count']), ha='center', fontsize=7, rotation=90)
plt.title('Full Distribution of Precise Defect Descriptions (Vertical)')
plt.xlabel('Defect Description')
plt.ylabel('Count')
plt.xticks(rotation=90, fontsize=7)
plt.tight_layout()
plt.show()

### 6. Bucket Label Frequencies
bins = [0, 10, 50, 100, 200, 500, 750, 1000, 1250]
labels = ['≤10', '<50', '<100', '<200', '<500', '<750', '<1000', '<1250']
label_dist['Bucket'] = pd.cut(label_dist['Count'], bins=bins, labels=labels, right=True)
bucket_counts = label_dist['Bucket'].value_counts().sort_index().reset_index()
bucket_counts.columns = ['Token Count Range', 'Number of Labels']

# Bar chart
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=bucket_counts, x='Token Count Range', y='Number of Labels', palette='plasma')
for i, row in bucket_counts.iterrows():
    ax.text(i, row['Number of Labels'] + 1, int(row['Number of Labels']), ha='center')
plt.title('Label Count Distribution by Frequency Buckets')
plt.xlabel('Token Count Range')
plt.ylabel('Number of Labels')
plt.tight_layout()
plt.show()

### 7. Distribution of Defect Place (Vertical)
place_dist = df['Defect Place'].value_counts().reset_index()
place_dist.columns = ['Defect Place', 'Count']
plt.figure(figsize=(max(12, 0.3 * len(place_dist)), 8))
ax1 = sns.barplot(data=place_dist, x='Defect Place', y='Count', palette='coolwarm')
ax1.set_title('Distribution of Defect Place (Vertical)')
ax1.set_xlabel('Defect Place')
ax1.set_ylabel('Count')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()

### 8. Distribution of Defect Type (Vertical)
type_dist = df['Defect Type'].value_counts().reset_index()
type_dist.columns = ['Defect Type', 'Count']
plt.figure(figsize=(max(10, 0.4 * len(type_dist)), 8))
ax2 = sns.barplot(data=type_dist, x='Defect Type', y='Count', palette='crest')
ax2.set_title('Distribution of Defect Type (Vertical)')
ax2.set_xlabel('Defect Type')
ax2.set_ylabel('Count')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()

### 9. Heatmap of Defect Place vs Defect Type
matrix = pd.crosstab(df['Defect Place'], df['Defect Type'])
matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index,
                    matrix.sum(axis=0).sort_values(ascending=False).index]

plt.figure(figsize=(20, 12))
sns.heatmap(matrix, cmap='viridis', linewidths=0.5, linecolor='gray')
plt.title('Defect Place vs Defect Type - Count Matrix')
plt.xlabel('Defect Type')
plt.ylabel('Defect Place')
plt.tight_layout()
plt.show()

filtered_df = df['Capture Remark'].dropna()

# Step 10: Ensure all values are strings
filtered_df = filtered_df.astype(str)

# Step 11: Combine all remarks into a single string
text = " ".join(filtered_df.tolist())

# Step 12: Generate Word Cloud
wordcloud = WordCloud(stopwords=STOPWORDS, width=800, height=400, background_color='white').generate(text)

# Step 13: Plot the Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



# Step 14: Prepare text
filtered = df['Capture Remark'].dropna().astype(str)
text = " ".join(filtered.tolist()).lower()

# Step 15: Tokenize & clean
words = re.findall(r'\b[a-zA-Z]{2,}\b', text)  # keep only alphabetic words with length >= 2

# Step 16: Remove stopwords and irrelevant terms
custom_stopwords = set(STOPWORDS)
# You can extend with domain-specific stopwords:
custom_stopwords.update(["mm", "na", "nan"])

words = [w for w in words if w not in custom_stopwords]

# Step 17: Frequency count
freq = Counter(words)

# Step 18: Get most frequent, median, and seldom terms
if not freq:
    print("No valid words to analyze.")
else:
    most_common_term, max_freq = freq.most_common(1)[0]

    freqs_sorted = sorted(freq.items(), key=lambda kv: kv[1])
    median_idx = len(freqs_sorted) // 2
    median_term, median_freq = freqs_sorted[median_idx]

    least_common_term, min_freq = freqs_sorted[0]

    print(f"Most frequent term: '{most_common_term}' (count = {max_freq})")
    print(f"Median-frequency term: '{median_term}' (count = {median_freq})")
    print(f"Seldom term: '{least_common_term}' (count = {min_freq})")
