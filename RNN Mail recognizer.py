import pandas as pd
import numpy as np
import chardet
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from textblob import TextBlob
import spacy
nlp = spacy.load("en_core_web_sm")



class TextCleaner:
    def __init__(self, df, column):
        self.df = df.copy()
        self.column = column
        self.df[self.column] = self.df[self.column].astype(str)

    def dropna(self):
        self.df.dropna(inplace=True)
        return self

    def remove_unamed_column(self):
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        return self

    def lowercase(self):
        self.df[self.column] = self.df[self.column].str.lower()
        return self

    def strip_columns(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].astype(str).str.strip()
        return self


    def remove_punctuation(self):
        self.df[self.column] = self.df[self.column].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        return self

    def remove_stopwords(self):
        stopwords = ENGLISH_STOP_WORDS
        self.df[self.column] = self.df[self.column].apply(
            lambda x: ' '.join([word for word in x.split() if word not in stopwords])
        )
        return self

    def remove_numbers(self):
        self.df[self.column] = self.df[self.column].str.replace(r'\d+', '', regex=True)
        return self

    def remove_extra_spaces(self):
        self.df[self.column] = self.df[self.column].apply(lambda x: re.sub(' +', ' ', x))
        return self

    def remove_empty_rows(self):
        self.df = self.df[self.df[self.column] != '']
        return self

    def lemmatize(self):
        texts = list(self.df[self.column])
        lemmatized = []

        # Fast batch processing (disables unnecessary components)
        for doc in nlp.pipe(texts, batch_size=500, disable=["ner", "parser"]):
            lemmatized.append(' '.join([token.lemma_ for token in doc]))

        self.df[self.column] = lemmatized
        return self


    def remove_emojis(self):
        self.df[self.column] = self.df[self.column].apply(
            lambda x: x.encode('ascii', 'ignore').decode('ascii')
        )
        return self

    def remove_linebreaks(self):
      self.df[self.column] = self.df[self.column].str.replace(r'[\r\n]+', ' ', regex=True)
      return self

    def remove_email_headers(self):
      self.df[self.column] = self.df[self.column].str.replace(r'(?i)^subject:', '', regex=True)
      return self


    def correct_spelling(self):
        self.df[self.column] = self.df[self.column].apply(
            lambda x: str(TextBlob(x).correct())
        )
        return self

    def get(self):
        return self.df


# Tokenize
def tokenize_and_remove_stopwords(text):
    # Tokenize only (text is already lowercase)
    tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text)
    # Remove stopwords
    filtered = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return filtered


# Read all the data
dataframes = []
filenames = ['dataset1.csv','dataset2.csv']

for file in filenames:
    with open(file, 'rb') as f:
        rawdata = f.read(10000)
        result = chardet.detect(rawdata)
        encoding = result['encoding'] if result['encoding'] else 'utf-8'

    try:
        df = pd.read_csv(file, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(file, encoding='ISO-8859-1')  # fallback

    dataframes.append(df)


# Renaming
dataframes[0].drop(columns=['label'], inplace=True)
dataframes[0].rename(columns={'label_num': 'label'}, inplace=True)
dataframes[1].rename(columns={'spam': 'label'}, inplace=True)
"""dataframes[3].rename(columns={'email': 'text'},inplace = True)
dataframes[4].rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
dataframes[4]['label'] = dataframes[4]['label'].map({'ham': 0, 'spam': 1})"""

# Cleaning DataFrames
for i in range(len(dataframes)):
    cleaner = (
        TextCleaner(dataframes[i], 'text')
        .dropna()
        .remove_unamed_column()
        .strip_columns()
        .lowercase()
        .remove_linebreaks()
        .remove_email_headers()
        #.remove_emojis()
        .remove_punctuation()
        .remove_numbers()
        .remove_stopwords()
        .remove_extra_spaces()
        .remove_empty_rows()
    )
    dataframes[i] = cleaner.get()

# Concatenate all dataframes using only 'text' and 'label' columns
final_df = pd.concat(
    [df[['text', 'label']] for df in dataframes if 'text' in df.columns and 'label' in df.columns],
    ignore_index=True
)

# Split to train and test!
X_train, X_test, y_train, y_test = train_test_split(
    final_df['text'], final_df['label'], test_size=0.2, random_state=42,stratify=final_df['label']
)

# Tokenize + remove stopwords
X_train_text = X_train.apply(lambda x: ' '.join(tokenize_and_remove_stopwords(x)))
X_test_text = X_test.apply(lambda x: ' '.join(tokenize_and_remove_stopwords(x)))

class SpamDataset(Dataset):
    def __init__(self, texts, labels, word2idx):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx].split()
        indices = [self.word2idx.get(word, 0) for word in words]
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded.contiguous(), torch.tensor(labels)



# Step 1: Initialize and reserve padding token
word2idx = defaultdict(lambda: len(word2idx))
word2idx['<PAD>'] = 0

# Step 2: Build vocab from training data
for text in X_train_text:
    for word in text.split():
        _ = word2idx[word]

# (Optional) Freeze it
word2idx = dict(word2idx)
train_dataset = SpamDataset(X_train_text.tolist(), y_train.tolist(), word2idx)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)                     # [batch, seq_len, emb_dim]
        out, hidden = self.rnn(embedded.contiguous())    # hidden: [num_layers, batch, hidden_dim]
        return self.fc(hidden[-1])                       # final output: [batch, output_dim]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNModel(len(word2idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("✅ Preprocessing complete. Starting training...\n")
for epoch in range(2):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Print every 5 batches
        if (batch_idx + 1) % 5 == 0:
            print(f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_loader)}")

    print(f"✅ Epoch {epoch+1} finished. Total loss: {total_loss:.4f}\n")

# Create test dataset and loader
test_dataset = SpamDataset(X_test_text.tolist(), y_test.tolist(), word2idx)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
model.eval()
all_preds = []
all_labels = []

model.eval()
all_preds = []
all_labels = []

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs.contiguous())
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# Report
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=["Ham", "Spam"]))
