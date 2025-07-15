# Email Spam Detection using RNN (GRU)

A PyTorch-based RNN model for classifying email messages as spam or non-spam.  
Includes a complete pipeline from raw text preprocessing to model training and evaluation.

## Overview

- Built with `PyTorch` using a GRU-based RNN
- Custom `Dataset` and `collate_fn` for sequence batching
- Preprocessing includes lemmatization, punctuation removal, tokenization, and stopword filtering
- Evaluation using `classification_report` and `confusion_matrix`

## Features

- Clean and modular `TextCleaner` class
- Manual vocabulary construction
- Sequence padding for batching
