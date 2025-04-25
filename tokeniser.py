import collections
import pickle
import pandas as pd
from tqdm import tqdm  # progress bar
import numpy as np

k = 5  # minimum frequency for words to be included in the vocabulary


def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = text.replace(".", " <PERIOD> ")
    text = text.replace(",", " <COMMA> ")
    text = text.replace('"', " <QUOTATION_MARK> ")
    text = text.replace(";", " <SEMICOLON> ")
    text = text.replace("!", " <EXCLAMATION_MARK> ")
    text = text.replace("?", " <QUESTION_MARK> ")
    text = text.replace("(", " <LEFT_PAREN> ")
    text = text.replace(")", " <RIGHT_PAREN> ")
    text = text.replace("--", " <HYPHENS> ")
    text = text.replace("?", " <QUESTION_MARK> ")
    text = text.replace(":", " <COLON> ")
    return text.split()


def pad_sequences(sequences, max_length=None, padding_value=0):
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)  # Find the longest sequence

    padded_sequences = np.array(
        [
            (
                seq + [padding_value] * (max_length - len(seq))
                if len(seq) < max_length
                else seq[:max_length]
            )
            for seq in sequences
        ]
    )
    return padded_sequences


def build_vocab():
    splits = ["train", "validation", "test"]
    dfs = [pd.read_csv(f"data/queries_{split}.tsv", sep="\t") for split in splits]
    queries_df = (
        pd.concat(dfs).drop_duplicates(subset="query_text").reset_index(drop=True)
    )
    documents_df = pd.read_csv("data/all_documents.tsv", sep="\t")

    print("Building vocabulary...")
    all_words = []

    # Process queries for vocabulary building
    for text in tqdm(queries_df["query_text"], desc="Processing queries for vocab"):
        all_words.extend(preprocess(text))

    # Process documents for vocabulary building
    for text in tqdm(documents_df["doc_text"], desc="Processing documents for vocab"):
        all_words.extend(preprocess(text))

    # Count word frequencies
    word_counts = collections.Counter(all_words)
    print(f"Found {len(word_counts)} unique words")

    # Filter vocabulary (keep words with frequency > 5)
    filtered_vocab = {word for word, count in word_counts.items() if count > k}
    print(f"Filtered to {len(filtered_vocab)} words")

    # Add special tokens
    special_tokens = ["<PAD>", "<UNK>", "<START>", "<END>"]
    vocab = special_tokens + list(filtered_vocab)

    # Create mapping dictionaries
    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_vocab = {i: word for i, word in enumerate(vocab)}

    # Save vocabulary
    with open("vocab_to_int.pkl", "wb") as f:
        pickle.dump(vocab_to_int, f)

    with open("int_to_vocab.pkl", "wb") as f:
        pickle.dump(int_to_vocab, f)

    tokens = [vocab_to_int[word] for word in vocab]
    print(len(tokens))  # 119479

    return vocab_to_int, int_to_vocab


def tokenize(text, vocab_to_int):
    tokens = [vocab_to_int.get(w, vocab_to_int["<UNK>"]) for w in preprocess(text)]
    return tokens
