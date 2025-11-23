import json
import os
import re
from typing import Optional, List, Tuple, Dict
from unittest.mock import DEFAULT
import numpy as np
import pandas as pd

FILENAME = "training_data_clean.csv"
TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)

TEXT_COLS = [1, 6, 9]  # Free Texts Questions
RATING_COLS = [2, 4, 7, 8]  # Rating Questions
MULTI_COLS = [3, 5]  # Multiplicative choices Questions

CANONICAL_MULTI_TYPES = [
    "Math computations",
    "Writing or debugging code",
    "Data processing or analysis",
    "Explaining complex concepts simply",
    "Writing or editing essays/reports",
    "Brainstorming or generating creative ideas",
    "Drafting professional text (e.g., résumés, emails)",
    "Converting content between formats (e.g., LaTeX)",
]

MULTI_MAP = {
    "math computations": "Math computations",
    "writing or debugging code": "Writing or debugging code",
    "data processing or analysis": "Data processing or analysis",
    "explaining complex concepts simply": "Explaining complex concepts simply",
    "writing or editing essays/reports": "Writing or editing essays/reports",
    "brainstorming or generating creative ideas": "Brainstorming or generating creative ideas",
    "drafting professional text": "Drafting professional text (e.g., résumés, emails)",
    "résumés": "Drafting professional text (e.g., résumés, emails)",
    "emails": "Drafting professional text (e.g., résumés, emails)",
    "converting content between formats": "Converting content between formats (e.g., LaTeX)",
    "latex": "Converting content between formats (e.g., LaTeX)",
}

stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
    'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
    'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've",
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
    'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
    "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 't', 's', 'd',
    'use', 'would', 'have', 'has', 'had', 'do', 'does', 'did', 'can', 'will',
    'with', 'as', 'on', 'by', 'from', 'into', 'through', 'this', 'that',
    'these',
    'those', 'some', 'any', 'all', 'another', 'other', 'very', 'just',
    'so', 'too', 'more', 'most', 'often', 'sometimes', 'usually',
    'what', 'when', 'where', 'why', 'how', 'which', 'not', 'never', 'my',
    'me', 'its', 'them', 'they', 'you'
}


def train_val_test_split(df: pd.DataFrame, train_ratio: float = 0.7,
                         val_ratio: float = 0.15, shuffle: bool = False,
                         random_state: int = 42) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train / val / test = 7 : 1.5 : 1.5
    """
    if shuffle:
        df = df.sample(frac=1.0, random_state=random_state).reset_index(
            drop=True)

    df_train = df.iloc[:576]
    df_val = df.iloc[576:702]
    df_test = df.iloc[702:]

    return df_train, df_val, df_test


def normalize_to_text_list(value) -> List[str]:
    """
    Transform a list of strings in database into a list of strings.
    The values can be list, str, numbers.
    Returns: list of strings or empty list if value is None or Nan
    """
    # check whether the input is None or Nan, handle missing values
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    # nested lists
    if isinstance(value, list):
        tokens = []
        for v in value:
            tokens.extend(normalize_to_text_list(v))  # expand the nest list
        return tokens

    # The normal case: split the long answer
    text = str(value).lower()
    tokens = TOKEN_PATTERN.findall(text)
    return tokens


def build_text_vocabulary(df, text_cols, dict_len: int = None,
                          remove_stopwords: bool = False) -> Dict[str, int]:
    """
    Return the list of unique words for all columns in text_cols, denoting as the vocabulary.

    Args:
        df: pandas DataFrame
        text_cols: list of column indices for text columns
        dict_len: maximum vocabulary size. If None, include all words.
        remove_stopwords: whether to remove common stopwords
        use_custom_stopwords: whether to use the custom stopwords list tailored to your data

    Returns:
        dict of {word: index} for the top dict_len most frequent words
    """
    cache_file = f'voc_dict_{dict_len}.json' if dict_len else 'voc_dict.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        return vocab_dict

    vocabulary = {}
    cols = df.columns

    # Count word frequencies across all text columns
    for col_idx in text_cols:
        col_name = cols[col_idx]
        for value in df[col_name]:
            tokens = normalize_to_text_list(value)
            for token in tokens:
                # Skip stopwords if remove_stopwords is True
                if remove_stopwords and token.lower() in stopwords:
                    continue
                vocabulary[token] = vocabulary.get(token, 0) + 1

    # Sort words by frequency and limit to dict_len
    if dict_len:
        print("yes dict_len")
        sorted_words = sorted(vocabulary.items(), key=lambda x: x[1],
                              reverse=True)[:dict_len]
        words = [word for word, freq in sorted_words]
    else:
        words = sorted(vocabulary.keys())

    # Create word to index mapping
    word_to_index = {word: i for i, word in enumerate(words)}

    # Save to cache
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(word_to_index, f, ensure_ascii=False, indent=2)

    print(f"Vocabulary size after filtering: {len(word_to_index)}")
    return word_to_index


def build_text_matrix(df: pd.DataFrame, text_cols: List[int],
                      vocab: Dict[str, int]) -> np.ndarray:
    """
    Build text feature matrix where each text column gets its own sub-matrix,
    and all sub-matrices are concatenated horizontally.

    Input:
        df: pandas DataFrame
        text_cols: list of int that indicate which columns in df to use
        vocab: dictionary mapping words to indices

    Returns:
        numpy array of shape (num_samples, len(text_cols) * vocab_size)
        Structure: [text_array1 | text_array2 | ... | text_arrayN]
    """
    num_samples = len(df)
    vocab_size = len(vocab)

    # Create separate matrices for each text column
    text_matrices = []
    cols = df.columns

    for col_idx in text_cols:
        # Create bag-of-words matrix for this specific text column
        col_matrix = np.zeros((num_samples, vocab_size), dtype=np.int32)
        col_name = cols[col_idx]

        for row_i, (_, row) in enumerate(df.iterrows()):
            text = row[col_name]
            tokens = normalize_to_text_list(text)

            for token in tokens:
                if token in vocab:
                    vocab_idx = vocab[token]
                    col_matrix[row_i, vocab_idx] += 1

        text_matrices.append(col_matrix)

    # Concatenate all matrices horizontally
    # Result shape: (num_samples, num_text_cols * vocab_size)
    X_text_combined = np.hstack(text_matrices)

    # print(f"Combined text matrix shape: {X_text_combined.shape}")
    # print(f"Individual text arrays: {[m.shape for m in text_matrices]}")

    return X_text_combined


def extract_rating(response) -> Optional[int]:
    """
    Extract numeric integer rating from responses like '3 - Sometimes'.
    Returns None for missing responses
    """
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None


def compute_rating_mode(df, rating_cols) -> dict:
    """
    Return the mode value for rating columns.
    """
    cols = df.columns
    modes = {}
    for col_i in rating_cols:
        col_name = cols[col_i]
        ratings = df[col_name].apply(extract_rating).dropna()
        mode_value = ratings.mode().iloc[0]
        modes[col_i] = mode_value
    return modes


def build_rating_matrix(df: pd.DataFrame, rating_cols: List[int]) -> np.ndarray:
    """
    Input:
    df: pandas DataFrame
    rating_cols: list of int that indicate which columns in df to use
    Return a matrix of shape (num_samples, len(rating_cols))
    """
    num_samples = len(df)
    num_ratings = len(rating_cols)

    X_rating = np.zeros((num_samples, num_ratings),
                        dtype=np.float32)  # (824， 4)

    cols = df.columns
    rating_modes = compute_rating_mode(df, rating_cols)

    for row_i, (_, row) in enumerate(
            df.iterrows()):  # row_i = each data's index, row = each data series
        for j, col_i in enumerate(
                rating_cols):  # j: j-th column of X_rating we want to modify, col_i: the column we are looking at in df
            col_name = cols[col_i]
            raw_value = row[col_name]  # one entry
            rating = extract_rating(
                raw_value)  # obtain the integer of this rating answer

            if rating is not None:
                X_rating[row_i, j] = rating
            else:
                X_rating[row_i, j] = rating_modes[
                    col_i]  # handle missing values

    return X_rating


def parse_multiselect(response):
    """
    Convert multiselect strings from database into lists.
    Returns: list of lists representing multiselect strings.
    The Example of returned list of 4 students:
    [
    ["Converting content between formats", "Math problems"],
    ["debugging code"],
    ["Data processing or analysis", "Converting content between formatsData processing or analysis"]
    []
    ]
    """
    if pd.isna(response) or response == "":
        return []
    clean = str(response).replace("\n", " ").replace("\r", " ")
    parts = [p.strip() for p in clean.split(",") if p.strip()]

    selected = set()
    for raw in parts:
        p = raw.lower()
        for key, canonical in MULTI_MAP.items():
            if key in p:
                selected.add(canonical)
                break
    return list(selected)


def build_multiselect_vocabulary(df, multi_cols):
    """
    Return {col_idx: {option: index}}
    """
    return {t: i for i, t in enumerate(CANONICAL_MULTI_TYPES)}


def build_multiselect_matrix(df, vocab):
    """
    Return a matrix of shape: (num_samples, vocab_size)
    Note that the number of one-hot vector choices is len(multi_cols) * len(each_col) ignoring
    if two columns have the same options.
    """
    n = len(df)
    V = len(vocab)
    Q = len(MULTI_COLS)  # 2 questions
    X = np.zeros((n, Q * V), dtype=np.int32)

    for i, (_, row) in enumerate(df.iterrows()):
        for q_idx, col_idx in enumerate(MULTI_COLS):
            response = row.iloc[col_idx]
            tasks = parse_multiselect(response)
            base = q_idx * V
            for t in tasks:
                if t in vocab:
                    X[i, base + vocab[t]] = 1
    return X


def build_feature_matrix(
        df: pd.DataFrame,
        text_vocab: Dict[str, int],
        multi_vocab: Dict[str, int],
        text_cols: List[int] = TEXT_COLS,
        rating_cols: List[int] = RATING_COLS,
) -> np.ndarray:
    X_text = build_text_matrix(df, text_cols, text_vocab)
    X_rating = build_rating_matrix(df, rating_cols)
    X_multi = build_multiselect_matrix(df, multi_vocab)
    return np.concatenate([X_text, X_rating, X_multi], axis=1)


def debug_print_matrix(name: str, X: np.ndarray, vocab=None):
    """
    This function can be deleted later. Just for debugging purposes
    and showing the shape of each matrix of each type of question.
    """
    print(f"\n===== {name} =====")
    print("Shape:", X.shape)
    print(X)

    if vocab is not None:
        print("\nVocabulary (first 30 entries):")
        items = list(vocab.items())[:30]
        for k, v in items:
            print(f"  {k} -> {v}")


def preprocess_train(filename: str, dict_len: int = None):
    df = pd.read_csv(filename)
    df_train, df_val, df_test = train_val_test_split(df)

    text_vocab = build_text_vocabulary(df_train, TEXT_COLS, dict_len)
    multi_vocab = build_multiselect_vocabulary(df_train, MULTI_COLS)

    X_train = build_feature_matrix(df_train, text_vocab, multi_vocab)
    X_val = build_feature_matrix(df_val, text_vocab, multi_vocab)
    X_test = build_feature_matrix(df_test, text_vocab, multi_vocab)

    y_train = df_train["label"].to_numpy()
    y_val = df_val["label"].to_numpy()
    y_test = df_test["label"].to_numpy()
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_test(filename: str, dict_len: int):
    df = pd.read_csv(filename)
    df_train, df_val, df_test = train_val_test_split(df, shuffle=True)

    text_vocab = build_text_vocabulary(df_train, TEXT_COLS, dict_len)
    multi_vocab = build_multiselect_vocabulary(df_train, MULTI_COLS)

    X_test = build_feature_matrix(df_test, text_vocab, multi_vocab)

    return X_test


def preprocess(filename: str, dict_len: int):
    df = pd.read_csv(filename)

    text_vocab = build_text_vocabulary(df, TEXT_COLS, dict_len)
    multi_vocab = build_multiselect_vocabulary(df, MULTI_COLS)
    X = build_feature_matrix(df, text_vocab, multi_vocab)
    return X


def main():
    df = pd.read_csv(FILENAME)

    # 1) split into train / val / test
    df_train, df_val, df_test = train_val_test_split(df)

    # 2) build vocabularies ONLY on train
    text_vocab = build_text_vocabulary(df_train, TEXT_COLS)
    multi_vocab = build_multiselect_vocabulary(df_train, MULTI_COLS)

    # Debug only on train set
    X_text = build_text_matrix(df_train, TEXT_COLS, text_vocab)
    X_rating = build_rating_matrix(df_train, RATING_COLS)
    X_multi = build_multiselect_matrix(df_train, multi_vocab)

    debug_print_matrix("TEXT MATRIX", X_text, text_vocab)
    debug_print_matrix("RATING MATRIX", X_rating)
    debug_print_matrix("MULTI MATRIX", X_multi, multi_vocab)

    # 3) build feature matrices for each split (use same vocab)
    X_train = build_feature_matrix(df_train, text_vocab, multi_vocab)
    X_val = build_feature_matrix(df_val, text_vocab, multi_vocab)
    X_test = build_feature_matrix(df_test, text_vocab, multi_vocab)

    # 4) labels
    y_train = df_train["label"].to_numpy()
    y_val = df_val["label"].to_numpy()
    y_test = df_test["label"].to_numpy()

    # print shapes
    print("X_train shape:", X_train.shape)

    print("X_val shape:  ", X_val.shape)
    print("X_test shape: ", X_test.shape)

    print("y_train shape:", y_train.shape)
    print("y_val shape:  ", y_val.shape)
    print("y_test shape: ", y_test.shape)

    # 5) save to disk
    np.save("X_train.npy", X_train)
    np.save("X_val.npy", X_val)
    np.save("X_test.npy", X_test)

    np.save("y_train.npy", y_train)
    np.save("y_val.npy", y_val)
    np.save("y_test.npy", y_test)


if __name__ == "__main__":
    main()
