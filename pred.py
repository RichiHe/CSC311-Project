import re
from typing import Optional, List, Tuple, Dict
from unittest.mock import DEFAULT
import numpy as np
import pandas as pd


FILENAME = "training_data_clean.csv"
TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)

TEXT_COLS = [1, 6, 9]   # Free Texts Questions
RATING_COLS = [2, 4, 7, 8]  # Rating Questions
MULTI_COLS = [3, 5]     # Multiplicative choices Questions


def train_val_test_split(df: pd.DataFrame,train_ratio: float = 0.7, val_ratio: float = 0.15,shuffle: bool = True,
    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train / val / test = 7 : 1.5 : 1.5
    """
    if shuffle:
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train + n_val]
    df_test = df.iloc[n_train + n_val:]

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
            tokens.extend(normalize_to_text_list(v)) # expand the nest list
        return tokens

    # The normal case: split the long answer
    text = str(value).lower()
    tokens = TOKEN_PATTERN.findall(text)
    return tokens


def build_text_vocabulary(df, text_cols,) -> Dict[str, int]:
    """
    Return the list of unique words for all columns in text_cols, denoting as the vocabulary.
    Note that we return a vocabulary combining all text columns in order to keep the same dimension
    across multiple columns for free text questions.
    Returns: dict of {word: index}
    Note that we count the frequencies of each word in text_cols just for possible future extension
    like filtering out low-frequency words or analyze data. We did not return the frequencies for our
    vocabulary this time.
    """
    vocabulary = {}
    cols = df.columns # get the list of column headers

    for col_idx in text_cols: # the index/position of free text questions
        col_name = cols[col_idx]
        for value in df[col_name]: # each data row in this column
            tokens = normalize_to_text_list(value) # make each answer into a list of string
            for token in tokens: # each word in each answer
                vocabulary[token] = vocabulary.get(token, 0) + 1 # update words and frequency

    words = sorted(vocabulary.keys()) # an ordering alphabetically list of words
    word_to_index = {word: i for i, word in enumerate(words)} # a dictionary {word: index}
    return word_to_index


def build_text_matrix(df: pd.DataFrame,text_cols: List[int],vocab: Dict[str, int]) -> np.ndarray:
    """
    Input:
    df: pandas DataFrame
    text_cols: list of int that indicate which columns in df to use
    Based on the vocabulary list, transfer free text to a bag-of-words matrix in shape (num_samples, vocab_size).
    """
    num_samples = len(df) # the size of data
    vocab_size = len(vocab) # the size of columns
    X_text = np.zeros((num_samples, vocab_size), dtype=np.int32) # make an all-zero matrix

    cols = df.columns
    for row_i, (_, row) in enumerate(df.iterrows()): # iterate each row with index row_i and the row series

        # modify the matrix X_text based on the frequencies of the corresponding word
        for col_i in text_cols: # only check the free-text questions
            col_name = cols[col_i]
            text = row[col_name]
            tokens = normalize_to_text_list(text) # make the entry answer into a list of string
            for token in tokens: # for each word in this answer
                if token in vocab:
                    col = vocab[token] # since our vocabulary is a fixed order dictionary in letters
                    X_text[row_i, col] += 1 # change the frequencies for this text

    return X_text


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


def build_rating_matrix(df: pd.DataFrame, rating_cols:List[int]) -> np.ndarray:
    """
    Input:
    df: pandas DataFrame
    rating_cols: list of int that indicate which columns in df to use
    Return a matrix of shape (num_samples, len(rating_cols))
    """
    num_samples = len(df)
    num_ratings = len(rating_cols)

    X_rating = np.zeros((num_samples, num_ratings), dtype=np.float32) #(824， 4)

    cols = df.columns
    rating_modes = compute_rating_mode(df, rating_cols)

    for row_i, (_, row) in enumerate(df.iterrows()): # row_i = each data's index, row = each data series
        for j, col_i in enumerate(rating_cols): # j: j-th column of X_rating we want to modify, col_i: the column we are looking at in df
            col_name = cols[col_i]
            raw_value = row[col_name] # one entry
            rating = extract_rating(raw_value) # obtain the integer of this rating answer

            if rating is not None:
                X_rating[row_i, j] = rating
            else:
                X_rating[row_i, j] = rating_modes[col_i] # handle missing values

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
    return [s.strip() for s in str(response).split(",") if s.strip()]


def build_multiselect_vocabulary(df, multi_cols):
    """
    Return {col_idx: {option: index}}
    """
    vocabs = {}
    cols = df.columns

    for col_idx in multi_cols:
        col_name = cols[col_idx]
        opts = set()

        for value in df[col_name]:
            for opt in parse_multiselect(value):
                opts.add(opt)

        vocab = {opt: i for i, opt in enumerate(sorted(opts))}
        vocabs[col_idx] = vocab

    return vocabs


def build_multiselect_matrix(df, multi_cols, vocabs):
    """
    Return a matrix of shape: (num_samples, vocab_size)
    Note that the number of one-hot vector choices is len(multi_cols) * len(each_col) ignoring
    if two columns have the same options.
    """
    matrices = []
    cols = df.columns

    for col_idx in multi_cols:
        col_name = cols[col_idx]
        vocab = vocabs[col_idx]
        X = np.zeros((len(df), len(vocab)), dtype=np.int32)

        for i, (_, row) in enumerate(df.iterrows()):
            for opt in parse_multiselect(row[col_name]):
                if opt in vocab:
                    X[i, vocab[opt]] = 1

        matrices.append(X)

    return np.concatenate(matrices, axis=1)


def build_feature_matrix(
    df: pd.DataFrame,
    text_vocab: Dict[str, int],
    multi_vocabs: Dict[int, Dict[str, int]],
    text_cols: List[int] = TEXT_COLS,
    rating_cols: List[int] = RATING_COLS,
    multi_cols: List[int] = MULTI_COLS
) -> np.ndarray:
    """
    Build text + rating + multiselect matrices and concatenate them.
    This function assumes vocabularies are already built (usually from train set).
    """
    X_text = build_text_matrix(df, text_cols, text_vocab)
    X_rating = build_rating_matrix(df, rating_cols)
    X_multi = build_multiselect_matrix(df, multi_cols, multi_vocabs)

    X_all = np.concatenate([X_text, X_rating, X_multi], axis=1)
    return X_all


def main():
    df = pd.read_csv(FILENAME)

    # 1) split into train / val / test
    df_train, df_val, df_test = train_val_test_split(df)

    # 2) build vocabularies ONLY on train
    text_vocab = build_text_vocabulary(df_train, TEXT_COLS)
    multi_vocabs = build_multiselect_vocabulary(df_train, MULTI_COLS)

    # 3) build feature matrices for each split (use same vocab)
    X_train = build_feature_matrix(df_train, text_vocab, multi_vocabs)
    X_val = build_feature_matrix(df_val, text_vocab, multi_vocabs)
    X_test = build_feature_matrix(df_test, text_vocab, multi_vocabs)

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