import json
import os
import re
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd



"""
Constants/Static
"""

# TODO: IMPORTANT
# 控制要不要这个feature vectors/matrix的
USE_BOW = True  #是否用 Bag-of-Words特征
USE_RATING = True #是否启用评分题（1-5 分）特征
USE_MULTI = True #是否启用多选题 one-hot 特征
USE_MULTI_INTERACTIONS = False # 是否加入多选题选项之间的 pairwise 交互特征
USE_TEXT_LEN = True # 是否加入文本回答长度特征
USE_MULTI_CNT = True # 是否加入多选题选择项的数量特征
USE_STATIC_EMBEDDINGS = False # 是否加入静态词向量（如 GloVe / FastText）的文本平均 embedding

FILENAME = "training_data_clean.csv"
TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)

# Define the index of each type of questions
TEXT_COLS = [1, 6, 9] # free-text (open-ended) questions
RATING_COLS = [2, 4, 7, 8] # rating questions
MULTI_COLS = [3, 5] # multi-select questions

# Save the path of Vocab (for pred.py)
TEXT_VOCAB_PATH = "text_vocab.json"
MULTI_VOCAB_PATH = "multi_vocab.json"

# Define
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

# Define words that are not useful for classifying
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "this", "that", "it", "at", "as",
}

# For extra features
STATIC_EMBED_PATH = "static_embeddings.txt"  # Path to the static word embedding file
USE_BIGRAMS = False  # Whether to include bigrams in the BoW representation
USE_MULTI_INTERACTIONS = False  # Whether to include pairwise interaction features for multi-select questions

# Internal cache for static embeddings
_STATIC_EMBED_CACHE: Optional[Dict[str, np.ndarray]] = None
_STATIC_EMBED_DIM: int = 0


"""
This following modulo is for split training/validation/test data without sklearn.
"""

def train_val_test_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, random_state: int = 666,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform a stratified split of a DataFrame into train / validation / test sets, without relying on sklearn.
        Stratified means each label is split independently according to the same ratios. The resulting train/val/test sets
        preserve the label distribution of the original dataset.

        :param df: DataFrame to split. Must contain a column named "label".
        :param train_ratio: Ratio of training set size.
        :param val_ratio: Ratio of validation set size.
        :param random_state: Random seed for reproducibility.
        :return: Tuple[df_train, df_val, df_test]
        """
        rng = np.random.default_rng(random_state) # random seed generator

        # store data of stratified subsets
        df_train_list = []
        df_val_list = []
        df_test_list = []

        labels = sorted(df["label"].unique()) # get all unique label values, sorted for consistency

        for lab in labels:
            df_lab = df[df["label"] == lab].copy() # Get all rows belonging to this label
            # Shuffle
            indices = df_lab.index.to_numpy()
            rng.shuffle(indices)
            df_lab = df_lab.loc[indices]

            # Split data based on ratios
            n = len(df_lab)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            n_test = n - n_train - n_val
            # Append the slices
            df_train_list.append(df_lab.iloc[:n_train])  # df_train_list = [data_label_A, data_label_B, data_label_C]
            df_val_list.append(df_lab.iloc[n_train:n_train + n_val])
            df_test_list.append(df_lab.iloc[n_train + n_val:])

        # Combine all label-specific segments and shuffle again
        df_train = pd.concat(df_train_list).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        df_val = pd.concat(df_val_list).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        df_test = pd.concat(df_test_list).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        return df_train, df_val, df_test



"""
This following modulo is for processing data of generative free texts questions
"""

def normalize_to_text_list(value) -> List[str]:
    """
    Convert a cell value into a normalized list of tokens by lowercasing, regex tokenization, and removal of simple
    English stopwords.
    :param value: A cell value from the DataFrame, which may be None, NaN, a scalar, or a list.
    :return: A list of cleaned tokens extracted from the value.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)): # value is None or NaN
        return []

    if isinstance(value, list):  # value is a list
        tokens: List[str] = []
        for v in value:  # iterate the element
            tokens.extend(normalize_to_text_list(v))  # expand the list
        return tokens

    # value is a string
    text = str(value).lower() # lowercase
    tokens = TOKEN_PATTERN.findall(text) # # tokenize the text using the predefined regex
    tokens = [t for t in tokens if t not in STOPWORDS] # remove stopwords
    return tokens


def load_static_embeddings(path: str = STATIC_EMBED_PATH, max_words: Optional[int] = None,) -> Dict[str, np.ndarray]:
    """
    Load static word embeddings from a plain-text file where each line has the format 'word v1 v2 v3 ...', and cache
    the result for reuse.
    :param path: Path to the embedding text file.
    :param max_words: Optional maximum number of lines (words) to read from the file.
    :return: A dictionary mapping each token string to its embedding vector as a NumPy array.
            If the file does not exist or embeddings are disabled, an empty dict is returned.
    """
    global _STATIC_EMBED_CACHE, _STATIC_EMBED_DIM, USE_STATIC_EMBEDDINGS

    if _STATIC_EMBED_CACHE is not None:
        return _STATIC_EMBED_CACHE

    if not USE_STATIC_EMBEDDINGS:
        _STATIC_EMBED_CACHE = {}
        _STATIC_EMBED_DIM = 0
        return _STATIC_EMBED_CACHE

    if not os.path.exists(path):
        print(f"[WARN] Static embedding file '{path}' not found. "
              f"Static embedding features will be skipped.")
        USE_STATIC_EMBEDDINGS = False
        _STATIC_EMBED_CACHE = {}
        _STATIC_EMBED_DIM = 0
        return _STATIC_EMBED_CACHE

    print(f"[INFO] Loading static embeddings from '{path}' ...")
    embeddings: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) <= 2:
                continue
            word = parts[0]
            try:
                vec = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue  # Some lines may be malformed; skip them
            embeddings[word] = vec
            if max_words is not None and i >= max_words:
                break

    if embeddings:
        _STATIC_EMBED_DIM = len(next(iter(embeddings.values())))
        print(f"[INFO] Loaded {len(embeddings)} words, dim = {_STATIC_EMBED_DIM}")
    else:
        _STATIC_EMBED_DIM = 0
        print("[WARN] No embeddings loaded. Check the embedding file format.")

    _STATIC_EMBED_CACHE = embeddings
    return embeddings


def generate_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """
    Generate n-grams from a sequence of tokens, such as turning
    ['debugging', 'code', 'issues'] into ['debugging code', 'code issues'] for n=2.

    :param tokens: A list of input tokens.
    :param n: The size of the n-grams to generate.
    :return: A list of n-gram strings joined by a space.
    """
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def build_static_embedding_features(df: pd.DataFrame, text_cols: List[int] = TEXT_COLS,) -> np.ndarray:
    """
    Build averaged static embedding features for each sample and each text column by mapping tokens to embeddings
    and averaging them.

    :param df: Input DataFrame containing the text columns.
    :param text_cols: A list of column indices corresponding to free-text questions.
    :return: A feature matrix of shape (num_samples, len(text_cols) * embed_dim). If embeddings
             are disabled or unavailable, a matrix of shape (num_samples, 0) is returned.
    """
    embeddings = load_static_embeddings()
    n = len(df)

    if (not USE_STATIC_EMBEDDINGS) or (_STATIC_EMBED_DIM == 0) or (len(embeddings) == 0):
        # Static embeddings are disabled or unavailable; return an empty feature block
        return np.zeros((n, 0), dtype=np.float32)

    embed_dim = _STATIC_EMBED_DIM
    num_q = len(text_cols)
    X_embed = np.zeros((n, num_q * embed_dim), dtype=np.float32)
    cols = df.columns

    for row_i, (_, row) in enumerate(df.iterrows()):
        for j, col_idx in enumerate(text_cols):
            col_name = cols[col_idx]
            tokens = normalize_to_text_list(row[col_name])

            vec_sum = np.zeros((embed_dim,), dtype=np.float32)
            count = 0
            for t in tokens:
                if t in embeddings:
                    vec_sum += embeddings[t]
                    count += 1

            if count > 0:
                avg_vec = vec_sum / float(count)
            else:
                avg_vec = vec_sum  # remains all zeros if no embeddings matched

            start = j * embed_dim
            end = (j + 1) * embed_dim
            X_embed[row_i, start:end] = avg_vec

    return X_embed


def build_text_vocabulary(df: pd.DataFrame, text_cols: List[int], min_freq: int = 5) -> Dict[int, Dict[str, int]]:
    """
    Build a vocabulary for each free-text column, mapping tokens to indices, optionally including bigrams and filtering
    by minimum frequency.

    :param df: Input DataFrame containing the text columns.
    :param text_cols: A list of column indices corresponding to free-text questions.
    :param min_freq: Minimum token frequency required in a column to be kept in the vocabulary.
    :return: A mapping {col_idx: {token: index}} defining a separate vocabulary per text column.
    """
    vocab_per_col: Dict[int, Dict[str, int]] = {}
    cols = df.columns

    for col_idx in text_cols:
        vocabulary: Dict[str, int] = {}
        col_name = cols[col_idx]

        for value in df[col_name]:
            tokens = normalize_to_text_list(value)

            # unigram
            all_tokens = list(tokens)

            # bigram 可选
            if USE_BIGRAMS:
                bigrams = generate_ngrams(tokens, n=2)
                all_tokens.extend(bigrams)

            for t in all_tokens:
                vocabulary[t] = vocabulary.get(t, 0) + 1

        filtered_words = sorted([w for w, c in vocabulary.items() if c >= min_freq])
        vocab_per_col[col_idx] = {w: i for i, w in enumerate(filtered_words)}

    return vocab_per_col


def build_text_matrix(df: pd.DataFrame, text_cols: List[int], vocab_per_col: Dict[int, Dict[str, int]],) -> np.ndarray:
    """
    Build Bag-of-Words (BoW) feature blocks for each text column and concatenate them along the feature dimension,
    using count-based BoW with log smoothing.

    :param df: Input DataFrame containing the text columns.
    :param text_cols: A list of column indices corresponding to free-text questions.
    :param vocab_per_col: A mapping {col_idx: {token: index}} defining per-column vocabularies.
    :return: A BoW feature matrix of shape (num_samples, total_text_feature_dim).
    """
    num_samples = len(df)
    cols = df.columns

    # 计算各列在大矩阵里的 offset
    offsets: Dict[int, int] = {}
    total_dim = 0
    for col_idx in text_cols:
        v = vocab_per_col[col_idx]
        offsets[col_idx] = total_dim
        total_dim += len(v)

    X = np.zeros((num_samples, total_dim), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        for col_idx in text_cols:
            col_name = cols[col_idx]
            vocab = vocab_per_col[col_idx]
            base = offsets[col_idx]

            tokens = normalize_to_text_list(row[col_name])

            if USE_BIGRAMS:
                bigrams = generate_ngrams(tokens, n=2)
                all_tokens = tokens + bigrams
            else:
                all_tokens = tokens

            for t in all_tokens:
                if t in vocab:
                    j = base + vocab[t]
                    X[i, j] += 1.0

    X = np.log1p(X)  # log(1 + count) smoothing
    return X


"""
This following modulo is for processing data of rating questions. Ratings are usually strings like '3 - Sometimes', 
and must be converted into numerical features.
"""

def extract_rating(response) -> Optional[int]:
    """
    Extract the numeric rating from strings such as '3 - Sometimes'. That is, obtain the integer 3.
    :param response: A cell value from the DataFrame, commonly a string containing a number at the beginning
    (e.g., "4 - Often")
    :return: A numerical rating in integer, None if the cell value is missing.
    """
    match = re.match(r"^(\d+)", str(response))  # Convert response to string, then match digits at the beginning
    return int(match.group(1)) if match else None # Return the integer if matched, otherwise return None


def compute_rating_mode(df: pd.DataFrame, rating_cols: List[int]) -> Dict[int, float]:
    """
    Compute the mode (most common value) for each rating column. Used later to fill missing ratings.
    :param df: A dataframe containing numeric ratings.
    :param rating_cols: A list of column indices containing rating values.
    :return: A dictionary mapping rating column indices to their mode. {column #2: mode integer}
    """
    cols = df.columns
    modes: Dict[int, float] = {}

    for col_i in rating_cols: # iterate the rating questions columns
        col_name = cols[col_i]
        ratings = df[col_name].apply(extract_rating).dropna()  # Extract numeric ratings for this column and drop NaNs
        mode_value = ratings.mode().iloc[0]  # ratings.mode() returns a Series, take the first mode
        modes[col_i] = mode_value  # store in dict

    return modes


def build_rating_matrix(df: pd.DataFrame, rating_cols: List[int]) -> np.ndarray:
    """
    Build a numeric feature matrix for rating questions.
    Each rating is:
      1. Extracted from strings such as "3 - Sometimes".
      2. Missing values are filled with the column's mode.
      3. Scaled from the original [1, 5] range into [0, 1]. (Optional)
    :param df: The dataset containing rating columns.
    :param rating_cols: List of column indices corresponding to rating questions.
    :return: A NumPy array of shape (num_samples, len(rating_cols)) where each entry is a normalized rating value
    within [0, 1].
    """
    num_samples = len(df)
    num_ratings = len(rating_cols)

    X_rating = np.zeros((num_samples, num_ratings), dtype=np.float32)  # Initialize all-0 output matrix

    cols = df.columns
    rating_modes = compute_rating_mode(df, rating_cols)  # Pre-compute mode values for imputation

    # Iterate through each row of the DataFrame
    for row_i, (_, row) in enumerate(df.iterrows()):
        for j, col_i in enumerate(rating_cols):
            col_name = cols[col_i]
            raw_value = row[col_name]

            rating = extract_rating(raw_value)  # Extract the integer rating

            if rating is not None:
                X_rating[row_i, j] = rating
            else:  # missing values will be fulfilled with the mode
                X_rating[row_i, j] = rating_modes[col_i]

    # Rescale ratings from original range [1,5] to [0,1] (Optional: this is for normalized the numbers)
    X_rating = (X_rating - 1.0) / 4.0

    return X_rating




"""
This following modulo is for processing data of multiplicative choices questions. The raw responses may contain commas, 
mixed text, inconsistent formatting, etc. These utilities map raw answers into canonical options and build numeric 
feature matrices.
"""

def parse_multiselect(response):
    """
    Parse a multi-select question response into a list of canonical option strings,
    based on keyword matching defined in MULTI_MAP.
    :param response: A cell value containing a multi-select response, usually a comma-separated string.
    :return: A list of canonical option labels selected in this response.
    """
    # Treat NaN or empty string as "no selection" (missing values solution)
    if pd.isna(response) or response == "":
        return []

    clean = str(response).replace("\n", " ").replace("\r", " ")  # Clean text: remove line breaks
    parts = [p.strip() for p in clean.split(",") if p.strip()]  # Split by comma, remove surrounding spaces, drop empty parts

    selected = set()  # use a set to ensure each selected option appears only once
    for raw in parts:
        p = raw.lower()  # lowercase
        # MULTI_MAP maps small keywords → canonical standardized names
        for key, canonical in MULTI_MAP.items():
            # If the keyword appears anywhere in the text, treat it as selected
            if key in p:
                selected.add(canonical)
                break

    return list(selected)  # Convert set → list


def build_multiselect_vocabulary(df: pd.DataFrame,multi_cols: List[int]) -> Dict[str, int]:
    """
    Build a vocabulary for multi-select questions using the predefined canonical option list.
    :param df: Input DataFrame (not used directly but kept for interface consistency).
    :param multi_cols: A list of column indices for multi-select questions (not used directly here).
    :return: A mapping {option: index} for all canonical multi-select options.
    """
    return {t: i for i, t in enumerate(CANONICAL_MULTI_TYPES)}


def build_multiselect_matrix(df: pd.DataFrame, vocab: Dict[str, int]) -> np.ndarray:
    """
    Build a one-hot encoded matrix for multi-select questions across all specified columns.
    :param df: Input DataFrame containing multi-select question responses.
    :param vocab: A mapping {option: index} for canonical multi-select options.
    :return: A feature matrix of shape (num_samples, len(MULTI_COLS) * len(vocab)).
    """
    n = len(df)  # Number of samples
    V = len(vocab)  # Number of canonical options
    Q = len(MULTI_COLS)  # Number of multi-select questions
    X = np.zeros((n, Q * V), dtype=np.int32)  # Final matrix: each question gets its own V-sized block

    for i, (_, row) in enumerate(df.iterrows()):
        # Process each multi-select question
        for q_idx, col_idx in enumerate(MULTI_COLS):
            response = row.iloc[col_idx]
            tasks = parse_multiselect(response)  # Extract canonical selections
            # Starting index of the block assigned to this question
            base = q_idx * V
            # Mark selected canonical options as 1
            for t in tasks:
                if t in vocab:
                    X[i, base + vocab[t]] = 1
    return X


def build_multiselect_interaction_features(df: pd.DataFrame, vocab: Dict[str, int], multi_cols: List[int] = MULTI_COLS,
) -> np.ndarray:
    """
    Build pairwise interaction features for multi-select questions. For each sample, the union of all selected canonical
    options across all multi-select questions is taken, and for every pair of options (opt_i, opt_j) with i < j, the
    corresponding feature is set to 1 if both are selected.
    :param df: Input DataFrame containing multi-select question responses.
    :param vocab: A mapping {option: index} for canonical multi-select options.
    :param multi_cols: A list of column indices for multi-select questions.
    :return: A feature matrix of shape (num_samples, num_pairs), where num_pairs = C(V, 2)
             and V is the number of canonical options.
    """
    n = len(df)
    option_list = sorted(vocab.keys(), key=lambda x: vocab[x])
    V = len(option_list)

    # 枚举所有 (i, j) 对，i < j
    pairs = []
    for i in range(V):
        for j in range(i + 1, V):
            pairs.append((i, j))
    num_pairs = len(pairs)

    X_pair = np.zeros((n, num_pairs), dtype=np.int32)

    for row_i, (_, row) in enumerate(df.iterrows()):
        # 收集该样本在所有 multi-select 问题中选择的 canonical 选项（用 index 表示）
        chosen_indices = set()
        for col_idx in multi_cols:
            response = row.iloc[col_idx]
            tasks = parse_multiselect(response)
            for t in tasks:
                if t in vocab:
                    chosen_indices.add(vocab[t])

        if len(chosen_indices) < 2:
            continue  # 少于两个选项，没有 pair

        chosen_indices = sorted(chosen_indices)
        chosen_set = set(chosen_indices)
        for pair_idx, (i, j) in enumerate(pairs):
            if i in chosen_set and j in chosen_set:
                X_pair[row_i, pair_idx] = 1

    return X_pair




"""
This following modulo is for adding extra features.
"""

def build_text_length_features(df: pd.DataFrame,text_cols: List[int] = TEXT_COLS) -> np.ndarray:
    """
    Build text length features by counting tokens for each sample and each text column,
    then applying log to compress larger values.

    :param df: Input DataFrame containing free-text question responses.
    :param text_cols: A list of column indices corresponding to free-text questions.
    :return: A feature matrix of shape (num_samples, len(text_cols)) with log1p token counts.
    """
    num_samples = len(df)
    num_q = len(text_cols)
    X_len = np.zeros((num_samples, num_q), dtype=np.float32)
    cols = df.columns

    for i, (_, row) in enumerate(df.iterrows()):
        for j, col_idx in enumerate(text_cols):
            tokens = normalize_to_text_list(row[cols[col_idx]])
            length = len(tokens)
            X_len[i, j] = np.log1p(length)

    return X_len


def build_multiselect_count_features(df: pd.DataFrame,multi_cols: List[int] = MULTI_COLS) -> np.ndarray:
    """
    Build features representing how many options were selected in each multi-select question,
    with log compression of the counts.

    :param df: Input DataFrame containing multi-select question responses.
    :param multi_cols: A list of column indices for multi-select questions.
    :return: A feature matrix of shape (num_samples, len(multi_cols)) with log1p option counts.
    """
    num_samples = len(df)
    num_q = len(multi_cols)
    X_cnt = np.zeros((num_samples, num_q), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        for j, col_idx in enumerate(multi_cols):
            response = row.iloc[col_idx]
            tasks = parse_multiselect(response)
            X_cnt[i, j] = np.log1p(len(tasks))

    return X_cnt




"""
This following modulo is for combining matrix.
"""

def build_feature_matrix(df: pd.DataFrame,text_vocab: Dict[int, Dict[str, int]],multi_vocab: Dict[str, int],
                         text_cols: List[int] = TEXT_COLS,rating_cols: List[int] = RATING_COLS,) -> np.ndarray:
    """
    Build the final feature matrix by concatenating multiple feature blocks, including text BoW,
    numeric ratings, multi-select one-hot features and interactions, text length, multi-select
    option counts, and averaged static embeddings.
    - Bag-of-Words (BoW) text features
    - Normalized numeric rating values
    - Multi-select one-hot encoded vectors
    - Multi-select pairwise interaction features
    - Text length features
    - Multi-select option count features
    - Static embedding averaged vectors

    :param df: Input DataFrame containing all raw survey responses.
    :param text_vocab: A mapping {col_idx: {token: index}} defining per-column text vocabularies.
    :param multi_vocab: A mapping {option: index} for canonical multi-select options.
    :param text_cols: A list of column indices corresponding to free-text questions.
    :param rating_cols: A list of column indices corresponding to rating-scale questions.
    :return: A 2D NumPy array representing the concatenated feature matrix for all samples.
    """
    feature_blocks = []

    # Text Bag-of-Words (BoW) features
    if USE_BOW:
        X_text = build_text_matrix(df, text_cols, text_vocab)
        feature_blocks.append(X_text)
    else:
        # Append an empty block (0-width) to maintain consistent concatenation
        feature_blocks.append(np.zeros((len(df), 0)))

    #
    # Rating-scale features (normalized to [0,1])
    if USE_RATING:
        X_rating = build_rating_matrix(df, rating_cols)
        feature_blocks.append(X_rating)
    else:
        feature_blocks.append(np.zeros((len(df), 0)))

    # Multi-select one-hot encoded features
    if USE_MULTI:
        X_multi = build_multiselect_matrix(df, multi_vocab)
        feature_blocks.append(X_multi)
    else:
        feature_blocks.append(np.zeros((len(df), 0)))

    # Multi-select pairwise interaction features
    # Only enabled if both USE_MULTI and USE_MULTI_INTERACTIONS are True
    if USE_MULTI and USE_MULTI_INTERACTIONS:
        X_multi_inter = build_multiselect_interaction_features(df, multi_vocab, MULTI_COLS)
        feature_blocks.append(X_multi_inter)
    else:
        feature_blocks.append(np.zeros((len(df), 0)))

    # Text length features (log of token count)
    if USE_TEXT_LEN:
        X_text_len = build_text_length_features(df, text_cols)
        feature_blocks.append(X_text_len)
    else:
        feature_blocks.append(np.zeros((len(df), 0)))

    # Multi-select option count features (log of option count)
    if USE_MULTI_CNT:
        X_multi_cnt = build_multiselect_count_features(df, MULTI_COLS)
        feature_blocks.append(X_multi_cnt)
    else:
        feature_blocks.append(np.zeros((len(df), 0)))

    # 7. Static embedding averaged vectors (GloVe, FastText, etc）
    if USE_STATIC_EMBEDDINGS:
        X_embed = build_static_embedding_features(df, text_cols)
        feature_blocks.append(X_embed)
    else:
        feature_blocks.append(np.zeros((len(df), 0)))

    # Final concatenation of all enabled feature blocks
    return np.concatenate(feature_blocks, axis=1)


def debug_print_matrix(name: str, X: np.ndarray):
    """
    Print the name and shape of a feature matrix for debugging purposes.
    :param name: A short label for the matrix.
    :param X: The NumPy array whose shape will be printed.
    :return: None. This function prints to stdout.
    """
    print(f"\n===== {name} =====")
    print("Shape:", X.shape)




"""
This following modulo is for rreprocessing: training phase / test phase
"""

def preprocess_train(filename: str):
    """
    Preprocess a CSV file for the training phase by splitting into train/val/test, building vocabularies on the training
    split only, and constructing feature matrices and label arrays for all three splits.
    :param filename: Path to the input CSV file containing labeled survey data.
    :return: A tuple (X_train, X_val, X_test, y_train, y_val, y_test) with feature matrices and corresponding label
    arrays as NumPy arrays.
    """
    df = pd.read_csv(filename)
    df_train, df_val, df_test = train_val_test_split(df)

    # Only build vocab on training data in case of data leakage
    text_vocab = build_text_vocabulary(df_train, TEXT_COLS)
    multi_vocab = build_multiselect_vocabulary(df_train, MULTI_COLS)

    # save vocab to json(for preprocess_test/pred.py)
    with open(TEXT_VOCAB_PATH, "w", encoding="utf-8") as f:
        # change to str type
        json.dump({str(k): v for k, v in text_vocab.items()},
                  f, ensure_ascii=False, indent=2)

    with open(MULTI_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(multi_vocab, f, ensure_ascii=False, indent=2)

    X_train = build_feature_matrix(df_train, text_vocab, multi_vocab)
    X_val = build_feature_matrix(df_val, text_vocab, multi_vocab)
    X_test = build_feature_matrix(df_test, text_vocab, multi_vocab)

    y_train = df_train["label"].astype(str).to_numpy(dtype=str)
    y_val = df_val["label"].astype(str).to_numpy(dtype=str)
    y_test = df_test["label"].astype(str).to_numpy(dtype=str)

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_test(filename: str):
    """
    Preprocess a new CSV file for the test or inference phase using vocabularies saved during training to ensure
    consistent feature construction.

    :param filename: Path to the input CSV file containing labeled or unlabeled survey data.
    :return: A tuple (X, y) where X is the feature matrix and y is the label array as strings. If the 'label' column is
    unavailable or incomplete, y may not be suitable for evaluation.
    """
    df = pd.read_csv(filename)

    with open(TEXT_VOCAB_PATH, "r", encoding="utf-8") as f:
        raw_text_vocab = json.load(f)
    text_vocab = {int(k): v for k, v in raw_text_vocab.items()}

    with open(MULTI_VOCAB_PATH, "r", encoding="utf-8") as f:
        multi_vocab = json.load(f)

    X = build_feature_matrix(df, text_vocab, multi_vocab)
    y = df["label"].astype(str).to_numpy(dtype=str)
    return X, y


"""
MAIN FUNCTION
"""

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_train(FILENAME)

    print("X_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("X_test shape: ", X_test.shape)

    print("y_train shape:", y_train.shape)
    print("y_val shape:  ", y_val.shape)
    print("y_test shape: ", y_test.shape)

    # save
    np.save("X_train.npy", X_train)
    np.save("X_val.npy", X_val)
    np.save("X_test.npy", X_test)

    np.save("y_train.npy", y_train)
    np.save("y_val.npy", y_val)
    np.save("y_test.npy", y_test)


if __name__ == "__main__":
    main()
