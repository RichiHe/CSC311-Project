import re
from typing import Optional

import numpy as np
import pandas as pd

FILENAME = "training_data_clean.csv"

def extract_rating(response) -> Optional[int]:
    """
    Extract numeric integer rating from responses like '3 - Sometimes'.
    Returns None for missing responses
    """
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None

def process_multiselect(response):
    """
    Convert multiselect strings into lists.
    Series: pandas.Series object.
    target_tasks: list of strings.
    Returns: list of lists representing multiselect strings.
    The Example for 4 students:
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

def build_features(df: pd.DataFrame):
    text_idxs = [1, 6, 9]
    rating_idxs = [2, 4, 7, 8]
    multi_idxs = [3, 5]

    cols = df.columns

    text_features = []
    for idx in text_idxs:
        col = cols[idx]
        word_counts = (
            df[col]
            .fillna("")
            .astype(str)
            .apply(lambda s: len(s.split()))
            .to_numpy(dtype=float)
        )
        text_features.append(word_counts)

    rating_features = []
    for idx in rating_idxs:
        col = cols[idx]
        nums = (
            df[col]
            .apply(extract_rating)  # "3 — xxx" -> 3
            .fillna(0)  # 缺失就 0
            .astype(float)
            .to_numpy()
        )
        rating_features.append(nums)

"""
Type 1 Free texts preprocessing
"""
# fill "" for the missing values
for col in TEXT_COLS:
    df[col] = df[col].fillna("")

df["full_text"] = (df[TEXT_COLS[0]].astype(str) + " " + df[TEXT_COLS[1]].astype(str) + " "
                   + df[TEXT_COLS[2]].astype(str)) # New a column combining all free text answers

# Construct a vectorizer instance
tfidf_vectorizer = TfidfVectorizer(preprocessor=str.lower, tokenizer=str.split, token_pattern=None,
                                   use_idf=True, norm=None, max_features=3000)
# max_feature is dependent on the average of text length we analyze in Colab

# Examples
print(df["full_text"][0])
print(df["full_text"][1])

X_text = tfidf_vectorizer.fit_transform(df["full_text"]).toarray()
print(X_text.shape) # (825, 3000), each row is a student's data, each column the single feature/word
print(X_text[0])

# download the vocabulary dictionary and idf for future pred.py
vocab_raw = tfidf_vectorizer.vocabulary_
vocab = {token: int(idx) for token, idx in vocab_raw.items()}
idf = tfidf_vectorizer.idf_

with open("tfidf_vocab.json", "w") as f:
    json.dump(vocab, f)
np.save("tfidf_idf.npy", idf)

with open("text_cols.json", "w") as f:
    json.dump(TEXT_COLS, f)

"""
Type 2 rating problems preprocessing
"""
# Record the mode for filling missing values for each column
rating_fill_values: Dict[str, str] = {}

# Type 2 missing values process, choose mode value
for col in RATING_COLS:
    mode_value = df[col].mode(dropna=True)[0]
    rating_fill_values[col] = str(mode_value)
    df[col] = df[col].fillna(mode_value)

# change each rating column to integers, e.g. from "3-sometimes" to 3
for col in RATING_COLS:
    df[col] = df[col].apply(extract_rating)

# make a matrix where each row is a data point, each column is an integer corresponds to each rating problem
X_Rating = df[RATING_COLS].to_numpy(dtype=float)
print("The shape of X_Rating is", X_Rating.shape)
print("Show one data row X_Rating[0]", X_Rating[0])

with open("rating_fill_values.json", "w") as f:
    json.dump(rating_fill_values, f)

with open("rating_cols.json", "w") as f:
    json.dump(RATING_COLS, f)

"""
Type 3 rating problems preprocessing
"""
# Get answers for all multiplicative questions
all_multi_answers = set()
for col in MULTI_COLS:
    for response in df[col]:
        for opt in process_multiselect(response):
            all_multi_answers.add(opt)

# Fix the order of the set "All_MULTI_ANSWERS" alphabetically
All_MULTI_ANSWERS = sorted(all_multi_answers)
print("All_MULTI_ANSWERS:", All_MULTI_ANSWERS)
print(len(All_MULTI_ANSWERS))

# 写一下doc
All_MULTI_ANSWERS_INDEX = {opt: i for i, opt in enumerate(All_MULTI_ANSWERS)}


# Store multi options for pred
with open("multi_cols.json", "w") as f:
    json.dump(MULTI_COLS, f)

with open("multi_answers.json", "w") as f:
    json.dump(All_MULTI_ANSWERS, f)

N_samples = len(df)
N_answers = len(All_MULTI_ANSWERS)
X_multi = np.zeros((N_samples, N_answers), dtype=float)

# Generate the X_multi matrix with two one-hot columns
for i in range(N_samples):
    answer_i = set()
    for col in MULTI_COLS:
        answer_i.update(process_multiselect(df.iloc[i][col]))
    for opt in answer_i:
        j = All_MULTI_ANSWERS_INDEX[opt]
        X_multi[i, j] = 1.0

print("X_multi.shape",X_multi.shape)

"""
Combine all types of data together.
"""
X_all = np.concatenate([X_text, X_Rating, X_multi], axis=1)
y = df["label"].to_numpy()

print(X_all.shape)
print(y.shape)