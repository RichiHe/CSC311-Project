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

def parse_multiselect(response):
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

def build_data_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Construct data Matrix from dataframe based on three types of features:
    free texts, ratings, multiplicative questions
    """
    text_cols = [1, 6, 9] # the generative questions columns
    rating_cols = [2, 4, 7, 8] # the rating questions columns
    multi_cols = [3, 5] # the multiplicative choices questions columns

    cols = df.columns

    # generate a text vocabulary list
    free_texts = []
    for i in text_cols:
        col = cols[i]
        word_counts = (df[col].fillna("").astype(str).apply(lambda s: len(s.split())))
        words = word_counts.to_numpy(dtype=float)
        free_texts.append(words)

    # generate the list of rating integers
    ratings = []
    for i in rating_cols:
        col = cols[i]

        # if there is a missing value, fill with the mode value or 0 if the mode value does
        # not exist
        original_lst = df[col].apply(extract_rating)
        modes = original_lst.mode()
        mode_value = modes.iloc[0] if not modes.empty else 0

        nums = original_lst.fillna(mode_value).astype(float)
        nums = nums.to_numpy()
        ratings.append(nums)

    # generate multi_cols based on the number of answers
    multiplicative_questions = []
    for i in multi_cols:
        col = cols[i]
        counts_answers = (df[col].apply(parse_multiselect).apply(len).astype(float))
        counts = counts_answers.to_numpy()
        multiplicative_questions.append(counts)

    x_all = free_texts + ratings + multiplicative_questions
    x = np.column_stack(x_all)

    return x

def main():
    df = pd.read_csv(FILENAME)
    x_matrix = build_data_matrix(df)
    y = df["label"].to_numpy()

    print(x_matrix)
    print(y)
    print(x_matrix[0])
    print(y[0])
    print("The shape of data X matrix", x_matrix.shape)
    print(y.shape)

    np.save("x_matrix.npy", x_matrix)
    np.save("y.npy", y)

if __name__ == "__main__":
    main()