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

def build_data_matrix(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Construct data Matrix from dataframe based on three types of features:
    free texts, ratings, multiplicative questions
    """
    text_cols = [1, 6, 9]
    rating_cols = [2, 4, 7, 8]
    multi_cols = [3, 5]

    cols = df.columns

    free_texts = []
    for i in text_cols:
        col = cols[i]
        word_counts = (df[col].fillna("").astype(str).apply(lambda s: len(s.split())))
        words = word_counts.to_numpy(dtype=int)
        free_texts.append(words)

    ratings = []
    for i in rating_cols:
        col = cols[i]

        original_lst = df[col].apply(extract_rating)
        if not original_lst.mode().empty:
            mode_value = original_lst.mode().iloc[0]
        else:
            mode_value = 0

        nums = original_lst.fillna(mode_value).astype(float)
        nums = nums.to_numpy()
        ratings.append(nums)

    multiplicative_questions = []
    for i in multi_cols:
        col = cols[i]
        counts = (df[col].apply(parse_multiselect).apply(len).astype(float))
        counts = counts.to_numpy()
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
    print(x_matrix.shape)
    print(y.shape)

    np.save("x_matrix.npy", x_matrix)
    np.save("y.npy", y)

if __name__ == "__main__":
    main()