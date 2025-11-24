import pandas as pd
import numpy as np
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess(
    csv_path: str,
    max_features: int = 400,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = False,
):
    """
    Load the CSC311 project data and preprocess it for an MLP.

    Returns:
        X_train, X_val : (scipy sparse matrix or numpy array)
        y_train, y_val : (numpy arrays, int labels)
        vectorizer     : fitted TfidfVectorizer
        scaler         : fitted StandardScaler or None
        label_encoder  : fitted LabelEncoder
    """

    # 1. Load data
    df = pd.read_csv(csv_path)

    # 2. Drop non-predictive columns
    # Make sure we don't error if student_id is already removed
    cols_to_drop = [c for c in ["student_id"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 3. Handle missing values by replacing with empty string
    df = df.fillna("")

    # 4. Separate features and label
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the dataset.")

    y_raw = df["label"].values
    feature_cols = [c for c in df.columns if c != "label"]

    # 5. Combine all text/categorical answers into one big text field
    #    (concatenate all feature columns as strings)
    df["text"] = df[feature_cols].astype(str).agg(" ".join, axis=1)

    # 6. TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        
        max_features=max_features,
        ngram_range=(1, 1),      # unigrams + bigrams
        stop_words="english",     # remove common English words
        min_df = 3,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df["text"])

    # 7. Encode labels as integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # 8. Train/validation split (stratified by label)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # 9. Optional feature scaling (helpful for MLP)
    scaler = None
    if scale_features:
        scaler = StandardScaler(with_mean=False)  # with_mean=False works with sparse
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val, vectorizer, scaler, label_encoder


def save_preprocessing_artifacts(vectorizer, scaler, label_encoder, path="preprocess"):
    os.makedirs(path, exist_ok=True)

    
    # Convert numpy.int64 values to plain Python int
    vocab = {term: int(idx) for term, idx in vectorizer.vocabulary_.items()}

    # Save vocabulary as JSON
    with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    # Save IDF
    np.save(os.path.join(path, "idf.npy"), vectorizer.idf_)

    # Save scaler params (if exists)
    if scaler is not None:
        if hasattr(scaler, "scale_"):
            np.save(os.path.join(path, "scaler_scale.npy"), scaler.scale_)
        if hasattr(scaler, "mean_"):
            np.save(os.path.join(path, "scaler_mean.npy"), scaler.mean_)

    # Save label classes
    np.save(os.path.join(path, "classes.npy"), label_encoder.classes_)

    print("Saved preprocessing artifacts.")

if __name__ == "__main__":
    
    X_train, X_val, y_train, y_val, vectorizer, scaler, label_encoder = load_and_preprocess(
        "training_data_clean.csv"
    )

    save_preprocessing_artifacts(vectorizer, scaler, label_encoder)

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_train distribution:", np.bincount(y_train))
    print("Classes:", label_encoder.classes_)

