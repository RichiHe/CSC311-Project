import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_mlp_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    
):
    """
    Train an MLP classifier on the preprocessed TF-IDF data.

    Returns:
        mlp : trained MLPClassifier
    """

    mlp = MLPClassifier(
        hidden_layer_sizes=(64,),    # was (128,), now even smaller
        activation="relu",
        solver="adam",
        learning_rate_init=5e-3,
        alpha=0.001,                  # was 1e-3, increase L2 penalty
        max_iter=200,
        random_state=42,
        batch_size=64,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=5,
        
        )

    # Fit on training data
    mlp.fit(X_train, y_train)

    # Evaluate on training set
    y_train_pred = mlp.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Evaluate on validation set
    y_val_pred = mlp.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    print("Train accuracy:", train_acc)
    print("Validation accuracy:", val_acc)
    print("\nClassification report (validation):")
    print(classification_report(y_val, y_val_pred))
    print("\nConfusion matrix (validation):")
    print(confusion_matrix(y_val, y_val_pred))

    return mlp

def save_mlp_parameters(mlp, filepath="model/mlp_params.npz"):
    """
    Save MLP weights and biases from a trained sklearn MLPClassifier
    to a .npz file for later use in pred.py.
    """
    import os
    os.makedirs("model", exist_ok=True)

    params = {}
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        params[f"W{i+1}"] = W
        params[f"b{i+1}"] = b

    np.savez(filepath, **params)
    print(f"Saved MLP parameters to {filepath}")
    
    
if __name__ == "__main__":
    from data_preprocess_try import load_and_preprocess  

    X_train, X_val, y_train, y_val, vectorizer, scaler, label_encoder = load_and_preprocess(
        "training_data_clean.csv"
    )

    mlp = train_mlp_classifier(X_train, y_train, X_val, y_val)

    # After training, save parameters for pred.py
    save_mlp_parameters(mlp)
