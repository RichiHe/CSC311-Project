"""
This module provides a complete workflow for training, tuning, and evaluating multiple classification models using
feature matrices generated from data_processing.py. Its capabilities include: performing hyperparameter searches,
comparing model performance across training/validation/test splits, generating summary tables and visualizations,
and producing detailed evaluation reports for top-performing models. The module also includes optional analysis tools
for exploring label-specific behaviors and linguistic variations within the dataset.
"""



from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from collections import Counter
import data_process_try_jackie as dp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_pred_data():
    X_train = np.load("X_train.npy")
    X_valid = np.load("X_val.npy")
    X_test = np.load("X_test.npy")

    y_train = np.load("y_train.npy")
    y_valid = np.load("y_val.npy")
    y_test = np.load("y_test.npy")

    return X_train, y_train, X_valid, y_valid, X_test, y_test

"""
predict
"""

def sample_predict():
    # X_train, y_train, X_valid, y_valid, X_test, y_test  = data_cleanup.data_matrix_setup()
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pred_data()

    # Initialize models
    models = {
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB()
    }

    # Train and evaluate models
    result = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_valid, model.predict(X_valid))
        result[name] = accuracy

    # Print results
    for key in result.keys():
        print("Model: ", key, "Accuracy: ", result[key])

    plt.figure(figsize=(6, 4))
    bars = plt.bar(result.keys(), result.values())
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}",
                 ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


def knn_predict():
    # X_train, y_train, X_valid, y_valid, X_test, y_test  = data_cleanup.data_matrix_setup()
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pred_data()

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    result = {}
    test_k = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30]
    for k in test_k:
        model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_valid, model.predict(X_valid))
        result[k] = accuracy

    plt.plot(result.keys(), result.values(), marker='o', linestyle='-')
    plt.title("KNN Model Performance")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    # test final performance
    # best_k = max(result, key=result.get)
    # model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    # model.fit(X_train, y_train)
    # print(f"Final accuracy for {best_k}-NN: ", accuracy_score(y_test, model.predict(scaler.transform(X_test))))


def bagging_knn():
    # X_train, y_train, X_valid, y_valid, X_test, y_test = data_cleanup.data_matrix_setup()
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pred_data()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    k_values = [1, 2, 3, 4, 5]
    bag_sizes = [1, 2, 3, 4, 5]
    results = []

    for bag_size in bag_sizes:
        models = []
        for i in range(bag_size):
            k = k_values[i % len(k_values)]
            model = KNeighborsClassifier(n_neighbors=k, weights='distance')
            model.fit(X_train, y_train)
            models.append(model)

        # Collect predictions
        all_pred = np.array([model.predict(X_valid) for model in models])

        # 这里是投票（标签是 "ChatGPT", "Claude", "Gemini"）
        pred = []
        categories = ["ChatGPT", "Claude", "Gemini"]
        for i in range(all_pred.shape[1]):
            votes = {c: 0 for c in categories}
            for p in all_pred[:, i]:
                votes[p] += 1
            majority = max(votes, key=votes.get)
            pred.append(majority)

        acc = accuracy_score(y_valid, pred)
        print(f"Bag size: {bag_size}, Accuracy: {acc}")
        results.append(acc)

    # Plot accuracy vs. bag size
    plt.figure(figsize=(6, 4))
    plt.plot(bag_sizes, results, marker='o')
    plt.xlabel("Number of KNN Models (Bag Size)")
    plt.ylabel("Validation Accuracy")
    plt.title("Bagging KNN Validation Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def rf_predict():
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pred_data()

    test_n = [150, 200, 250, 300, 350]
    test_d = [15, 20, 25, 30]

    result = []
    annot = []

    for n in test_n:
        row = []
        arow = []
        for d in test_d:
            print("Training:", n, d)
            model = RandomForestClassifier(n_estimators=n,
                                           max_depth=d,
                                           random_state=311)
            model.fit(X_train, y_train)
            val_acc = accuracy_score(y_valid, model.predict(X_valid))
            train_acc = accuracy_score(y_train, model.predict(X_train))
            row.append(val_acc)
            arow.append(f"{val_acc:.4f}\n({train_acc:.4f})")
        result.append(row)
        annot.append(arow)

    plt.figure(figsize=(8, 6))
    sns.heatmap(result, annot=annot, fmt="",
                xticklabels=test_d, yticklabels=test_n)
    plt.xlabel("max_depth")
    plt.ylabel("n_estimators")
    plt.title("Random Forest Accuracy (Validation / Train)")
    plt.tight_layout()
    plt.show()


def nn_predict():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pred_data()

    train_acc_list = []
    valid_acc_list = []
    EPOCHS = 25
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        batch_size=64,
        learning_rate_init=1e-4,
        max_iter=1,
        warm_start=True,
        random_state=311
    )

    # Train
    for epoch in range(EPOCHS):
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        valid_acc = accuracy_score(y_valid, model.predict(X_valid))
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        print(f"Epoch {epoch+1}: train={train_acc:.4f}, valid={valid_acc:.4f}")

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, EPOCHS + 1), train_acc_list, marker="o",
             label="Train Accuracy")
    plt.plot(range(1, EPOCHS + 1), valid_acc_list, marker="o",
             label="Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Neural Network Train vs Valid Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def linear_predict():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pred_data()

    Cs = [0.01, 0.03, 0.1, 0.3, 1, 3]

    for C in Cs:
        model = LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=20
        )
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_valid, model.predict(X_valid))
        print(C, val_acc)


def evaluate_model():
    # test set evaluate
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pred_data()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=311,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("Test accuracy: ", test_acc)

    from collections import Counter
    errors = Counter()
    for yt, yp in zip(y_test, y_pred):
        if yt != yp:
            errors[(yt, yp)] += 1

    print("\nMis-predictions:")
    for (true, pred), count in errors.most_common(10):
        print(f"  True={true}  Pred={pred}  Count={count}")

    print("\nMost common wrong target for each true label:")
    true_labels = sorted(set(y_test))
    for t in true_labels:
        sub = [(pred, c) for (tru, pred), c in errors.items() if tru == t]
        if not sub:
            print(f"  Label {t}: No errors")
            continue
        sub.sort(key=lambda x: -x[1])
        pred, c = sub[0]
        print(f"  Label {t}: most often confused with {pred} (count {c})")


def run_full_report():
    """
    Train, tune, and compare several classification models, then generate a full evaluation report.

    :param None: This function does not take any arguments and relies on load_pred_data() to fetch data.
    :return: None. Results are printed to stdout and plots are shown via matplotlib.
    """
    # get the data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_pred_data()

    # combine train + val for training and tuning
    X_train_full = np.concatenate([X_train, X_valid], axis=0)
    y_train_full = np.concatenate([y_train, y_valid], axis=0)

    results = {}  # name -> {"val_acc": , "test_acc": , "model": model}

    # Logistic Regression
    Cs = [0.01, 0.03, 0.1, 0.3, 1, 3]
    best_C = None
    best_val = -1.0

    print("\n=== Tuning Logistic Regression (C) ===")
    for C in Cs:
        log_clf = LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=200
        )
        log_clf.fit(X_train, y_train)
        val_acc = accuracy_score(y_valid, log_clf.predict(X_valid))
        print(f"C={C}: val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_C = C

    # use train+val retrain and evaluate on the test dataset again
    log_best = LogisticRegression(
        C=best_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=200
    )
    log_best.fit(X_train_full, y_train_full)
    test_acc = accuracy_score(y_test, log_best.predict(X_test))
    results["Logistic Regression"] = {
        "val_acc": best_val,
        "test_acc": test_acc,
        "model": log_best,
        "extra": {"C": best_C},
    }

    # Random Forest
    print("\n=== Tuning Random Forest (n_estimators, max_depth) ===")
    test_n = [150, 200, 250, 300, 350]
    test_d = [15, 20, 25, 30]

    best_params = None
    best_val = -1.0
    for n in test_n:
        for d in test_d:
            rf = RandomForestClassifier(
                n_estimators=n,
                max_depth=d,
                random_state=311,
            )
            rf.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, rf.predict(X_train))  # TODO: (Check later)
            val_acc = accuracy_score(y_valid, rf.predict(X_valid))
            print(f"n={n}, depth={d}: val_acc={val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                best_params = (n, d)

    # best hyperparameters for retraining
    best_n, best_d = best_params
    rf_best = RandomForestClassifier(
        n_estimators=best_n,
        max_depth=best_d,
        random_state=311,
    )
    rf_best.fit(X_train_full, y_train_full)
    test_acc = accuracy_score(y_test, rf_best.predict(X_test))
    results["Random Forest"] = {
        "val_acc": best_val,
        "test_acc": test_acc,
        "model": rf_best,
        "extra": {"n_estimators": best_n, "max_depth": best_d},
    }

    # KNN
    print("\n=== Tuning KNN (k) ===")
    # normalization
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)
    X_train_full_s = scaler.fit_transform(X_train_full)  # 用全部重新 fit
    X_test_s = scaler.transform(X_test)

    test_k = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30]
    best_k = None
    best_val = -1.0
    for k in test_k:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn.fit(X_train_s, y_train)
        val_acc = accuracy_score(y_valid, knn.predict(X_valid_s))
        print(f"k={k}: val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_k = k

    knn_best = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
    knn_best.fit(X_train_full_s, y_train_full)
    test_acc = accuracy_score(y_test, knn_best.predict(X_test_s))
    results["KNN"] = {
        "val_acc": best_val,
        "test_acc": test_acc,
        "model": (knn_best, scaler),
        "extra": {"k": best_k},
    }


    # graph
    print("\n=== Summary (use best setting for each model) ===")
    print(f"{'Model':<20} {'Best Val':>10} {'Test Acc':>10}  Extra")
    for name, info in results.items():
        print(f"{name:<20} {info['val_acc']:.4f}    {info['test_acc']:.4f}  {info['extra']}")

    # Test Accuracy comparasion graph
    model_names = list(results.keys())
    test_accs = [results[m]["test_acc"] for m in model_names]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(model_names, test_accs)
    plt.ylabel("Test Accuracy")
    plt.title("Model Comparison on Test Set")
    plt.ylim(0, 1)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01,
                 f"{yval:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()

    # Best model: confusion matrix + classification report
    best_model_name = max(results.keys(), key=lambda n: results[n]["test_acc"])
    best_info = results[best_model_name]
    print(f"\n=== Best model on test set: {best_model_name} ===")

    # KNN
    if best_model_name == "KNN":
        best_model, best_scaler = best_info["model"]
        X_test_used = best_scaler.transform(X_test)
        y_pred = best_model.predict(X_test_used)
    else:
        best_model = best_info["model"]
        y_pred = best_model.predict(X_test)

    labels = sorted(np.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({best_model_name})")
    plt.tight_layout()
    plt.show()

    print("\nClassification report for best model:")
    print(classification_report(y_test, y_pred, target_names=labels))


def analyze_claude_vs_gemini(min_word_count: int = 5):
    """
    Analyze differences between Claude and Gemini on the original CSV, including:
      1) Mean ratings for rating questions.
      2) Preference patterns for multi-select options.
      3) Differences in free-text length.
      4) Differences in high-frequency keywords.

    :param min_word_count: Minimum total frequency across both labels for a token to be included
                           in the keyword analysis, to filter out pure noise.
    :return: None. The analysis is printed to stdout.
    """
    # Load original data and keep only Claude / Gemini
    df = pd.read_csv(dp.FILENAME)
    df = df[df["label"].isin(["Claude", "Gemini"])].copy()
    print(f"Loaded {len(df)} rows with label in {{Claude, Gemini}}")

    labels = ["Claude", "Gemini"]

    # rating questions
    print("\n===== [Part 1] Rating 问题的 Claude / Gemini 差异 =====")
    cols = df.columns
    for col_idx in dp.RATING_COLS:
        col_name = cols[col_idx]
        tmp_col = f"rating_col_{col_idx}"
        df[tmp_col] = df.iloc[:, col_idx].apply(dp.extract_rating)

        print(f"\n[Rating Question] 列 {col_idx}: {col_name}")
        for lab in labels:
            sub = df[df["label"] == lab][tmp_col].dropna()
            if len(sub) == 0:
                print(f"  {lab}: 无有效数据")
                continue
            print(f"  {lab}: mean={sub.mean():.3f}, std={sub.std():.3f}, n={len(sub)}")

    # multi-select questions
    print("\n===== [Part 2] Multi-select 问题的 Claude / Gemini 差异 =====")
    V = len(dp.CANONICAL_MULTI_TYPES)

    for q_idx, col_idx in enumerate(dp.MULTI_COLS):
        col_name = cols[col_idx]
        print(f"\n[Multi Question #{q_idx+1}] 列 {col_idx}: {col_name}")

        # label -> option -> count
        label_option_count = {lab: Counter() for lab in labels}
        label_total = {lab: 0 for lab in labels}

        for _, row in df.iterrows():
            lab = row["label"]
            tasks = dp.parse_multiselect(row.iloc[col_idx])
            if len(tasks) == 0:
                continue
            label_total[lab] += 1
            for t in tasks:
                label_option_count[lab][t] += 1

        # Compute selection rates per option
        diff_list = []
        for opt in dp.CANONICAL_MULTI_TYPES:
            claude_cnt = label_option_count["Claude"][opt]
            gemini_cnt = label_option_count["Gemini"][opt]
            claude_rate = claude_cnt / label_total["Claude"] if label_total["Claude"] > 0 else 0.0
            gemini_rate = gemini_cnt / label_total["Gemini"] if label_total["Gemini"] > 0 else 0.0
            diff = claude_rate - gemini_rate
            diff_list.append((opt, claude_rate, gemini_rate, diff))

        print("  Option                                          Claude%    Gemini%    (Claude-Gemini)")
        for opt, c_r, g_r, diff in diff_list:
            print(f"  {opt:<45} {c_r:7.3f}   {g_r:7.3f}   {diff:7.3f}")

        # Show top-5 options with the largest absolute difference
        print("\n  >>> 按差值绝对值排序的 Top 5 选项：")
        diff_list_sorted = sorted(diff_list, key=lambda x: abs(x[3]), reverse=True)[:5]
        for opt, c_r, g_r, diff in diff_list_sorted:
            print(f"    {opt}: Claude={c_r:.3f}, Gemini={g_r:.3f}, diff={diff:+.3f}")

    # free-text length differences
    print("\n===== [Part 3] 自由文本长度的 Claude / Gemini 差异 =====")
    for col_idx in dp.TEXT_COLS:
        col_name = cols[col_idx]
        len_col = f"text_len_{col_idx}"
        df[len_col] = df.iloc[:, col_idx].apply(
            lambda x: len(dp.normalize_to_text_list(x))
        )

        print(f"\n[Text Question] 列 {col_idx}: {col_name}")
        for lab in labels:
            sub = df[df["label"] == lab][len_col]
            print(f"  {lab}: mean_len={sub.mean():.3f}, std={sub.std():.3f}, n={len(sub)}")

    # keyword preference differences
    print("\n===== [Part 4] 文本关键词的 Claude / Gemini 差异 =====")
    for col_idx in dp.TEXT_COLS:
        col_name = cols[col_idx]
        print(f"\n[Text Question] 列 {col_idx}: {col_name}")

        # Count word frequencies per label
        word_count = {lab: Counter() for lab in labels}
        total_tokens = {lab: 0 for lab in labels}

        for _, row in df.iterrows():
            lab = row["label"]
            tokens = dp.normalize_to_text_list(row.iloc[col_idx])
            word_count[lab].update(tokens)
            total_tokens[lab] += len(tokens)

        # Only keep words with total count >= min_word_count
        all_words = set(word_count["Claude"].keys()) | set(word_count["Gemini"].keys())
        stats = []
        for w in all_words:
            c_cnt = word_count["Claude"][w]
            g_cnt = word_count["Gemini"][w]
            if c_cnt + g_cnt < min_word_count:
                continue
            c_freq = c_cnt / total_tokens["Claude"] if total_tokens["Claude"] > 0 else 0.0
            g_freq = g_cnt / total_tokens["Gemini"] if total_tokens["Gemini"] > 0 else 0.0
            diff = c_freq - g_freq
            stats.append((w, c_cnt, g_cnt, c_freq, g_freq, diff))

        if not stats:
            print("  (没有达到频次阈值的词，可以调低 min_word_count 再试一次。)")
            continue

        # Words more preferred by Claude
        print("  >>> Claude 更常用的 Top 20 词（按 Claude_freq - Gemini_freq 排序）")
        top_claude = sorted(stats, key=lambda x: x[5], reverse=True)[:20]
        for w, c_cnt, g_cnt, c_f, g_f, diff in top_claude:
            print(f"    '{w}': Claude_cnt={c_cnt}, Gemini_cnt={g_cnt}, "
                  f"Claude_freq={c_f:.4f}, Gemini_freq={g_f:.4f}, diff={diff:+.4f}")

        # Words more preferred by Gemini
        print("\n  >>> Gemini 更常用的 Top 20 词")
        top_gemini = sorted(stats, key=lambda x: x[5])[:20]
        for w, c_cnt, g_cnt, c_f, g_f, diff in top_gemini:
            print(f"    '{w}': Claude_cnt={c_cnt}, Gemini_cnt={g_cnt}, "
                  f"Claude_freq={c_f:.4f}, Gemini_freq={g_f:.4f}, diff={diff:+.4f}")


if __name__ == "__main__":
    run_full_report()   # Run the full model comparison report
    print("\n==============================")
    print("模型评估完成，接下来做 Claude vs Gemini 差异分析")  # TODO: 分析结果
    print("==============================\n")

    # Claude / Gemini 差异分析（min_word_count 可以调）
    analyze_claude_vs_gemini(min_word_count=5)

    print("\nAll done.")
