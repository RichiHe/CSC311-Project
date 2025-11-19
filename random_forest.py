import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import data_preprocess

X_train, X_val, _, y_train, y_val, _ = \
    data_preprocess.preprocess_train('training_data_clean.csv')

def train_random_forest():
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")

    # 创建随机森林模型
    rf_model = RandomForestClassifier(
        n_estimators=200,  # 树的数量
        max_depth=25,  # 控制树深度，防止过拟合
        min_samples_split=15,  # 节点分裂所需最小样本数
        min_samples_leaf=1,  # 叶节点最少样本数
    )
    # 训练模型
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_val)

    print(f"accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"宏平均F1: {f1_score(y_val, y_pred, average='macro'):.4f}")
    print("\n详细分类报告:")
    print(classification_report(y_val, y_pred))


    return rf_model, {
        'accuracy': accuracy_score(y_val, y_pred),
        'f1_macro': f1_score(y_val, y_pred, average='macro')
    }


# Tuning
def train_tuned_random_forest():
    from sklearn.model_selection import GridSearchCV

    # 定义参数网格
    param_grid = {
        'n_estimators': [200,],
        'max_depth': [25],
        'min_samples_split': [5, 15, 25, 40],
        'min_samples_leaf': [1, 2, 4, 8]
    }

    rf = RandomForestClassifier(random_state=42)

    print("=== 开始网格搜索 ===")
    grid_search = GridSearchCV(
        rf, param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    print(f"best params: {grid_search.best_params_}")
    print(f"best score: {grid_search.best_score_:.4f}")

    # 使用最佳模型在验证集上评估
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)

    print(f"验证集F1分数: {f1_score(y_val, y_pred, average='macro'):.4f}")

    return best_model


# 运行训练
if __name__ == "__main__":
    # 基础版本
    model, results = train_random_forest()

    # 如果需要调参版本，取消下面的注释
    tuned_model = train_tuned_random_forest()
