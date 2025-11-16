import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib  # 用于保存模型

# 1. 加载和准备数据
# 假设你的数据已经预处理并组合成特征矩阵
# 特征矩阵形状: [n_samples, n_features]
# 其中n_features = TF-IDF维度 + one-hot维度 + 1(评分)

# 示例数据加载（根据你的实际情况修改）
# X = np.load('your_features.npy')  # 特征矩阵
# y = np.load('your_labels.npy')    # 标签

# 或者从DataFrame加载
# df = pd.read_csv('your_data.csv')
# X = df.drop('target_column', axis=1).values
# y = df['target_column'].values

# 2. 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 保持类别分布
)

print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")
print(f"类别分布: {np.unique(y_train, return_counts=True)}")

# 3. 创建并训练随机森林模型
rf_model = RandomForestClassifier(
    n_estimators=100,      # 树的数量
    max_depth=15,          # 树的最大深度，防止过拟合
    min_samples_split=5,   # 内部节点再划分所需最小样本数
    min_samples_leaf=2,    # 叶节点最少样本数
    max_features='sqrt',   # 每次分裂考虑的特征数，sqrt是常用选择
    bootstrap=True,        # 使用自助采样
    random_state=42,       # 固定随机种子，确保结果可重现
    n_jobs=-1,            # 使用所有CPU核心并行训练
    verbose=1             # 显示训练进度
)

print("开始训练随机森林...")
rf_model.fit(X_train, y_train)
print("训练完成!")

# 4. 在验证集上评估模型
y_pred = rf_model.predict(X_val)
y_pred_proba = rf_model.predict_proba(X_val)  # 获取预测概率

print("\n=== 模型评估结果 ===")
print(f"准确率: {accuracy_score(y_val, y_pred):.4f}")
print(f"宏平均F1: {f1_score(y_val, y_pred, average='macro'):.4f}")
print("\n详细分类报告:")
print(classification_report(y_val, y_pred))

# 5. 分析特征重要性（如果你的特征有具体含义）
feature_importance = rf_model.feature_importances_
print(f"\n最重要的10个特征索引: {np.argsort(feature_importance)[-10:][::-1]}")
print(f"对应的重要性分数: {np.sort(feature_importance)[-10:][::-1]}")

# 6. 保存模型（用于后续分析或pred.py参考）
joblib.dump(rf_model, 'random_forest_model.pkl')
print("\n模型已保存为 'random_forest_model.pkl'")

# 7. 可选：交叉验证（更稳健的性能评估）
from sklearn.model_selection import cross_val_score
print("\n进行5折交叉验证...")
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='f1_macro')
print(f"交叉验证F1分数: {cv_scores}")
print(f"平均交叉验证F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
