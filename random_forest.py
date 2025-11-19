import json

from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import data_preprocess


class RandomForest:
    _X_train: ndarray
    _X_val: ndarray
    _y_train: ndarray
    _y_val: ndarray

    def __init__(self, file_name: str):
        self._X_train, self._X_val, _, self._y_train, self._y_val, _ = \
            data_preprocess.preprocess_train(file_name)

    def train_random_forest(self):
        print(f"X_train shape: {self._X_train.shape}")
        print(f"X_val shape: {self._X_val.shape}")

        # 创建随机森林模型
        rf_model = RandomForestClassifier(
            n_estimators=200,  # 树的数量
            max_depth=25,  # 控制树深度，防止过拟合
            min_samples_split=15,  # 节点分裂所需最小样本数
            min_samples_leaf=1,  # 叶节点最少样本数
        )
        # 训练模型
        rf_model.fit(self._X_train, self._y_train)

        y_pred = rf_model.predict(self._X_val)

        print(f"accuracy: {accuracy_score(self._y_val, y_pred):.4f}")
        print(f"macro avg: {f1_score(self._y_val, y_pred, average='macro'):.4f}")
        print("\ndetail report:")
        print(classification_report(self._y_val, y_pred))

        return rf_model, {
            'accuracy': accuracy_score(self._y_val, y_pred),
            'f1_macro': f1_score(self._y_val, y_pred, average='macro')
        }

    # Tuning
    def train_tuned_random_forest(self):
        from sklearn.model_selection import GridSearchCV

        # 定义参数网格
        param_grid = {
            'n_estimators': [200],
            'max_depth': [25],
            'min_samples_split': [5, 15, 25, 40],
            'min_samples_leaf': [1, 2, 4, 8]
        }

        rf = RandomForestClassifier(random_state=42)

        print("=== start grid search ===")
        grid_search = GridSearchCV(
            rf, param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(self._X_train, self._y_train)

        print(f"best params: {grid_search.best_params_}")
        print(f"best score: {grid_search.best_score_:.4f}")

        # 使用最佳模型在验证集上评估
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self._X_val)

        print(
            f"验证集F1分数: {f1_score(self._y_val, y_pred, average='macro'):.4f}")

        return best_model

    def save_model_for_predpy(self, model, filename_prefix='rf_model'):
        """
        提取随机森林参数并保存为pred.py可用的格式
        """
        model_params = {
            'n_estimators': model.n_estimators,
            'n_features': model.n_features_in_,
            'trees': []
        }

        # 提取每棵树的参数
        for i, tree in enumerate(model.estimators_):
            tree_data = {
                'node_count': tree.tree_.node_count,
                'children_left': tree.tree_.children_left.tolist(),
                'children_right': tree.tree_.children_right.tolist(),
                'feature': tree.tree_.feature.tolist(),
                'threshold': tree.tree_.threshold.tolist(),
                'value': tree.tree_.value.reshape(tree.tree_.node_count,
                                                  -1).tolist()
            }
            model_params['trees'].append(tree_data)

        # 保存为JSON文件（pred.py可以读取）
        with open(f'{filename_prefix}_params.json', 'w') as f:
            json.dump(model_params, f)

        print(f"model paras are stored in {filename_prefix}_params.json")
        return model_params

    def train_and_save(self):
        """
        训练并保存模型参数的完整流程
        """
        # 训练模型
        rf_model, metrics = self.train_random_forest()

        # 保存模型参数
        model_params = self.save_model_for_predpy(rf_model)

        return rf_model, metrics, model_params


randm_forest = RandomForest('training_data_clean.csv')
randm_forest.train_and_save()

# 如果需要调参版本，取消下面的注释
# tuned_model = randm_forest.train_tuned_random_forest()
