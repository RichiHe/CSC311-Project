import json

from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import data_preprocess


def save_model_for_predpy(model, filename_prefix='rf_model'):
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


class RandomForest:
    _X_train: ndarray
    _X_val: ndarray
    _X_test: ndarray
    _y_train: ndarray
    _y_val: ndarray
    _y_test: ndarray
    _model: RandomForestClassifier

    def __init__(self, file_name: str, dict_len: int):
        self._X_train, self._X_val, self._X_test, self._y_train, self._y_val, self._y_test = \
            data_preprocess.preprocess_train(file_name, dict_len)
        self._model = None

    def train_random_forest(self):
        print(f"X_train shape: {self._X_train.shape}")
        print(f"X_val shape: {self._X_val.shape}")

        # 创建随机森林模型
        rf_model = RandomForestClassifier(
            n_estimators=50,  # 树的数量
            max_depth=15,  # 控制树深度，防止过拟合
            min_samples_split=15,  # 节点分裂所需最小样本数
            min_samples_leaf=1,  # 叶节点最少样本数
            random_state=42,
        )
        # 训练模型
        rf_model.fit(self._X_train, self._y_train)

        y_pred = rf_model.predict(self._X_val)

        print(f"accuracy: {accuracy_score(self._y_val, y_pred):.4f}")
        print(
            f"macro avg: {f1_score(self._y_val, y_pred, average='macro'):.4f}")
        print("\ndetail report:")
        print(classification_report(self._y_val, y_pred))

        self._model = rf_model

        return rf_model, {
            'accuracy': accuracy_score(self._y_val, y_pred),
            'f1_macro': f1_score(self._y_val, y_pred, average='macro')
        }

    def test_prediction(self):
        y_pred = self._model.predict(self._X_test)

        print(f"accuracy: {accuracy_score(self._y_test, y_pred):.4f}")
        print(
            f"macro avg: {f1_score(self._y_test, y_pred, average='macro'):.4f}")
        print("\ndetail report:")
        print(classification_report(self._y_test, y_pred))

    def _print_metrics(self, y_true, y_pred, dataset_name: str):
        """
        统一的指标打印函数，同时显示准确率和F1分数
        """
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        # print(f"\n=== {dataset_name} estimate result ===")
        # print(f"Accuracy: {accuracy:.4f} ← 主要选择指标")
        # print(f"F1 Macro: {f1_macro:.4f}")
        # print(f"F1 Weighted: {f1_weighted:.4f}")
        # print("\ndetail report:")
        # print(classification_report(y_true, y_pred))

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }

    # Tuning
    def train_tuned_random_forest(self):
        from sklearn.model_selection import GridSearchCV

        # 定义参数网格
        param_grid = {
            'n_estimators': [50, 100, 150, 200, 250],
            'max_depth': [5, 10, 15, 20, 25],
            'min_samples_split': [5, 15, 25, 40],
            'min_samples_leaf': [1, 2, 4, 8],
            # 'max_features': ['sqrt', 'log2', None],
            'random_state': [42]
        }

        rf = RandomForestClassifier(random_state=42)

        print("=== start grid search ===")
        grid_search = GridSearchCV(
            rf, param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(self._X_train, self._y_train)
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        print(f"best params: {best_params}")
        print(f"best score: {best_score:.4f}")

        # 使用最佳模型在验证集上评估
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self._X_val)

        val_metrics = self._print_metrics(self._y_val, y_pred,
                                          "tuned validation set")

        return best_score, best_params, val_metrics

    def train_and_save(self):
        """
        训练并保存模型参数的完整流程
        """
        # 训练模型
        rf_model, metrics = self.train_random_forest()

        # 保存模型参数
        model_params = save_model_for_predpy(rf_model)

        return rf_model, metrics, model_params


if __name__ == "__main__":
    while 1:
        user_input = input()
        if user_input == "1":
            global_best_score = 0
            global_best_params = {}
            global_best_dict_len = 0
            global_best_metrix = None
            for i in range(80, 130, 10):
                random_forest = RandomForest('training_data_clean.csv', i)
                # random_forest.train_and_save()

                # random_forest.train_random_forest()
                # 如果需要调参版本，取消下面的注释
                best_score, best_params, best_metrix = random_forest.train_tuned_random_forest()
                if best_score > global_best_score:
                    global_best_score = best_score
                    global_best_params = best_params
                    global_best_metrix = best_metrix
                    global_best_dict_len = i

            print(f"global best params: {global_best_params}")
            print(f"global best score: {global_best_score:.4f}")
            print(f"global best metrix: {global_best_metrix}")
            print(f"best dict len: {global_best_dict_len}")
        elif user_input == "2":
            random_forest = RandomForest('training_data_clean.csv', 80)
            random_forest.train_random_forest()
            random_forest.test_prediction()
        elif user_input == "3":
            random_forest = RandomForest('training_data_clean.csv', 80)
            random_forest.train_and_save()
