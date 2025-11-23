import json
import numpy as np

from random_forest import RandomForest


class ManualRandomForest:
    def __init__(self, model_params_path):
        # 加载模型参数
        with open(model_params_path, 'r') as f:
            self.params = json.load(f)

        self.n_estimators = self.params['n_estimators']
        self.classes = ["ChatGPT", "Claude", "Gemini"]
        self.trees = self.params['trees']

    def predict_single_tree(self, tree, X):
        """对单棵树进行预测"""
        node = 0
        while tree['children_left'][node] != -1:
            if X[tree['feature'][node]] <= tree['threshold'][node]:
                node = tree['children_left'][node]
            else:
                node = tree['children_right'][node]

        # 到达叶节点，返回预测概率
        node_value = np.array(tree['value'][node])
        return node_value / node_value.sum()

    def predict(self, X):

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        all_predictions = []

        for x in X:
            tree_predictions = []

            for tree in self.trees:
                probs = self.predict_single_tree(tree, x)
                tree_predictions.append(probs)

            avg_probs = np.mean(tree_predictions, axis=0)
            predicted_class = self.classes[np.argmax(avg_probs)]
            all_predictions.append(predicted_class)

        return np.array(all_predictions)




def validate_manual_model():
    rf = RandomForest('training_data_clean.csv', 80)
    trained_model, metrics, params = rf.train_and_save()

    original_predictions = trained_model.predict(rf._X_val[:100])

    manual_rf = ManualRandomForest('rf_model_params.json')
    manual_predictions = manual_rf.predict(rf._X_val[:100])

    print("original model prediction:", original_predictions)
    print("manual model prediction:", manual_predictions)
    print("prediction uniformity:", np.all(original_predictions == manual_predictions))


if __name__ == '__main__':
    validate_manual_model()
