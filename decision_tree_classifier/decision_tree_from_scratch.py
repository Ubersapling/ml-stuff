import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo

class DecisionTreeNode():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value
        
class DecisionTreeClassifier():
    def __init__(self, max_depth):
        self.max_depth = None
        self.root = None
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y, current_depth=0)
        
    def predict(self, X):
        return [self._predict(self.root, x) for x in X]
    
    def _calculate_entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-15)) # +1e-15 prevents log(0)
        return entropy
    
    def _split_dataset(self, X, y, feature_index, threshold):
        left_indices = np.argwhere(X[:, feature_index] < threshold).flatten()
        right_indices = np.argwhere(X[:, feature_index] >= threshold).flatten()
        return left_indices, right_indices
    
    def _get_best_split(self, X, y):
        best_feature_index, best_threshold, best_info_gain, best_splits = None, None, -float("inf"), None
        n_samples, n_features = X.shape
        current_entropy = self._calculate_entropy(y)
        
        for feature_index in range(n_features):
            possible_thresholds = np.unique(X[:, feature_index])
            for threshold in possible_thresholds:
                left_indices, right_indices = self._split_dataset(X, y, feature_index, threshold)
                if len(left_indices) > 0 and len(right_indices) > 0:
                    entropy_left = self._calculate_entropy(y[left_indices])
                    entropy_right = self._calculate_entropy(y[right_indices])
                    weighted_entropy = len(left_indices) / n_samples * entropy_left + len(right_indices) / n_samples * entropy_right
                    info_gain = current_entropy - weighted_entropy
                    
                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_feature_index = feature_index
                        best_threshold = threshold
                        best_splits = (left_indices, right_indices)

        return best_feature_index, best_threshold, best_splits, best_info_gain
    
    def _build_tree(self, X, y, current_depth):
        num_samples, num_features = X.shape
        if num_samples == 1 or current_depth == self.max_depth:
            leaf_value = max(list(y), key=list(y).count)
            return DecisionTreeNode(value=leaf_value)
        
        feature_index, threshold, splits, info_gain = self._get_best_split(X, y)
        if splits is None:
            leaf_value = max(list(y), key=list(y).count)
            return DecisionTreeNode(value=leaf_value)
        
        left_indices, right_indices = splits
        left_subtree = self._build_tree(X[left_indices], y[left_indices], current_depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], current_depth + 1)
        
        return DecisionTreeNode(feature_index, threshold, left_subtree, right_subtree, info_gain)
    
    def _predict(self, node, X_test):
        if node.value is not None:
            return node.value
        if X_test[node.feature_index] < node.threshold:
            return self._predict(node.left, X_test)
        return self._predict(node.right, X_test)

def main():
    iris = fetch_ucirepo(id=53)
    X = iris.data.features.values
    y = iris.data.targets.values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=4)
    
    classifier = DecisionTreeClassifier(max_depth=3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()