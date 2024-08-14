import numpy as np


def cal_gaussian(val, mean, var, eps=1e-8):
    return np.exp(-(val - mean) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var + eps)


class NativeBayes:
    def __init__(self):
        """
        label_dict: key 为标签, val 为标签所占比例 p(y = c)
        params: [n_label, n_feature, 2), 两个值分别为 mean, var
        """
        self.label_dict = { }
        self.params = [ ]

    def fit(self, X, y):
        n = X.shape[ 0 ]
        labels, counts = np.unique(y, return_counts=True)
        for label, count in zip(labels, counts):
            self.label_dict[ label ] = count / n

        self.params = [ [ ] for _ in range(labels.shape[ 0 ]) ]
        for i, label in enumerate(labels):
            x_c = X[ y == label ]
            for col in x_c.T:
                self.params[ i ].append([ col.mean(), col.var() ])

    def _cal_pred(self, x):
        # x: (1, n_feature)
        prob = [ ]
        for i, label in enumerate(self.label_dict):
            p = self.label_dict[ label ]

            # p(x | y = c) = \Pi p(x = x_i | y = c)
            for feature_val, param in zip(x, self.params[ i ]):
                # use gaussian to avoid mul by 0
                p *= cal_gaussian(feature_val, param[ 0 ], param[ 1 ])

            prob.append(p)

        labels = list(self.label_dict.keys())
        return labels[ np.argmax(prob) ]

    def predict(self, X):
        return np.array([ self._cal_pred(x) for x in X ])

    def score(self, X, y):
        pred = self.predict(X)
        return (pred == y).mean()


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    iris = load_iris()
    data = iris.data
    target = iris.target
    shuffle_idx = np.random.permutation(data.shape[ 0 ])
    data, target = data[ shuffle_idx ], target[ shuffle_idx ]
    n_train = int(data.shape[ 0 ] * 0.8)
    train_x, test_x = data[ :n_train ], data[ n_train: ]
    train_y, test_y = target[ :n_train ], target[ n_train: ]

    nb = NativeBayes()
    nb.fit(train_x, train_y)

    print(f"Test Accuracy: {nb.score(test_x, test_y)}")
