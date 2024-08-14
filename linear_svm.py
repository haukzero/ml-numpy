import numpy as np


def batch_norm(x, eps=1e-8):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean) / (x_std + eps)


class LinearSVM:
    def __init__(self, n_features, lr=1e-2, alpha=1e-3):
        self.lr = lr
        self.alpha = alpha
        self.W = np.random.rand(1, n_features)
        self.b = np.random.rand(1)

    def fit(self, X, y, epochs=100):
        y = np.sign(y)
        for _ in range(epochs):
            for i, x in enumerate(X):
                x = x.reshape(-1, 1)
                h = (self.W @ x + self.b).item()
                condition = (y[ i ] * h) >= 0
                if condition:
                    self.W -= self.lr * self.W
                else:
                    self.W -= self.lr * (self.W - self.alpha * y[ i ] * x.T)
                    self.b -= self.lr * self.alpha * y[ i ]

    def predict(self, X):
        h = self.W @ X.T + self.b
        return np.sign(h)

    def score(self, X, y):
        y = np.sign(y).reshape(1, -1)
        pred = self.predict(X)
        return np.sum(y == pred) / y.shape[ -1 ]


if __name__ == '__main__':
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.2)
    n_train = int(X.shape[ 0 ] * 0.8)
    train_x, test_x = X[ :n_train ], X[ n_train: ]
    train_y, test_y = y[ :n_train ], y[ n_train: ]
    train_x, test_x = batch_norm(train_x), batch_norm(test_x)

    ls = LinearSVM(2)
    ls.fit(train_x, train_y)

    print(f"Test Accuracy: {ls.score(test_x, test_y)}")
