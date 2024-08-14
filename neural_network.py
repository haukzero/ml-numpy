import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


def batch_norm(x, eps=1e-8):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean) / (x_std + eps)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=1e-2, eps=1e-8):
        self.input_size = input_size + 1
        self.hidden_size = hidden_size + 1
        self.output_size = output_size
        self.lr = lr
        self.eps = eps
        self.theta_x = np.random.randn(self.input_size, self.hidden_size)
        self.theta_h = np.random.randn(self.hidden_size, self.output_size)

    def train_one_step(self, X, y):
        # x: (batch_size, n_feature + 1)
        x = np.concatenate((X, np.ones((X.shape[ 0 ], 1))), axis=1)
        # y: (batch_size, 1)
        y = y.reshape(-1, 1)
        # h: (batch_size, hidden_size)
        h = sigmoid(x @ self.theta_x)
        # o: (batch_size, n_label)
        o = sigmoid(h @ self.theta_h)

        # output_error: (batch_size, n_label)
        output_error = y - o
        # output_delta: (batch_size, n_label)
        output_delta = output_error * d_sigmoid(o)
        # hidden_error: (batch_size, hidden_size)
        hidden_error = output_delta @ self.theta_h.T
        # hidden_delta: (batch_size, hidden_size)
        hidden_delta = hidden_error * d_sigmoid(h)

        self.theta_x += self.lr * x.T @ hidden_delta
        self.theta_h += self.lr * h.T @ output_delta

        # loss = (1/m) * \sum (-(y * log(h) + (1 - y) * log(1 - h)))
        loss = - np.sum(y * np.log2(o + self.eps) + (1 - y) * np.log2(1 - o + self.eps)) / x.shape[ 0 ]
        return loss

    def fit(self, X, y, max_iter=100):
        losses = [ ]
        for i in range(max_iter):
            loss = self.train_one_step(X, y)
            losses.append(loss)
            if (i + 1) % 10 == 0:
                print(f"Epoch {i + 1}/{max_iter} Loss: {loss}")
        return losses

    def predict(self, X):
        x = np.concatenate((X, np.ones((X.shape[ 0 ], 1))), axis=1)
        h = sigmoid(x @ self.theta_x)
        o = sigmoid(h @ self.theta_h)
        pred = np.where(o >= 0.5, 1, 0)
        return pred.reshape(-1)

    def score(self, X, y):
        pred = self.predict(X)
        return (pred == y).mean()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    iris = load_iris()
    data = iris.data[ :100 ]
    target = iris.target[ :100 ]
    shuffle_idx = np.random.permutation(data.shape[ 0 ])
    data, target = data[ shuffle_idx ], target[ shuffle_idx ]
    n_train = int(data.shape[ 0 ] * 0.8)
    train_x, test_x = data[ :n_train ], data[ n_train: ]
    train_y, test_y = target[ :n_train ], target[ n_train: ]
    train_x, test_x = batch_norm(train_x), batch_norm(test_x)

    nn = NeuralNetwork(train_x.shape[ 1 ], 64, 1, lr=1e-2)
    losses = nn.fit(train_x, train_y)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()

    print(f"Test Accuracy: {nn.score(test_x, test_y)}")
