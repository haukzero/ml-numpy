import numpy as np


def dist(a, b):
    return ((a - b) ** 2).sum(axis=1) ** 0.5


class KMeans:
    def __init__(self, k):
        """
        k: 分类数
        clusters: (k, n_feature), 索引为标签值, 元素为对应聚点的坐标
        """
        self.k = k
        self.clusters = None

    def fit(self, X):
        # 初始聚点从样本中随机选取
        idx = np.random.choice(len(X), self.k, replace=False)
        self.clusters = X[ idx ]
        # 一开始把所有样本都分为 [0] 这一个标签
        labels = np.zeros(X.shape[ 0 ])

        while True:
            cnt = 0
            for i, x in enumerate(X):
                # 计算当前样本到每个聚点的距离, 选出最小距离的聚点作为当前样本的新标签
                d = dist(x, self.clusters)
                min_idx = np.argmin(d)
                if labels[ i ] != min_idx:
                    labels[ i ] = min_idx
                    cnt += 1

            # 若 cnt = 0, 说明此时分好类了, 迭代完成
            if cnt == 0:
                break

            # 更新聚点坐标
            for label in range(self.k):
                # 取被分为同一标签的样本的坐标中心作为新的聚点坐标
                centroid = X[ labels == label ].mean(axis=0)
                self.clusters[ label ] = centroid

        return labels

    def predict(self, X):
        pred = np.zeros(X.shape[ 0 ])
        for i, x in enumerate(X):
            d = dist(x, self.clusters)
            pred[ i ] = np.argmin(d)
        return pred


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    k = 3
    X, _ = make_blobs(n_samples=400, n_features=2, centers=k, cluster_std=1, random_state=10)

    km = KMeans(k)
    labels = km.fit(X)

    plt.scatter(X[ :, 0 ], X[ :, 1 ], c=labels)
    plt.show()
