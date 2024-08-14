import numpy as np


def split_data_by_feature_id(X, sample_ids, feature_id):
    """
    依据 feature_id 代表的特征把当前数据划分成多份
    :return: (dict) key=feature_val val=sample_ids
    """
    feature_dict = { }
    for sample_id in sample_ids:
        if X[ sample_id ][ feature_id ] not in feature_dict:
            feature_dict[ X[ sample_id ][ feature_id ] ] = [ ]
        feature_dict[ X[ sample_id ][ feature_id ] ].append(sample_id)
    return feature_dict


def same_in_features(X, sample_ids, feature_ids):
    if len(sample_ids) <= 1:
        return True
    for j in feature_ids:
        base = X[ sample_ids[ 0 ], j ]
        for i in range(1, len(sample_ids)):
            tmp_feature = X[ sample_ids[ i ], j ]
            if tmp_feature != base:
                return False
    return True


def same_in_label(y, sample_ids):
    if len(sample_ids) <= 1:
        return True
    label_list = [ ]
    for i in sample_ids:
        if y[ i ] not in label_list:
            label_list.append(y[ i ])
    return len(label_list) == 1


def cal_gini(y, sample_ids):
    # 获取在当前情况下每种标签的数量
    label_cnt = { }
    for sample_id in sample_ids:
        if y[ sample_id ] not in label_cnt:
            label_cnt[ y[ sample_id ] ] = 0
        label_cnt[ y[ sample_id ] ] += 1

    # gini = 1 - \sum p^2
    gini = 1
    for label in label_cnt:
        p = label_cnt[ label ] / len(sample_ids)
        gini -= p ** 2

    return gini


def cal_gini_index(X, y, sample_ids, feature_id):
    # index = \sum \frac{|D^v|}{|D|} Gini(D^v)
    index = 0
    feature_dict = split_data_by_feature_id(X, sample_ids, feature_id)
    for val in feature_dict:
        gini = cal_gini(y, feature_dict[ val ])
        index += (len(feature_dict[ val ]) / len(sample_ids)) * gini
    return index


def gini_metric(X, y, sample_ids, feature_ids):
    """
    返回能使得基尼指数最小的 feature id
    """
    min_gini, f_id = 0x7fff, feature_ids[ 0 ]
    for feature_id in feature_ids:
        index = cal_gini_index(X, y, sample_ids, feature_id)
        if index < min_gini:
            min_gini = index
            f_id = feature_id
    return f_id


def max_num_label(y, sample_ids):
    """
    返回当前标签列表中数量最多的标签
    """
    uni_label, cnt = np.unique(y[ sample_ids ], return_counts=True)
    max_cnt = np.argmax(cnt)
    return uni_label[ max_cnt ]


class TreeNode:
    def __init__(self):
        self.sample_ids = None
        self.feature_ids = None

        self.children = [ ]

        self.divided_feature_id = -1
        self.child_div_vals = [ ]

        self.final_label = None

    def append(self, child_node, child_div_val):
        self.children.append(child_node)
        self.child_div_vals.append(child_div_val)

    def is_leaf(self):
        return len(self.children) == 0

    def predict(self, x):
        """
        :param x: (1, n_features)
        :return: label
        """
        if self.is_leaf():
            return self.final_label

        div_feature_val = x[ self.divided_feature_id ]
        child_id = self.child_div_vals.index(div_feature_val)
        child_node = self.children[ child_id ]
        return child_node.predict(x)


class DecisionTree:
    def __init__(self):
        self.root = TreeNode()

    def _generate(self, X, y, cur_node: TreeNode):
        # 如果当前数据的所有标签都为同一个, 说明分好类了
        if same_in_label(y, cur_node.sample_ids):
            cur_node.final_label = y[ cur_node.sample_ids[ 0 ] ]
            return

        # 若特征集为空或取值相同, 选择剩余标签中数量最多的
        most_pos_label = max_num_label(y, cur_node.sample_ids)
        if ((not len(cur_node.sample_ids)) or
                same_in_features(X,
                                 cur_node.sample_ids,
                                 cur_node.feature_ids)):
            cur_node.final_label = most_pos_label
            return

        # 找到当前最优划分特征
        best_div_feature_id = gini_metric(X, y,
                                          cur_node.sample_ids,
                                          cur_node.feature_ids)
        cur_node.divided_feature_id = best_div_feature_id

        # 子结点的特征 id 列表去除当前最优划分特征 id, 相当于使用了这个特征
        sub_available_feature_ids = cur_node.feature_ids.copy()
        sub_available_feature_ids.remove(best_div_feature_id)

        feature_dict = split_data_by_feature_id(X,
                                                cur_node.sample_ids,
                                                best_div_feature_id)

        for val in feature_dict:
            sub_node = TreeNode()
            sub_feature_ids = sub_available_feature_ids.copy()
            sub_sample_ids = feature_dict[ val ]
            sub_node.sample_ids = sub_sample_ids
            sub_node.feature_ids = sub_feature_ids
            cur_node.append(sub_node, val)
            self._generate(X, y, sub_node)

    def fit(self, X, y):
        n_sample, n_feature = X.shape
        sample_ids = [ i for i in range(n_sample) ]
        feature_ids = [ i for i in range(n_feature) ]
        self.root.sample_ids = sample_ids
        self.root.feature_ids = feature_ids
        self._generate(X, y, self.root)

    def predict(self, X):
        pred = [ ]
        for x in X:
            pred.append(self.root.predict(x))
        return np.array(pred)

    def score(self, X, y):
        pred = self.predict(X)
        return (pred == y).mean()


if __name__ == '__main__':
    data = np.array([
        [ '色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好坏' ],
        [ '青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜' ],
        [ '乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜' ],
        [ '乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜' ],
        [ '青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜' ],
        [ '浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜' ],
        [ '青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜' ],
        [ '乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜' ],
        [ '乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜' ],
        [ '乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜' ],
        [ '青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜' ],
        [ '浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜' ],
        [ '浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜' ],
        [ '青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜' ],
        [ '浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜' ],
        [ '乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜' ],
        [ '浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜' ],
        [ '青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜' ],
    ])

    X, y = data[ :, :-1 ], data[ :, -1 ]
    dt = DecisionTree()
    dt.fit(X, y)
    print(dt.score(X, y))
