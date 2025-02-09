import numpy as np
from collections import Counter


__all__ = ['DecisionTree', 'find_best_split']


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feature_vector = np.asarray(feature_vector)
    target_vector = np.asarray(target_vector)

    if len(feature_vector) != len(target_vector):
        raise ValueError("Length of feature vector and target vector must match")

    n = feature_vector.shape[0]
    if n == 0:
        return np.array([]), np.array([]), None, None

    sorted_idx = np.argsort(feature_vector)
    sorted_feature_vector = feature_vector[sorted_idx]
    sorted_target_vector = target_vector[sorted_idx]

    diff = np.diff(sorted_feature_vector)
    valid_idx = np.where(diff != 0)[0]

    if valid_idx.size == 0:
        return np.array([]), np.array([]), None, None

    thresholds = (sorted_feature_vector[valid_idx] + sorted_feature_vector[valid_idx + 1]) / 2.0
    cum_ones_sum = np.cumsum(sorted_target_vector)

    left_counts = valid_idx + 1
    left_ones = cum_ones_sum[valid_idx]

    total_counts = n
    total_ones = cum_ones_sum[-1]
    right_counts = total_counts - left_counts
    right_ones = total_ones - left_ones

    p_left = left_ones / left_counts
    p_right = right_ones / right_counts

    H_left = 1 - (1 - p_left)**2 - p_left**2
    H_right = 1 - (1 - p_right)**2 - p_right**2

    # в условии задания `-`, поэтому будем максимизировать общий отрицательный Джини (т.е. чем больше общий критерий Джини, тем
    # более однородные данные относительно целевой переменной)
    ginis = - (left_counts / total_counts) * H_left - (right_counts / total_counts) * H_right
    max_gini = np.max(ginis)
    best_candidates_thresholds = np.where(ginis == max_gini)[0]

    best_idx = best_candidates_thresholds[np.argmin(thresholds[best_candidates_thresholds])]

    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if len(np.unique(sub_y)) == 1:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, best_split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    current_click = clicks.get(key, 0)
                    ratio[key] = current_count / (current_click + 1e-3)

                sorted_categories = list(map(lambda x: x[0],
                                             sorted(ratio.items(), key=lambda x: x[1])))

                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError("Unknown feature type")

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if threshold is None:
                continue

            current_split = feature_vector < threshold

            if current_split.sum() < self._min_samples_leaf or (
                    len(sub_y) - current_split.sum()) < self._min_samples_leaf:
                continue

            if gini_best is None or (gini is not None and gini > gini_best):
                feature_best = feature
                gini_best = gini
                best_split = current_split

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError("Unknown feature type")

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError("Unknown feature type")

        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(
            sub_X[best_split],
            sub_y[best_split],
            node["left_child"],
            depth + 1
        )
        self._fit_node(
            sub_X[np.logical_not(best_split)],
            sub_y[np.logical_not(best_split)],
            node["right_child"],
            depth + 1
        )

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Unknown feature type")

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=True):
        if deep:
            return {
                "feature_types": self._feature_types,
                "max_depth": self._max_depth,
                "min_samples_split": self._min_samples_split,
                "min_samples_leaf": self._min_samples_leaf
            }
        else:
            return {
                "feature_types": self._feature_types,
                "max_depth": self._max_depth
            }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self._tree

        if node["type"] == "terminal":
            print(indent + "Лист: class=" + str(node["class"]))
        else:
            feature = node["feature_split"]
            if "threshold" in node:
                print(indent + f"Node: feature[{feature}] < {node['threshold']}")
            elif "categories_split" in node:
                print(indent + f"Node: feature[{feature}] in {node['categories_split']}")

            print(indent + "├─ Left:")
            self.print_tree(node["left_child"], indent + "│  ")

            print(indent + "└─ Right:")
            self.print_tree(node["right_child"], indent + "   ")

