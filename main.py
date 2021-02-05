import numpy as np
from pyit2fls import TSK, IT2FS_Gaussian_UncertStd, IT2FS_plot, \
    product_t_norm, max_s_norm

from numpy import linspace
from sklearn import datasets


def normalize_dataset(dataset):
    # Normalize the dataset to [0, 1]
    min_arr = np.amin(dataset, axis=0)
    return (dataset - min_arr) / (np.amax(dataset, axis=0) - min_arr)


def apply_fuzzy_sets():
    domain = linspace(0.0, 1., 100)
    Short = IT2FS_Gaussian_UncertStd(domain, [0.15, 0.15, 0.1, 1.])
    Medium = IT2FS_Gaussian_UncertStd(domain, [0.5, 0.15, 0.1, 1.])
    Long = IT2FS_Gaussian_UncertStd(domain, [0.85, 0.15, 0.1, 1.])
    IT2FS_plot(Short, Medium, Long, title="Sets",
               legends=["Short", "Medium", "Long"])
    return [Short, Medium, Long]


def apply_rules_and_variables(IT2FLS, Short, Medium, Long):
    IT2FLS.add_input_variable("x1")
    IT2FLS.add_input_variable("x2")
    IT2FLS.add_input_variable("x3")
    IT2FLS.add_input_variable("x4")
    IT2FLS.add_output_variable("y1")

    # 1
    IT2FLS.add_rule([("x1", Short), ("x2", Long), ("x3", Short), ("x4", Short)],
                    [("y1", {"const": 0., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 2
    IT2FLS.add_rule([("x1", Short), ("x2", Long), ("x3", Medium), ("x4", Medium)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 3
    IT2FLS.add_rule([("x1", Short), ("x2", Short), ("x3", Short), ("x4", Short)],
                    [("y1", {"const": 0., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 4
    IT2FLS.add_rule([("x1", Short), ("x2", Short), ("x3", Medium), ("x4", Medium)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 5
    IT2FLS.add_rule([("x1", Short), ("x2", Medium), ("x3", Short), ("x4", Short)],
                    [("y1", {"const": 0., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 6
    IT2FLS.add_rule([("x1", Short), ("x2", Long), ("x3", Long), ("x4", Long)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 7
    IT2FLS.add_rule([("x1", Short), ("x2", Medium), ("x3", Long), ("x4", Long)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 7
    IT2FLS.add_rule([("x1", Short), ("x2", Medium), ("x3", Long), ("x4", Long)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 8
    IT2FLS.add_rule([("x1", Short), ("x2", Short), ("x3", Long), ("x4", Long)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 9
    IT2FLS.add_rule([("x1", Short), ("x2", Medium), ("x3", Medium), ("x4", Long)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 10
    IT2FLS.add_rule([("x1", Short), ("x2", Medium), ("x3", Long), ("x4", Medium)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 11
    IT2FLS.add_rule([("x1", Short), ("x2", Short), ("x3", Medium), ("x4", Long)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 12
    IT2FLS.add_rule([("x1", Short), ("x2", Short), ("x3", Long), ("x4", Medium)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 13
    IT2FLS.add_rule([("x1", Short), ("x2", Medium), ("x3", Medium), ("x4", Medium)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 14
    IT2FLS.add_rule([("x1", Short), ("x2", Long), ("x3", Medium), ("x4", Long)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 15
    IT2FLS.add_rule([("x1", Short), ("x2", Long), ("x3", Long), ("x4", Medium)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 16
    IT2FLS.add_rule([("x1", Medium), ("x2", Long), ("x3", Short), ("x4", Short)],
                    [("y1", {"const": 0., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 17
    IT2FLS.add_rule([("x1", Medium), ("x2", Long), ("x3", Medium), ("x4", Medium)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 18
    IT2FLS.add_rule([("x1", Medium), ("x2", Short), ("x3", Short), ("x4", Short)],
                    [("y1", {"const": 0., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 19
    IT2FLS.add_rule([("x1", Long), ("x2", Long), ("x3", Short), ("x4", Short)],
                    [("y1", {"const": 0., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 20
    IT2FLS.add_rule([("x1", Long), ("x2", Medium), ("x3", Short), ("x4", Short)],
                    [("y1", {"const": 0., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 21
    IT2FLS.add_rule([("x1", Long), ("x2", Long), ("x3", Long), ("x4", Long)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 22
    IT2FLS.add_rule([("x1", Long), ("x2", Medium), ("x3", Medium), ("x4", Medium)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 23
    IT2FLS.add_rule([("x1", Long), ("x2", Medium), ("x3", Long), ("x4", Medium)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 24
    IT2FLS.add_rule([("x1", Medium), ("x2", Short), ("x3", Medium), ("x4", Medium)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 25
    IT2FLS.add_rule([("x1", Long), ("x2", Medium), ("x3", Long), ("x4", Long)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 26
    IT2FLS.add_rule([("x1", Medium), ("x2", Short), ("x3", Medium), ("x4", Long)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 27
    IT2FLS.add_rule([("x1", Medium), ("x2", Short), ("x3", Long), ("x4", Medium)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 28
    IT2FLS.add_rule([("x1", Medium), ("x2", Medium), ("x3", Medium), ("x4", Medium)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 29
    IT2FLS.add_rule([("x1", Long), ("x2", Short), ("x3", Medium), ("x4", Medium)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 30
    IT2FLS.add_rule([("x1", Medium), ("x2", Short), ("x3", Long), ("x4", Long)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 31
    IT2FLS.add_rule([("x1", Medium), ("x2", Medium), ("x3", Long), ("x4", Medium)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 32
    IT2FLS.add_rule([("x1", Medium), ("x2", Medium), ("x3", Medium), ("x4", Long)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 33
    IT2FLS.add_rule([("x1", Long), ("x2", Short), ("x3", Long), ("x4", Medium)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 34
    IT2FLS.add_rule([("x1", Long), ("x2", Short), ("x3", Medium), ("x4", Long)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 35
    IT2FLS.add_rule([("x1", Medium), ("x2", Medium), ("x3", Long), ("x4", Long)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 36
    IT2FLS.add_rule([("x1", Long), ("x2", Short), ("x3", Long), ("x4", Long)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 37
    IT2FLS.add_rule([("x1", Long), ("x2", Medium), ("x3", Medium), ("x4", Long)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 38
    IT2FLS.add_rule([("x1", Medium), ("x2", Long), ("x3", Long), ("x4", Long)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 39
    IT2FLS.add_rule([("x1", Long), ("x2", Long), ("x3", Medium), ("x4", Long)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 40
    IT2FLS.add_rule([("x1", Long), ("x2", Long), ("x3", Long), ("x4", Medium)],
                    [("y1", {"const": 2., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 41
    IT2FLS.add_rule([("x1", Long), ("x2", Long), ("x3", Medium), ("x4", Medium)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 42
    IT2FLS.add_rule([("x1", Medium), ("x2", Long), ("x3", Long), ("x4", Medium)],
                    [("y1", {"const": 1., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 43
    IT2FLS.add_rule([("x1", Medium), ("x2", Medium), ("x3", Short), ("x4", Short)],
                    [("y1", {"const": 0., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    # 44
    IT2FLS.add_rule([("x1", Long), ("x2", Short), ("x3", Short), ("x4", Short)],
                    [("y1", {"const": 0., "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0})])
    return IT2FLS


def main():
    iris = datasets.load_iris()
    normalized_iris = normalize_dataset(iris.data)

    n_features = normalized_iris.shape[1]
    print("START")
    myIT2FLS = TSK(product_t_norm, max_s_norm)
    fuzzy_sets = apply_fuzzy_sets()
    myIT2FLS = apply_rules_and_variables(myIT2FLS, fuzzy_sets[0], fuzzy_sets[1], fuzzy_sets[2])

    # print(normalized_iris[149][0], normalized_iris[0][1], normalized_iris[0][2], normalized_iris[0][3])

    print(myIT2FLS.evaluate(
        {"x1": normalized_iris[149][0], "x2": normalized_iris[149][1], "x3": normalized_iris[149][2],
         "x4": normalized_iris[149][3]}))


if __name__ == "__main__":
    main()
