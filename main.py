import numpy as np
from ga import genetic_algorithm
from pyit2fls import TSK, IT2FS_Gaussian_UncertStd, IT2FS_plot, \
    product_t_norm, max_s_norm, IT2FS_Gaussian_UncertMean
from sklearn.model_selection import train_test_split
from numpy import linspace
from sklearn import datasets


def normalize_dataset(dataset):
    # Normalize the dataset to [0, 1]
    min_arr = np.amin(dataset, axis=0)
    return (dataset - min_arr) / (np.amax(dataset, axis=0) - min_arr)


def apply_fuzzy_sets(ind):
    domain = linspace(0.0, 1., 100)
    Short = IT2FS_Gaussian_UncertMean(domain, [ind[0], ind[1], ind[2], 1.])
    Medium = IT2FS_Gaussian_UncertMean(domain, [ind[3], ind[4], ind[5], 1.])
    Long = IT2FS_Gaussian_UncertMean(domain, [ind[6], ind[7], ind[8], 1.])
    # IT2FS_plot(Short, Medium, Long, title="Sets",               legends=["Short", "Medium", "Long"])
    return [Short, Medium, Long]


def apply_rules_and_variables(IT2FLS, Short, Medium, Long):
    IT2FLS.rules.clear()
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


# s_center, s_spread, s_deviation
def evaluate_solution(tsk, individual, dataset, actuals):
    fs = apply_fuzzy_sets(individual)
    tsk = apply_rules_and_variables(tsk, fs[0], fs[1], fs[2])
    avg_error = 0
    for x in range(0, dataset.shape[0]):
        result = tsk.evaluate({"x1": dataset[x][0], "x2": dataset[x][1], "x3": dataset[x][2], "x4": dataset[x][3]})[
            'y1']
        error = abs(result - actuals[x])
        #   print(error)
        avg_error += error
    print("AVG_ERROR: ", avg_error / dataset.shape[0])
    return avg_error / dataset.shape[0]


'''
epoch 19, best fitness = 0.1085864007
[0.01097474 0.26553151 0.11786498 0.42229316 0.03308331 0.03509484
 0.85343658 0.11353733 0.08195224]
 Validated on 20% of data - Accuracy 100%
 with genetic_algorithm(fitness_func=ff, genotype_size=9, generation_size=30, generations=20, debug=True)
'''


def main():
    iris = datasets.load_iris()
    X = normalize_dataset(iris.data)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("START")
    myIT2FLS = TSK(product_t_norm, max_s_norm)
    # fuzzy_sets = apply_fuzzy_sets([0.15, 0.15, 0.1, 0.5, 0.15, 0.1, 0.85, 0.15, 0.1])
    # myIT2FLS = apply_rules_and_variables(myIT2FLS, fuzzy_sets[0], fuzzy_sets[1], fuzzy_sets[2])
    ff = lambda x: evaluate_solution(myIT2FLS, x, X_train, y_train)
    best, fbest = genetic_algorithm(fitness_func=ff, genotype_size=9, generation_size=30, generations=20, debug=True)

    fs = apply_fuzzy_sets(best)
    tsk = apply_rules_and_variables(myIT2FLS, fs[0], fs[1], fs[2])
    correct = 0
    for x in range(0, X_test.shape[0]):
        result = myIT2FLS.evaluate(
            {"x1": X_test[x][0], "x2": X_test[x][1], "x3": X_test[x][2],
             "x4": X_test[x][3]})['y1']
        print(result)
        print(X_test[x][0], X_test[x][1], X_test[x][2], X_test[x][3],
              y_test[x], round(result))
        if round(result) == y_test[x]:
            correct += 1
    print("Accuracy: ", correct / y_test.shape[0])


# for debugging
def main1():
    iris = datasets.load_iris()
    normalized_iris = normalize_dataset(iris.data)

    domain = linspace(0.0, 1., 100)
    Short = IT2FS_Gaussian_UncertMean(domain, [0.1, 0.1, 0.1, 1.])
    Medium = IT2FS_Gaussian_UncertMean(domain, [0.97, 0.07, 0.84, 1.])
    Long = IT2FS_Gaussian_UncertMean(domain, [0.7, 0.1, 0.1, 1.])
    IT2FS_plot(Short, Medium, Long, title="Sets", legends=["Short", "Medium", "Long"])

    n_features = normalized_iris.shape[1]
    print("START")
    myIT2FLS = TSK(product_t_norm, max_s_norm)
    evaluate_solution(myIT2FLS, [0.15, 0.15, 0.1, 0.5, 0.15, 0.1, 0.85, 0.15, 0.1], normalized_iris, iris.target)
    for x in range(0, normalized_iris.shape[0]):
        result = myIT2FLS.evaluate(
            {"x1": normalized_iris[x][0], "x2": normalized_iris[x][1], "x3": normalized_iris[x][2],
             "x4": normalized_iris[x][3]})['y1']
        print(result)
        print(normalized_iris[x][0], normalized_iris[x][1], normalized_iris[x][2], normalized_iris[x][3],
              iris.target[x], round(result))


if __name__ == "__main__":
    main()
