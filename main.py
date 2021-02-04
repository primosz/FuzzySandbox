import numpy as np
from pyit2fls import TSK, IT2FS_Gaussian_UncertStd, IT2FS_plot, \
    product_t_norm, max_s_norm

from numpy import linspace
from sklearn import datasets


def normalize_dataset(dataset):
	# Normalize the dataset to [0, 1]
	min_arr = np.amin(dataset, axis=0)
	return (dataset - min_arr) / (np.amax(dataset, axis=0) - min_arr)

domain = linspace(0.0, 1., 100)

Short = IT2FS_Gaussian_UncertStd(domain, [0.15, 0.15, 0.1, 1.])
Medium = IT2FS_Gaussian_UncertStd(domain, [0.5, 0.15, 0.1, 1.])
Long = IT2FS_Gaussian_UncertStd(domain, [0.85, 0.15, 0.1, 1.])
IT2FS_plot(Short, Medium, Long, title="Sets",
           legends=["Short", "Medium", "Long"])

myIT2FLS = TSK(product_t_norm, max_s_norm)

myIT2FLS.add_input_variable("x1")
myIT2FLS.add_input_variable("x2")
myIT2FLS.add_input_variable("x3")
myIT2FLS.add_input_variable("x4")
myIT2FLS.add_output_variable("y1")

#1
myIT2FLS.add_rule([("x1", Short), ("x2", Long), ("x3", Short), ("x4", Short)],
                  [("y1", {"const": 0., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#2
myIT2FLS.add_rule([("x1", Short), ("x2", Long), ("x3", Medium), ("x4", Medium)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#3
myIT2FLS.add_rule([("x1", Short), ("x2", Short), ("x3", Short), ("x4", Short)],
                  [("y1", {"const": 0., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#4
myIT2FLS.add_rule([("x1", Short), ("x2", Short), ("x3", Medium), ("x4", Medium)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#5
myIT2FLS.add_rule([("x1", Short), ("x2", Medium), ("x3", Short), ("x4", Short)],
                  [("y1", {"const": 0., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#6
myIT2FLS.add_rule([("x1", Short), ("x2", Long), ("x3", Long), ("x4", Long)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#7
myIT2FLS.add_rule([("x1", Short), ("x2", Medium), ("x3", Long), ("x4", Long)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#7
myIT2FLS.add_rule([("x1", Short), ("x2", Medium), ("x3", Long), ("x4", Long)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#8
myIT2FLS.add_rule([("x1", Short), ("x2", Short), ("x3", Long), ("x4", Long)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#9
myIT2FLS.add_rule([("x1", Short), ("x2", Medium), ("x3", Medium), ("x4", Long)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#10
myIT2FLS.add_rule([("x1", Short), ("x2", Medium), ("x3", Long), ("x4", Medium)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#11
myIT2FLS.add_rule([("x1", Short), ("x2", Short), ("x3", Medium), ("x4", Long)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#12
myIT2FLS.add_rule([("x1", Short), ("x2", Short), ("x3", Long), ("x4", Medium)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#13
myIT2FLS.add_rule([("x1", Short), ("x2", Medium), ("x3", Medium), ("x4", Medium)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#14
myIT2FLS.add_rule([("x1", Short), ("x2", Long), ("x3", Medium), ("x4", Long)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#15
myIT2FLS.add_rule([("x1", Short), ("x2", Long), ("x3", Long), ("x4", Medium)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#16
myIT2FLS.add_rule([("x1", Medium), ("x2", Long), ("x3", Short), ("x4", Short)],
                  [("y1", {"const": 0., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#17
myIT2FLS.add_rule([("x1", Medium), ("x2", Long), ("x3", Medium), ("x4", Medium)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#18
myIT2FLS.add_rule([("x1", Medium), ("x2", Short), ("x3", Short), ("x4", Short)],
                  [("y1", {"const": 0., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#19
myIT2FLS.add_rule([("x1", Long), ("x2", Long), ("x3", Short), ("x4", Short)],
                  [("y1", {"const": 0., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#20
myIT2FLS.add_rule([("x1", Long), ("x2", Medium), ("x3", Short), ("x4", Short)],
                  [("y1", {"const": 0., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#21
myIT2FLS.add_rule([("x1", Long), ("x2", Long), ("x3",Long), ("x4", Long)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#22
myIT2FLS.add_rule([("x1", Long), ("x2", Medium), ("x3", Medium), ("x4", Medium)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#23
myIT2FLS.add_rule([("x1", Long), ("x2", Medium), ("x3", Long), ("x4", Medium)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#24
myIT2FLS.add_rule([("x1", Medium), ("x2", Short), ("x3", Medium), ("x4", Medium)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#25
myIT2FLS.add_rule([("x1", Long), ("x2", Medium), ("x3", Long), ("x4", Long)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#26
myIT2FLS.add_rule([("x1", Medium), ("x2", Short), ("x3", Medium), ("x4", Long)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#27
myIT2FLS.add_rule([("x1", Medium), ("x2", Short), ("x3", Long), ("x4", Medium)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#28
myIT2FLS.add_rule([("x1", Medium), ("x2", Medium), ("x3", Medium), ("x4", Medium)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#29
myIT2FLS.add_rule([("x1", Long), ("x2", Short), ("x3", Medium), ("x4", Medium)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#30
myIT2FLS.add_rule([("x1", Medium), ("x2", Short), ("x3", Long), ("x4", Long)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#31
myIT2FLS.add_rule([("x1", Medium), ("x2", Medium), ("x3", Long), ("x4", Medium)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#32
myIT2FLS.add_rule([("x1", Medium), ("x2", Medium), ("x3", Medium), ("x4", Long)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#33
myIT2FLS.add_rule([("x1", Long), ("x2", Short), ("x3", Long), ("x4", Medium)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#34
myIT2FLS.add_rule([("x1", Long), ("x2", Short), ("x3", Medium), ("x4", Long)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#35
myIT2FLS.add_rule([("x1", Medium), ("x2", Medium), ("x3", Long), ("x4", Long)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#36
myIT2FLS.add_rule([("x1", Long), ("x2", Short), ("x3", Long), ("x4", Long)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#37
myIT2FLS.add_rule([("x1", Long), ("x2", Medium), ("x3", Medium), ("x4", Long)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#38
myIT2FLS.add_rule([("x1", Medium), ("x2", Long), ("x3", Long), ("x4", Long)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#39
myIT2FLS.add_rule([("x1", Long), ("x2", Long), ("x3", Medium), ("x4", Long)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#40
myIT2FLS.add_rule([("x1", Long), ("x2", Long), ("x3", Long), ("x4", Medium)],
                  [("y1", {"const": 2., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#41
myIT2FLS.add_rule([("x1", Long), ("x2", Long), ("x3", Medium), ("x4", Medium)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#42
myIT2FLS.add_rule([("x1", Medium), ("x2", Long), ("x3", Long), ("x4", Medium)],
                  [("y1", {"const": 1., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#43
myIT2FLS.add_rule([("x1", Medium), ("x2", Medium), ("x3", Short), ("x4", Short)],
                  [("y1", {"const": 0., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])
#44
myIT2FLS.add_rule([("x1", Long), ("x2", Short), ("x3", Short), ("x4", Short)],
                  [("y1", {"const": 0., "x1":0.0, "x2":0.0, "x3":0.0, "x4": 0.0})])

iris = datasets.load_iris()
normalized_iris = normalize_dataset(iris.data)

n_features = normalized_iris.shape[1]

print(normalized_iris[149][0], normalized_iris[0][1], normalized_iris[0][2], normalized_iris[0][3])

print(myIT2FLS.evaluate({"x1": normalized_iris[149][0], "x2": normalized_iris[149][1], "x3": normalized_iris[149][2], "x4": normalized_iris[149][3]}))
