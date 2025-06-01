import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm
import scipy.special as sc
from class_CP_QQ import calc_matrix_M, sum_hypergeo, Multi_Boucle

m = 3 # number of machines
n = 90 # numbert of points per machine
M = calc_matrix_M(m, n, .0, mid=False)
print(M)

a = .2 # coverage

mm = list(np.ravel(M))
v = min(i for i in mm if i > (1-a))
k = int(np.where(M == v)[0])
l = int(np.where(M == v)[1])
print(k, ', ', l, ', ', M[k, l])