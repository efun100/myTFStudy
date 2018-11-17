import numpy as np
a=np.array([[0.6, 0.8]])
b=np.array([[[-1.9618274, 2.582354, 1.6820377],[-3.4681718, 1.0698233, 2.11789]]])
c=np.array([[-1.8247149], [2.6854665], [1.418195]])
d=np.dot(a, b)
e=np.dot(d, c)
print(e)
