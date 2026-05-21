import milia_lib
import numpy as np

m = np.array([
  [1,2,3,4],
  [4,5,1,2]
])

theta = [0,1,2,0]

sigmoid = milia_lib.sigmoid(m, theta)
z = m.dot(theta)
print(z)
print(sigmoid)

theta = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

cost1 = milia_lib.cout(x,y,theta)
