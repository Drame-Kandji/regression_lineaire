import numpy as np


def sigmoid(x, theta):
    return 1 / (1 + np.exp(-x.dot(theta)))


def cout(x, y, theta):
    m = x.shape[0]
    y_chapeau = sigmoid(x, theta)
    j_theta = (1/m)* np.sum(y*np.log(y_chapeau) + (1- y)* np.log(1-y_chapeau))
    return j_theta

def gradient(x, y,theta, alpha):
    m = x.shape[0]
    y_chapeau = sigmoid(x, theta)
    tmp = y_chapeau - y
    tmp = x.T.dot(tmp)
    return theta - ((alpha/m) * tmp)


#def batch_gradient(x,theta,alpha, nbr_iteration, seuil):
    

