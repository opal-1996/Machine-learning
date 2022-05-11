import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from numpy import linalg as LA
from sklearn.model_selection import train_test_split

def get_a(deg_true):
    """
    Inputs:
    deg_true: (int) degree of the polynomial g
    
    Returns:
    a: (np array of size (deg_true + 1)) coefficients of polynomial g
    """
    return 5 * np.random.randn(deg_true + 1)#For random samples from N(mu, sigma**2),use: sigma * np.random.randn(...) + mu instead.

def get_design_mat(x, deg):
    """
    Inputs:
    x: (np.array of size N)
    deg: (int) max degree used to generate the design matrix
    
    Returns:
    X: (np.array of size N x (deg_true + 1)) design matrix
    """
    X = np.array([x ** i for i in range(deg + 1)]).T
    return X

def draw_sample(deg_true, a, N):
    """
    Inputs:
    deg_true: (int) degree of the polynomial g
    a: (np.array of size deg_true) parameter of g
    N: (int) size of sample to draw
    
    Returns:
    x: (np.array of size N)
    y: (np.array of size N)
    """    
    x = np.sort(np.random.rand(N))
    X = get_design_mat(x, deg_true)
    y = X @ a
    return x, y

def draw_sample_with_noise(deg_true, a, N):  
    """
    Inputs:
    deg_true: (int) degree of the polynomial g
    a: (np.array of size deg_true) parameter of g
    N: (int) size of sample to draw
    
    Returns:
    x: (np.array of size N)
    y: (np.array of size N)
    """  
    x = np.sort(np.random.rand(N))
    X = get_design_mat(x, deg_true)
    y = X @ a + np.random.randn(N)
    return x, y

def least_square_estimator(X, y):
    """
    Inputs:
    X: (np.array of (N x (d+1))) design matrix X
    y: (np.array of N x 1)the corresponding vector 

    Returns:
    b: (np.array of (d+1)x 1) estimates of the coefficients of the function 
    """
    if np.shape(X)[0] <= (np.shape(X)[1] - 1):
        print("The number of samples N is smaller than the degree of polynomial d ")
    else:
        b = np.linalg.inv(X.T @ X) @ X.T @y
        return b.T

def empirical_risk(X, y, b, N):
    """
    Inputs:
    X: design matrix
    y: labels
    b: estimated coefficients
    N: size of sample to draw

    Returns:
    r: empirical risks of the predicted function
    """
    r = (LA.norm(np.dot(X,b) - y)**2) / (2 * N)
    return r

def main():
    a = get_a(5)
    x_train, y_train = draw_sample(5, a, N=10)
    X_train = get_design_mat(x_train, 5)
    b_hat = least_square_estimator(X_train, y_train)

	#plot the training datasets and actual values of the true underlying function
	plt.plot(x_train, X_train @ b_hat, label = "predicted values")
	plt.plot(x_train, y_train, label = "actual values")
	plt.legend()
	plt.title("Plot 1")
	plt.savefig("plot1.png")

	#Adjust d and return its empirical risk
	for i in range(1, 10):
	    a = get_a(i)
	    x_train, y_train = draw_sample(i, a, N=10)
	    X_train = get_design_mat(x_train, i)
	    b_hat = least_square_estimator(X_train, y_train)
	    print(f"The empirical risk with degree {0} is {1}",(i, empirical_risk(X_train, y_train, b_hat, N=10)))

if __name__=="__main__":
	main()













