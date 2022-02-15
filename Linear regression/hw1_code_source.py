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

# a = get_a(5)
# x_train, y_train = draw_sample(5, a, N=10)
# X_train = get_design_mat(x_train, 5)
# b_hat = least_square_estimator(X_train, y_train)


# #plot the training datasets and actual values of the true underlying function
# plt.plot(x_train, X_train @ b_hat, label = "predicted values")
# plt.plot(x_train, y_train, label = "actual values")
# plt.legend()
# plt.title("Plot 1")
# plt.savefig("plot1.png")


# #Adjust d and return its empirical risk
# for i in range(1, 10):
#     a = get_a(i)
#     x_train, y_train = draw_sample(i, a, N=10)
#     X_train = get_design_mat(x_train, i)
#     b_hat = least_square_estimator(X_train, y_train)
#     print(f"The empirical risk with degree {0} is {1}",(i, empirical_risk(X_train, y_train, b_hat, N=10)))

#Hands on 
#--------------------------------------------------------------------------------------------------#
##In presence of noise
#11. Plot e_t and e_g as a function of N
# d = 2
# a = get_a(d)
# e_t = [ ]#training error
# e_g = [ ]#generalization error
# N_test = 1000

# x_test, y_test = draw_sample(d,a,N_test)
# X_test = get_design_mat(x_test, d)

# for n in range(d+1,1000):
#     x_train, y_train = draw_sample_with_noise(d, a, n)
#     X_train = get_design_mat(x_train, d)
#     b_hat = least_square_estimator(X_train, y_train)
#     e_t.append(np.log(empirical_risk(X_train, y_train, b_hat,n)))
#     e_g.append(np.log(empirical_risk(X_test,y_test,b_hat,N_test)))

# e_t = np.array(e_t)
# e_g = np.array(e_g)
# plt.plot(np.arange(d+1, 1000), e_t, label="d=2 & log(e_t)")
# plt.plot(np.arange(d+1, 1000), e_g, label="d=2 & log(e_g)")
# plt.legend()
# plt.title("Plot 2-1(d=2)")
# plt.savefig("plot2-1.png")

# d = 5
# a = get_a(d)
# e_t = [ ]#training error
# e_g = [ ]#generalization error
# N_test = 1000

# x_test, y_test = draw_sample(d,a,N_test)
# X_test = get_design_mat(x_test, d)

# for n in range(d+1,1000):
#     x_train, y_train = draw_sample_with_noise(d, a, n)
#     X_train = get_design_mat(x_train, d)
#     b_hat = least_square_estimator(X_train, y_train)
#     e_t.append(np.log(empirical_risk(X_train, y_train, b_hat,n)))
#     e_g.append(np.log(empirical_risk(X_test,y_test,b_hat,N_test)))

# e_t = np.array(e_t)
# e_g = np.array(e_g)
# plt.plot(np.arange(d+1, 1000), e_t, label="d=5 & log(e_t)")
# plt.plot(np.arange(d+1, 1000), e_g, label="d=5 & log(e_g)")
# plt.legend()
# plt.title("Plot 2-2(d=5)")
# plt.savefig("plot2-2.png")

# d = 10
# a = get_a(d)
# e_t = [ ]#training error
# e_g = [ ]#generalization error
# N_test = 1000

# x_test, y_test = draw_sample(d,a,N_test)
# X_test = get_design_mat(x_test, d)

# for n in range(d+1,1000):
#     x_train, y_train = draw_sample_with_noise(d, a, n)
#     X_train = get_design_mat(x_train, d)
#     b_hat = least_square_estimator(X_train, y_train)
#     e_t.append(np.log(empirical_risk(X_train, y_train, b_hat,n)))
#     e_g.append(np.log(empirical_risk(X_test,y_test,b_hat,N_test)))

# e_t = np.array(e_t)
# e_g = np.array(e_g)
# plt.plot(np.arange(d+1, 1000), e_t, label="d=10 & log(e_t)")
# plt.plot(np.arange(d+1, 1000), e_g, label="d=10 & log(e_g)")
# plt.legend()
# plt.title("Plot 2-3(d=10)")
# plt.savefig("plot2-3.png")

#12. The estimation error  = R(f_n) - R(f_d) where f_n is the estimated function in hypothesis class d with noisy samples
#and f_d is the optimal function in hypothesis class d; because Bayes prediction function f_* changes with degree of polynomial
#, f_d is the same as f_*, which means the approximation error is always 0 under different degrees.


#--------------------------------------------------------------------------------------------------#
##Application to Ozone data

#load data 
# data = np.loadtxt("ozone_wind.data")
# x = data[:,0]# np array of size 1 x 111
# y = data[:,1]# np array of size 1 x 111

# #d = 2,5,10 & test_size = 0.33,0.44,0.55
# N_size = [ ]

# d_1 = 2
# e_t_2 = [ ]
# e_g_2 = [ ]
# for test_size in (0.55,0.44,0.33):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
#     X_train = get_design_mat(x_train, d_1)
#     X_test = get_design_mat(x_test, d_1)
#     b_hat = least_square_estimator(X_train, y_train)
#     e_t_2.append(np.log(empirical_risk(X_train, y_train, b_hat, x_train.shape[0]))) 
#     e_g_2.append(np.log(empirical_risk(X_test,y_test,b_hat,x_test.shape[0])))
#     N_size.append(x_train.shape[0])

# d_2= 5
# e_t_5 = [ ]
# e_g_5 = [ ]
# for test_size in (0.55,0.44,0.33):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
#     X_train = get_design_mat(x_train, d_2)
#     X_test = get_design_mat(x_test, d_2)
#     b_hat = least_square_estimator(X_train, y_train)
#     e_t_5.append(np.log(empirical_risk(X_train, y_train, b_hat, x_train.shape[0]))) 
#     e_g_5.append(np.log(empirical_risk(X_test,y_test,b_hat,x_test.shape[0])))
# # print(e_t_5)

# d_3= 10
# e_t_10 = [ ]
# e_g_10 = [ ]
# for test_size in (0.55,0.44,0.33):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
#     X_train = get_design_mat(x_train, d_3)
#     X_test = get_design_mat(x_test, d_3)
#     b_hat = least_square_estimator(X_train, y_train)
#     e_t_10.append(np.log(empirical_risk(X_train, y_train, b_hat, x_train.shape[0]))) 
#     e_g_10.append(np.log(empirical_risk(X_test,y_test,b_hat,x_test.shape[0])))
# # print(e_t_10)

# plt.plot(N_size, e_t_2, label = "d = 2 & log(e_t)")
# plt.plot(N_size, e_t_5, label = "d = 5 & log(e_t)")
# plt.plot(N_size, e_t_10, label = "d = 10 & log(e_t)")
# plt.legend()
# plt.title("Ozone Wind")
# plt.xlabel("N")

# plt.plot(N_size, e_g_2, label = "d = 2 & log(e_g)")
# plt.plot(N_size, e_g_5, label = "d = 5 & log(e_g)")
# plt.plot(N_size, e_g_10, label = "d = 10 & log(e_g)")
# plt.legend()
# plt.title("Ozone Wind")
# plt.xlabel("N")

# plt.savefig("ozone_wind")






































