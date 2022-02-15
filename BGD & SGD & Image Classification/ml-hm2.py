from skeleton_code import *
from mnist_classification_source_code import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d

###Batch gradient descent###
#Question 12: Plot the average square loss on the training set as a function of the number of steps for each step size
# num_step = 1000
# for alpha in (0.5, 0.1, 0.05, 0.01):
# 	X_train, y_train, X_test, y_test = load_data()
# 	theta_hist_train, loss_hist_train = batch_grad_descent(X_train, y_train, alpha = alpha, num_step = num_step, grad_check = False)

# 	plt.plot(np.arange(0,num_step+1), loss_hist_train)
# 	plt.savefig("q12-n1000-{}.jpg".format(alpha))
# 	plt.clf()

"""
Findings:
	When the step size are 0.5, 0.1 respectively, the grad explodes; when the step size are 0.05 and 0.01 respectively,
	the average square loss keeps decreasing in the way we expected, and it performs best when step size is 0.05.
"""

#Question 13: Plot the average square loss on the testing set as a function of the number of steps for each step size
# num_step = 1000
# X_train, y_train, X_test, y_test = load_data()
# for alpha in (0.5, 0.1, 0.05, 0.01):
# 	theta_hist_train, loss_hist_train = batch_grad_descent(X_train, y_train, alpha = alpha, num_step = num_step, grad_check = False)
# 	test_loss = [ ]
# 	for theta in theta_hist_train:
# 		test_loss.append(compute_square_loss(X_test, y_test, theta))
# 	plt.plot(np.arange(0, num_step+1), test_loss)
# 	plt.savefig('q13-{}.jpg'.format(alpha))
# 	plt.clf()

###Ridge regression###
#17. Plot training average square loss and the test average square loss(without regularization) as a function of the training iterations 
# for various values of lambda_reg
# num_step = 10000
# alpha = 0.05#step size

# for lambda_reg in (1e-7, 1e-5, 1e-3, 1e-1, 1, 10, 100):
# 	X_train, y_train, X_test, y_test = load_data()
# 	theta_hist_train, loss_hist_train = regularized_grad_descent(X_train, y_train, alpha = alpha, lambda_reg = lambda_reg, num_step = num_step)
# 	loss_hist_test = [ ]
# 	for theta in theta_hist_train:
# 		loss_hist_test.append(compute_square_loss(X_test, y_test, theta))

# 	plt.plot(np.arange(0,num_step+1), loss_hist_train, label='Training error')
# 	plt.plot(np.arange(0,num_step+1), loss_hist_test, label='Testing error')
# 	plt.legend()

# 	plt.savefig("q17-n10000-{}.jpg".format(lambda_reg))
# 	plt.clf()

#18. Plot the training average square loss and the test average square loss at the end of training as a function of lambda_reg
#19. Plot the minimum of the test average square loss along training as a function of lambda_reg
# num_step = 10000
# alpha = 0.05#step size
# train_loss = [ ]
# test_loss = [ ]
# test_loss_min = [ ]

# for lambda_reg in (1e-7, 1e-5, 1e-3, 1e-1, 1, 10, 100):
# 	X_train, y_train, X_test, y_test = load_data()
# 	theta_train_hist, loss_train_hist = regularized_grad_descent(X_train, y_train, alpha=alpha, lambda_reg=lambda_reg, num_step=num_step)
# 	train_loss.append(loss_train_hist[-1])

# 	loss_hist_test = [ ]
# 	for theta in theta_train_hist:
# 		loss_hist_test.append(compute_square_loss(X_test, y_test, theta))

# 	test_loss.append(loss_hist_test[-1])
# 	test_loss_min.append(min(loss_hist_test))

# plt.plot(np.log([1e-7, 1e-5, 1e-3, 1e-1, 1, 10, 100]), train_loss, label="train_loss")
# plt.plot(np.log([1e-7, 1e-5, 1e-3, 1e-1, 1, 10, 100]), test_loss, label="test_loss")
# plt.plot(np.log([1e-7, 1e-5, 1e-3, 1e-1, 1, 10, 100]), test_loss_min, label="min_test_loss")
# plt.legend()
# plt.savefig("q19.jpg")

###Stochastic gradient descent###
#25. (1)Find the theta that minimize the ridege regression objective for lambda = 1e-05, fixed step size ={0.05, 0.005}
# num_epoch = 1000
# lambda_reg = 10**-2
# for eta in (0.05, 0.005):
# 	X_train, y_train, X_test, y_test = load_data()
# 	theta_hist, loss_hist = stochastic_grad_descent(X_train, y_train, alpha = eta, lambda_reg = lambda_reg, num_epoch = num_epoch, eta0 = False)

# 	print(theta_hist[-1,-1,:])#print the theta minimizes the ridge regression
# 	print(loss_hist[:,10])

	# plt.plot(np.arange(num_epoch), loss_hist[:,10])
	# plt.savefig("q25-fixed-{}.jpg".format(eta))
	# plt.clf()

#25. (2)Find the theta that minimize the ridge regression objective for decreasing learning rate: eta = C/t & eta = C/(t**(1/2)) where C <= 1
# num_epoch = 1000
# lambda_reg = 10**-2
# C = 0.01

# X_train, y_train, X_test, y_test = load_data()
# theta_hist, loss_hist = stochastic_grad_descent_with_decaying_lr(X_train, y_train, lambda_reg=lambda_reg, num_epoch=num_epoch, C = C, eta0=False)

# plt.plot(np.arange(num_epoch), loss_hist[:,50])
# plt.savefig("q25-decaying-lr-sqr-t.jpg")
# plt.clf()

