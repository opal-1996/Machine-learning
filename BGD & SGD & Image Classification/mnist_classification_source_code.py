import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def pre_process_mnist_01():
    """
    Load the mnist datasets, selects the classes 0 and 1 
    and normalize the data.
    Args: none
    Outputs: 
        X_train: np.array of size (n_training_samples, n_features)
        X_test: np.array of size (n_test_samples, n_features)
        y_train: np.array of size (n_training_samples)
        y_test: np.array of size (n_test_samples)
    """
    X_mnist, y_mnist = fetch_openml('mnist_784', version=1, 
                                    return_X_y=True, as_frame=False)
    indicator_01 = (y_mnist == '0') + (y_mnist == '1')
    X_mnist_01 = X_mnist[indicator_01]
    y_mnist_01 = y_mnist[indicator_01]
    X_train, X_test, y_train, y_test = train_test_split(X_mnist_01, y_mnist_01,
                                                        test_size=0.33,
                                                        shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 
    X_test = scaler.transform(X_test)

    y_test = 2 * np.array([int(y) for y in y_test]) - 1
    y_train = 2 * np.array([int(y) for y in y_train]) - 1
    return X_train, X_test, y_train, y_test


def sub_sample(N_train, X_train, y_train):
    """
    Subsample the training data to keep only N first elements
    Args: none
    Outputs: 
        X_train: np.array of size (n_training_samples, n_features)
        X_test: np.array of size (n_test_samples, n_features)
        y_train: np.array of size (n_training_samples)
        y_test: np.array of size (n_test_samples)
    """
    assert N_train <= X_train.shape[0]
    return X_train[:N_train, :], y_train[:N_train]

def classification_error(clf, X, y):
    """
    Return the classification error: for a given sample, the classification error is 1 if no example was labeled correctly and 0 if 
    all examples were perfectly labeled.

    Args:
        clf - SGDClassifier
        X - design matrix
        y - target vector
    Returns:
        error - classification error
    """
    error = 1 - np.sum(clf.predict(X) == y) / y.shape[0]
    return error


# plt.imsave("Image.jpg", X_train[2,:].reshape((28,28)))

##29. Plot the means and stds of testing error under different regularization parameter alpha
# means = [ ]
# stds = [ ]

# for alpha in np.linspace(1e-4, 1e-1, 10, endpoint=False):

#     clf = SGDClassifier(loss='log', max_iter=1000, 
#                     tol=1e-3,
#                     penalty='l1', alpha=alpha, 
#                     learning_rate='invscaling', 
#                     power_t=0.5,                
#                     eta0=0.01,
#                     verbose=1)

#     temp = np.zeros(10)
#     for i in range(10):
#         X_train, X_test, y_train, y_test = pre_process_mnist_01()
#         N_train = 100
#         X_train_sub, y_train_sub = sub_sample(N_train, X_train, y_train)
#         clf.fit(X_train_sub, y_train_sub)
#         error = classification_error(clf, X_test, y_test)
#         temp[i] = error

#     means.append(np.mean(temp))
#     stds.append(np.std(temp))

# print(means)
# print(stds)
# means = [0.005555555555555591, 0.004735547355473579, 0.004038540385403855, 0.0037105371053710477, 0.0035055350553505173, 0.0034850348503484784, 0.0031775317753177323, 0.002726527265272616, 0.002316523165231621, 0.002296022960229571]
# stds = [0.0005758332064447703, 0.000561046829987894, 0.0006355063550635904, 0.00042460670719987056, 0.0005381059757444493, 0.0005186187224548624, 0.0005443180831222834, 0.0004941357438783841, 0.0002906405674202057, 0.00023907141840283813]

# plt.plot(np.log(np.linspace(1e-4, 1e-1, 10, endpoint=False)), means)
# plt.errorbar(np.log(np.linspace(1e-4, 1e-1, 10, endpoint=False)), means, yerr = np.array(stds), fmt = 'o')
# plt.savefig('q29.jpg')
# plt.clf()

#Question 32. Plot the value of the fitted theta for different regularization parameter alpha
# for alpha in np.linspace(1e-4, 1e-1, 10, endpoint=False):
#     clf = SGDClassifier(loss='log', max_iter=1000, 
#                     tol=1e-3,
#                     penalty='l1', alpha=alpha, 
#                     learning_rate='invscaling', 
#                     power_t=0.5,                
#                     eta0=0.01,
#                     verbose=1)

#     X_train, X_test, y_train, y_test = pre_process_mnist_01()
#     N_train = 100
#     X_train_sub, y_train_sub = sub_sample(N_train, X_train, y_train)
#     clf.fit(X_train_sub, y_train_sub)
#     fitted_theta = clf.coef_

#     scale = np.abs(fitted_theta).max()
#     fig = plt.figure(figsize=(10,8))
#     im = plt.imshow(fitted_theta.reshape((28,28)), cmap = plt.cm.RdBu, vmax = scale, vmin = -scale)
#     fig.colorbar(im)
#     plt.savefig("q32-{}.jpg".format(alpha))
#     plt.clf()
    































