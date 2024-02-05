"""
Useful functions to implement State-Of-The-Art (sota) methods.
"""
# to use cleanlab
from sklearn.linear_model import LogisticRegression
from cleanlab.classification import CleanLearning

import numpy as np
import cvxpy as cvx
import time

def clean_labels(data, noisy_labels):
    '''
    Function to clean the labels using cleanlab.

    Parameters
    ----------
    data    : `array`-like of shape (`number_training_size` * `number_features`)
            matrix representing the features X

    noisy_labels  : `array` of shape (`number_training_size` * 1)
            vector representing the noisy (observed) labels to be cleansed

    true_labels   : `array` of shape (`number_training_size` * 1)
            vector representing the true (maybe unknown) labels
            default = None

    Returns
    -------
    cl   : CleanLearning model with all the info regarding the cleaning process

    cleansed_labels: `array` of shape (`number_training_size` * 1)
            vector representing the cleansed labels
    '''

    cl = CleanLearning(clf=LogisticRegression(max_iter=300))

    label_issues = cl.find_label_issues(X=data, labels=noisy_labels)
    number_found_issues = np.sum(np.array(label_issues['is_label_issue']))
    cl.number_found_issues = number_found_issues

    cleansed_labels = np.array(label_issues['predicted_label'])
    print(f'Cleanlab found {number_found_issues} potential labels errors')

    return cl, cleansed_labels


def natarajan_estimation(mdl, x, y):
    '''

    Parameters
    ----------
    mdl:   model of natarajan (like a constructor)
    x  :   data already modified with feature mapping
    y  :   labels

    Returns
    -------

    mdl : with the estimator tau_

    '''

    if mdl.n_classes != 2:
        raise ValueError('This method is implemented only for 2 classes: sorry!')

    n, d = x.shape

    tinv = np.linalg.inv(mdl.T)

    m = np.zeros((n, d))
    for i in range(n):
        m[i, :] = (tinv[0, y[i]] - tinv[1, y[i]]) * x[i, :]
    m = m / 2

    tau_ = np.mean(m, axis=0)
    mdl.tau_ = tau_
    return mdl


def natarajan_fit(mdl, x):
    '''
    Fit the Natarajan model.

    Computes the parameters required for the optimization
    and then calls the `minimax_risk` function to solve the optimization.

    Parameters
    ----------
    mdl    :  model of natarajan (it's like a constructor)

    x : `array`-like of shape (`n_samples`, `n_dimensions`)
        Training instances used in

        - Calculating the expectation estimates
          that constrain the uncertainty set
          for the minimax risk classification
        - Solving the minimax risk optimization problem.

        `n_samples` is the number of training samples and
        `n_dimensions` is the number of features.

    Y : `array`-like of shape (`n_samples`, 1), default = `None`
        Labels corresponding to the training instances
        used only to compute the expectation estimates.


    Returns
    -------
    self :
        Fitted estimator
    '''

    # Limit the number of training samples used in the optimization for large datasets
    # Reduces the training time and use of memory
    n_max = 5000
    not_all_instances = True
    n, d = x.shape

    if not_all_instances and n_max < n:
        n = n_max
        x = x[:n]

    lambda_val = 1 / n
    muu = cvx.Variable((d, 1))
    sum_log_sum_exp = 0

    for i in range(n):
        Mi = np.zeros(shape=(mdl.n_classes, d))
        Mi[0, :] = x[i, :]/2
        Mi[1, :] = - x[i, :]/2
        sum_log_sum_exp = sum_log_sum_exp + cvx.log_sum_exp(Mi @ muu)

    sum_log_sum_exp = sum_log_sum_exp / n

    # log_sum = cvx.log(cvx.exp(x @ muu / 2) + cvx.exp(-x @ muu / 2))

    # x_muu_1 = x @ muu / 2
    # x_muu_2 = -x @ muu / 2
    # exp_term = cvx.exp(cvx.hstack((x_muu_2, x_muu_1)))
    # sum_exp = cvx.sum(exp_term, axis=0)
    # log_sum_exp = cvx.log(sum_exp)
    # sum_log_sum_exp = cvx.sum(log_sum_exp) / n

    if mdl.regularization == "ridge":
        reg_term = lambda_val * cvx.norm(muu, 2)**2

    elif mdl.regularization == "lasso":
        reg_term = lambda_val * cvx.norm(muu, 1)

    objective = -mdl.tau_.T @ muu + sum_log_sum_exp + reg_term
    problem = cvx.Problem(cvx.Minimize(objective))
    problem.solve()

    if problem.status in ["unbounded", "infeasible"]:
        raise ValueError("The problem is ", problem.status)
    else:
        mdl.is_fitted_ = True

    mdl.lambda_ = lambda_val
    mdl.mu_ = muu.value
    mdl.opt = problem.value

    return mdl


def natarajan_predict(mdl, x):

    nte = x.shape[0]
    labels = np.sign(x @ mdl.mu_)
    labels = np.reshape(labels, newshape=(nte,))
    # encode the labels as 0, 1 again
    labels[labels == 1] = 0
    labels[labels == -1] = 1
    mdl.y_pred = labels

    return mdl


def my_natarajan(x_train, y_train, x_test, y_test, T, regularization='ridge'):

    classes_ = np.unique(y_train)
    n_classes = len(classes_)

    mdl = type('', (), {})()
    mdl.T = T
    mdl.regularization = regularization
    mdl.n_classes = n_classes

    start_time = time.time()
    mdl = natarajan_estimation(mdl, x_train, y_train)
    mdl = natarajan_fit(mdl, x_train)

    mdl = natarajan_predict(mdl, x_test)
    end_time = time.time()
    elapsed_time = end_time - start_time

    error = np.average(mdl.y_pred != y_test)
    mdl.time = elapsed_time
    mdl.error = error
    mdl.method = 'natarajan'

    return mdl
