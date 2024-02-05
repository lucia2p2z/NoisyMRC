"""
.. general_utilities:
Useful functions to do learning with noisy labels
"""

import os
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import Bunch
from sklearn.utils import resample

from MRCpy import MRC


def corruption(Y, T, seed=1):
    """
    Corrupts the vector Y according to matrix T.

    Parameters
    ----------
    Y
    T
    seed

    Returns
    -------

    """
    np.random.seed(seed)
    coin = np.random.uniform(0, 1, size=Y.shape[0], )
    classes_ = np.unique(Y)
    n = Y.shape[0]

    Ytilda = [
        classes_[1] if (Y[i] == classes_[0] and coin[i] >= T[0, 0]) else
        classes_[0] if (Y[i] == classes_[1] and coin[i] >= T[1, 1]) else
        Y[i]
        for i in range(n)
    ]
    return np.array(Ytilda)


# same as def normalizeLabels(origY) that can be found in load.py
def normalizeLabels(Y):
    """
    Normalize the labels of the instances in the range 0,...r-1 for r classes
    """

    origY = Y
    classes_ = np.unique(origY)
    Y = np.zeros(origY.shape[0], dtype=int)

    # Map the values of Y from 0 to n_classes-1
    for i, y in enumerate(classes_):
        Y[origY == y] = i

    return np.array(Y)


def balance_data(X, Y, seed=1):
    """
    Balance the data according to the labels in Y.

    Parameters
    ----------
    X
    Y
    seed

    Returns
    -------

    """

    classes_ = np.unique(Y)
    class_1_indices = np.where(Y == classes_[0])[0]
    class_2_indices = np.where(Y == classes_[1])[0]

    if len(class_1_indices) > len(class_2_indices):
        # Downsample 1st class
        downsampled_indices = resample(class_1_indices, replace=False,
                                       n_samples=len(class_2_indices), random_state=seed)
        indices_to_keep = np.concatenate((downsampled_indices, class_2_indices))
    elif len(class_2_indices) > len(class_1_indices):
        # Downsample 2nd class
        downsampled_indices = resample(class_2_indices, replace=False,
                                       n_samples=len(class_1_indices), random_state=seed)
        indices_to_keep = np.concatenate((class_1_indices, downsampled_indices))
    else:
        indices_to_keep = np.concatenate((class_1_indices, class_2_indices))

    Xbal = X[indices_to_keep]
    Ybal = Y[indices_to_keep]

    shuffle = np.random.permutation(len(Xbal))
    Xbal = Xbal[shuffle]
    Ybal = Ybal[shuffle]

    return Xbal, Ybal


def mean_std(mymatrix, axis_=0):
    """
    Compute the mean and standard deviation of the parameter mymatrix.

    The operations are done along the specified axis.
    """
    mean_ = np.average(mymatrix, axis=axis_)
    std_ = np.std(mymatrix, axis=axis_)

    return mean_, std_


def load_mortality(categorical, with_info=False):
    """Load and return the mortality incomes prediction dataset (classification).


    Without categorical variables
    =================   ==============
    Classes                          2
    Samples per class     [83748,7901]
    Samples total                91649
    Dimensionality                 126
    Features             int, positive
    =================   ==============

    With categorical variables
    =================   ==============
    Classes                          2
    Samples per class     [83748,7901]
    Samples total                91649
    Dimensionality                 202
    Features             int, positive
    =================   ==============

    Parameters
    ----------
    categorical : boolean, default=False.
        If True, read the dataset containing also the categorical variables.

    with_info : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.


    Returns
    -------
    bunch : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification targets,
        'DESCR', the full description of the dataset,
        and 'filename', the physical location of the dataset.

    (data, target) : tuple if ``with_info`` is True

    """

    if categorical:
        categorical_type = 'alsocat'
    else:
        categorical_type = 'nocat'

    tbl = pd.read_csv(f'{os.path.dirname(__file__)}/data_mortality/mortality_{categorical_type}.csv')

    print(f'Loaded mortality_{categorical_type}')

    tbl = tbl.drop(tbl.columns[0], axis=1)
    tbl = tbl.values

    # take the covariates and standardize them
    data = tbl[:, 1:]
    data = StandardScaler().fit_transform(data)

    # take the labels and encode them into {0,1}
    target = tbl[:, 0]
    target = normalizeLabels(target)

    if not with_info:
        return data, target
    else:
        raise ValueError('Not implemented yet ... ')
        #return data, normalizeLabels(target)


def my_mrc(phi, loss, det, lambda0, random_seed, method, x_train, y_train, x_test, y_test, T=None, projection=False):

    mrc = MRC(fit_intercept=False,
              deterministic=det,
              loss=loss,
              phi=phi,
              # sigma='scale2',
              solver='cvx',
              s=lambda0,
              one_hot=True,
              random_state=random_seed)
    mrc.cvx_solvers = ['MOSEK', 'SCS', 'ECOS']

    # Fit the classifier on the training data
    start_time = time.time()

    if method == 'classic':
        mrc.fit(x_train, y_train)
        mrc.T = None
    elif method == 'backward':
        if T is None:
            raise ValueError('You should provide a noise matrix T ... ')
        mrc.fit_noise(x_train, y_train, T, projection=projection)
        mrc.T = T
    elif method == 'forward':
        raise ValueError('Not implemented yet ... ')
    else:
        raise ValueError('Unexpected method: ', method)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Predict and Calculate the error made by MRC classificator
    y_pred = mrc.predict(x_test)
    error = np.average(y_pred != y_test)

    # Save into mrc all the important quantities
    mrc.y_pred = y_pred
    mrc.time = elapsed_time
    mrc.error = error
    mrc.method = method

    return mrc


def my_lr(x_train, y_train, x_test, y_test, max_iter=100):

    start_time = time.time()
    lr = LogisticRegression(max_iter=max_iter)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    lr.y_pred = y_pred

    lr.accuracy = lr.score(x_test, y_test)
    lr.error = 1 - lr.accuracy
    end_time = time.time()
    elapsed_time = end_time - start_time
    lr.time = elapsed_time

    return lr


def unbiased_loss(mdl, yeval):

    tinv = np.linalg.inv(mdl.T)
    n = yeval.shape[0]
    loss = np.zeros(shape=(n, 1))

    for i in range(n):
        l = tinv[0, yeval[i]] * (mdl.y_pred[i] != 0) + tinv[1, yeval[i]] * (mdl.y_pred[i] != 1)
        loss[i] = l

    return np.mean(loss)


def save_dataframe(df, df_name='a', save_dir='./', format_to_save='gzip'):
    print('Saving {} dataframe...'.format(df_name))
    if format_to_save == 'gzip':
        # Compressed pickle with gzip: fastest, not most compressed
        with open('{}/{}.gz'.format(save_dir, df_name), 'wb') as f:
            df.to_pickle(f, compression='gzip')

    elif format_to_save == 'csv':
        # Raw csv
        with open('{}/{}.csv'.format(save_dir, df_name), 'wb') as f:
            df.to_csv(f, index=False)
    else:
        raise ValueError('Format to save = {} not implemented yet!'.format(format_to_save))

    print('\t... saved!')


def load_dataframe(load_dir, df_name='a', saved_format='gzip'):
    print('Loading {} dataframe...'.format(df_name))
    if saved_format == 'gzip':
        # Compressed pickle with gzip: fastest, not most compressed
        with open('{}/{}.gz'.format(load_dir, df_name), 'rb') as f:
            df = pd.read_pickle(f, compression='gzip')

    elif saved_format == 'csv':
        # Raw csv
        with open('{}/{}.csv'.format(load_dir, df_name), 'rb') as f:
            df = pd.read_csv(f)
    else:
        raise ValueError('Format to load = {} not implemented yet!'.format(saved_format))

    print('\t... loaded!')
    return df
