import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from general_utilities import *
from sota_utilities import *

# to transform the data
from MRCpy.phi import \
    BasePhi, \
    RandomFourierPhi, \
    RandomReLUPhi, \
    ThresholdPhi


def simulation_perfeval_mrc(X, Y, nvector, nrep, det, lambda0, phi='fourier', loss='0-1', rho1=0.2, rho2=0.2):
    """
    Simulation for different training size for MRCs models, evaluated on noisy test data.

    Parameters
    ----------
    X : `array`-like of shape (`n_samples` * `n_features`)
        matrix representing the data (features)

    Y : `array`-like of shape (`n_samples` * 1)
        matrix representing the labels

    nvector : `array`-like of shape (`number_training_size` * 1)
            vector representing different training size

    nrep : `int`
        number of repetitions for the simulations

    det : `bool`
        True if we want to use the deterministic rule to predict
        False if we want to use the probabilistic rule to predict

    lambda0 : `float`
        hyperparameter of MRC that controls the size of the set U:
        lambda = lambda0 * std/ sqrt(ntrain)
        Typically: lambda0 = 0.3, 0.7, 1

    phi : `str` or `BasePhi` instance, default = 'fourier'
        Type of feature mapping function to use for mapping the input data.
        The currently available feature mapping methods are
        'fourier', 'relu', 'threshold' and 'linear'.

    loss : `str` {'0-1', 'log'}, default = '0-1'
        Type of loss function to use for the risk minimization.

    rho1 : `float`, default = 0.2
        Noise rate on class 1
        :math:` \\mathbb{P}(ytilda = 2 | ytrue = 1)`

    rho2 : `float`, default = 0.2
        Noise rate on class 2
        :math:` \\mathbb{P}(ytilda = 1 | ytrue = 2)`

    Returns
    -------
    summary_data:   Pandas Dataframe
                    summary of all the simulation's setting
                    (loss, feat. mapp, rhos)

    summary_results: Pandas Dataframe
                     summary of the main results
                     (errors of the methods, times and eventually bounds)

    results_models:  Pandas Dataframe
                     containing the detailed information regarding each modela of each repetition and training size

    """

    n_samples, n_features = X.shape

    results_models = pd.DataFrame(
        {'rep': [],
         'ntrain': [],
         'mrc_nocorr': [],
         'mrc_back': []
         }
    )
    T = np.array([[1 - rho1, rho2], [rho1, 1 - rho2]])

    # corrupt the data with fixed seed
    Ytilda = corruption(Y, T)

    my_shape = (nrep, len(nvector))
    tot_time1, tot_time2 = [np.zeros(my_shape) for _ in range(2)]
    tot_BLE1, tot_BLE2, tot_ULE2 = [np.zeros(my_shape) for _ in range(3)]
    tot_bd1, tot_bd2 = [np.zeros(my_shape) for _ in range(2)]

    for rep in range(nrep):

        seed = rep
        print("Rep:", rep)

        # split data into training and test sets:
        X_train, X_test, y_tilda_train, y_tilda_test, \
            train_indx, test_indx = train_test_split(X, Ytilda, range(n_samples), test_size=0.2, random_state=seed,
                                                     stratify=Ytilda)

        # retrieve the clean test and training set:
        y_test = Y[test_indx]

        for i, ntrain in enumerate(nvector):

            X_tr = X_train[0:ntrain, :]
            y_ti = y_tilda_train[0:ntrain]

            mrc1 = my_mrc(phi, loss, det, lambda0, seed, 'classic', X_tr, y_ti, X_test, y_tilda_test)  # w/out correction
            mrc2 = my_mrc(phi, loss, det, lambda0, seed, 'backward', X_tr, y_ti, X_test, y_tilda_test, T, projection=True)  # backward
            mrc2.ULE = unbiased_loss(mrc2, y_tilda_test)

            tot_time1[rep, i] = mrc1.time
            tot_time2[rep, i] = mrc2.time

            tot_bd1[rep, i] = mrc1.upper_
            tot_bd2[rep, i] = mrc2.upper_

            tot_BLE1[rep, i] = mrc1.error  # representing the Biased Loss Estimator w/out correction
            tot_BLE2[rep, i] = mrc2.error  # representing the Biased Loss Estimator w/out correction

            tot_ULE2[rep, i] = mrc2.ULE    # representing the Unbiased Loss Estimator corrected using T^-1

            results_models = results_models._append(
                {'rep': rep,
                 'ntrain': ntrain,
                 'mrc_nocorr': {'model': mrc1,
                                'noise_matrix_used_in_learning': None, 'noise_matrix_used_in_corruption': T,
                                'predicted_labels': [mrc1.y_pred], 'true_labels': [y_test],
                                'noisy_labels': [y_tilda_test]},
                 'mrc_back': {'model': mrc2,
                              'noise_matrix_used_in_learning': T, 'noise_matrix_used_in_corruption': T,
                              'predicted_labels': [mrc2.y_pred], 'true_labels': [y_test],
                              'noisy_labels': [y_tilda_test]}
                 }, ignore_index=True
            )

    summary_data = pd.DataFrame(
        data={
            'n_samples': '%d' % n_samples,
            'n_attributes': '%d' % n_features,
            'phi':  phi,
            'loss': loss,
            'deterministic_rule': det,
            'lambda0': lambda0,
            'rho1': '%f' % rho1,
            'rho2': '%f' % rho2,
            'noise_matrix': [T],
            'ntrain_vector': [nvector],
        }
    )
    summary_results = pd.DataFrame(
        {'mrc_nocorr':  {'biased_loss': tot_BLE1, 'unbiased_loss': [], 'bounds': tot_bd1, 'times': tot_time1},
         'mrc_back':  {'biased_loss': tot_BLE2, 'unbiased_loss': tot_ULE2, 'bounds': tot_bd2, 'times': tot_time2},
         }
    )

    return summary_data, summary_results, results_models


def simulation_perfeval_lr_nata(X, Y, nvector, nrep, phi='fourier', rho1=0.2, rho2=0.2, **phi_kwargs):
    """
    Simulation for different training size for Logistic regression models, evaluated on noisy test data.

    Parameters
    ----------
    X : `array`-like of shape (`n_samples` * `n_features`)
        matrix representing the data (features)

    Y : `array`-like of shape (`n_samples` * 1)
        matrix representing the labels

    nvector : `array`-like of shape (`number_training_size` * 1)
            vector representing different training size

    nrep : `int`
        number of repetitions for the simulations

    phi : `str` or `BasePhi` instance, default = 'fourier'
        Type of feature mapping function to use for mapping the input data.
        The currently available feature mapping methods are
        'fourier', 'relu', 'threshold' and 'linear'.

    rho1 : `float`, default = 0.2
        Noise rate on class 1
        :math:` \\mathbb{P}(ytilda = 2 | ytrue = 1)`

    rho2 : `float`, default = 0.2
        Noise rate on class 2
        :math:` \\mathbb{P}(ytilda = 1 | ytrue = 2)`


    Returns
    -------
    results : all the important variables
    """

    n_samples, n_features = X.shape
    classes_ = np.unique(Y)
    n_classes = len(classes_)

    # TRASFORMA I DATI CON FEAT. MAPPING
    # Feature mappings
    fit_intercept = False
    random_state = False
    if phi == 'fourier':
        phi = RandomFourierPhi(n_classes=n_classes,
                               fit_intercept=fit_intercept,
                               random_state=random_state,
                               **phi_kwargs)
    elif phi == 'linear':
        phi = BasePhi(n_classes=n_classes,
                      fit_intercept=fit_intercept,
                      **phi_kwargs)
    elif phi == 'threshold':
        phi = ThresholdPhi(n_classes=n_classes,
                           fit_intercept=fit_intercept,
                           **phi_kwargs)
    elif phi == 'relu':
        phi = RandomReLUPhi(n_classes=n_classes,
                            fit_intercept=fit_intercept,
                            random_state=random_state,
                            **phi_kwargs)
    elif not isinstance(phi, BasePhi):
        raise ValueError('Unexpected feature mapping type ... ')

    # Fit the feature mappings
    phi.fit(X, Y)
    X_feat = phi.transform(X)

    results_models = pd.DataFrame(
        {'rep': [],
         'ntrain': [],
         'lr_nocorr': [],
         'natarajan': [],
         }
    )

    T = np.array([[1 - rho1, rho2], [rho1, 1 - rho2]])
    Ytilda = corruption(Y, T)

    my_shape = (nrep, len(nvector))

    tot_time1, tot_BLE1 = [np.zeros(my_shape) for _ in range(2)]
    tot_time2, tot_BLE2, tot_ULE2 = [np.zeros(my_shape) for _ in range(3)]

    for rep in range(nrep):

        seed = rep
        print("Rep:", rep)

        # split data into training and test sets:
        X_train, X_test, y_tilda_train, y_tilda_test, \
            train_indx, test_indx = train_test_split(X_feat, Ytilda, range(n_samples), test_size=0.2, random_state=seed,
                                                     stratify=Ytilda)

        # retrieve the clean test and training set:
        y_test = Y[test_indx]

        for i, ntrain in enumerate(nvector):

            X_tr = X_train[0:ntrain, :]
            y_ti = y_tilda_train[0:ntrain]

            # LR on noisy labels without correction
            lr = my_lr(X_tr, y_ti, X_test, y_tilda_test, max_iter=300)

            # NATARAJAN METHOD: modified logistic regression
            nat = my_natarajan(X_tr, y_ti, X_test, y_tilda_test, T, regularization='ridge')
            nat.ULE = unbiased_loss(nat, y_tilda_test)

            tot_time1[rep, i] = lr.time
            tot_time2[rep, i] = nat.time

            tot_BLE1[rep, i] = lr.error   # representing the Biased Loss Estimator w/out correction
            tot_BLE2[rep, i] = nat.error  # representing the Biased Loss Estimator w/out correction

            tot_ULE2[rep, i] = nat.ULE    # representing the Unbiased Loss Estimator corrected using T^-1

            results_models = results_models._append(
                {'rep': rep,
                 'ntrain': ntrain,
                 'lr_nocorr': {'model': lr,
                               'noise_matrix_used_in_learning': None, 'noise_matrix_used_in_corruption': [T],
                               'predicted_labels': [lr.y_pred], 'true_labels': [y_test],
                               'noisy_labels': [y_tilda_test]},
                 'natarajan': {'model': nat,
                               'noise_matrix_used_in_learning': [T], 'noise_matrix_used_in_corruption': [T],
                               'predicted_labels': [nat.y_pred], 'true_labels': [y_test],
                               'noisy_labels': [y_tilda_test]}
                 }, ignore_index=True
            )

    summary_data = pd.DataFrame(
        data={
            'n_samples': '%d' % n_samples,
            'n_attributes': '%d' % n_features,
            'phi': phi,
            'rho1': '%f' % rho1,
            'rho2': '%f' % rho2,
            'noise_matrix': [T],
            'ntrain_vector': [nvector],
        }
    )

    summary_results = pd.DataFrame(
        {'lr_nocorr': {'biased_loss': tot_BLE1, 'unbiased_loss': [], 'times': tot_time1},
         'natarajan': {'biased_loss': tot_BLE2, 'unbiased_loss': tot_ULE2, 'times': tot_time2},
         }
    )

    return summary_data, summary_results, results_models


def simulation_perfeval_cleanlab(X, Y, nvector, nrep, phi='fourier', rho1=0.2, rho2=0.2, **phi_kwargs):
    """
    Simulation for different training size and using CleanLearning proivided by cleanlab.

    Here we train on noisy labels a method provided by library cleanlab (https://docs.cleanlab.ai/stable/index.html)
    then we evaluate the performance of this classifier on cleansed test labels (metric: ble).
    We use it as an estimate of the true error cl.error
    computed on truly clean test-data that in real-life are not provided.

    Parameters
    ----------
    X : `array`-like of shape (`n_samples` * `n_features`)
        matrix representing the data (features)

    Y : `array`-like of shape (`n_samples` * 1)
        matrix representing the labels

    nvector : `array`-like of shape (`number_training_size` * 1)
            vector representing different training size

    nrep : `int`
        number of repetitions for the simulations

    phi : `str` or `BasePhi` instance, default = 'fourier'
        Type of feature mapping function to use for mapping the input data.
        The currently available feature mapping methods are
        'fourier', 'relu', 'threshold' and 'linear'.

    rho1 : `float`, default = 0.2
        Noise rate on class 1
        :math:` \\mathbb{P}(ytilde = 2 | ytrue = 1)`

    rho2 : `float`, default = 0.2
        Noise rate on class 2
        :math:` \\mathbb{P}(ytilde = 1 | ytrue = 2)`


    Returns
    -------
    summary_data:   Pandas Dataframe
                    summary of all the simulation's setting
                    (loss, feat. mapp, rhos)

    summary_results: Pandas Dataframe
                     summary of the main results
                     (errors of the methods, times and eventually bounds)

    results_models:  Pandas Dataframe
                     containing the detailed information regarding each modela of each repetition and training size

    """

    n_samples, n_features = X.shape
    classes_ = np.unique(Y)
    n_classes = len(classes_)

    # TRASFORMA I DATI CON FEAT. MAPPING
    # Feature mappings
    fit_intercept = False
    random_state = False
    if phi == 'fourier':
        phi = RandomFourierPhi(n_classes=n_classes,
                               fit_intercept=fit_intercept,
                               random_state=random_state,
                               **phi_kwargs)
    elif phi == 'linear':
        phi = BasePhi(n_classes=n_classes,
                      fit_intercept=fit_intercept,
                      **phi_kwargs)
    elif phi == 'threshold':
        phi = ThresholdPhi(n_classes=n_classes,
                           fit_intercept=fit_intercept,
                           **phi_kwargs)
    elif phi == 'relu':
        phi = RandomReLUPhi(n_classes=n_classes,
                            fit_intercept=fit_intercept,
                            random_state=random_state,
                            **phi_kwargs)
    elif not isinstance(phi, BasePhi):
        raise ValueError('Unexpected feature mapping type ... ')

    # Fit the feature mappings
    phi.fit(X, Y)
    X_feat = phi.transform(X)

    results_models = pd.DataFrame(
        {'rep': [],
         'ntrain': [],
         'cl': [],
         }
    )

    T = np.array([[1 - rho1, rho2], [rho1, 1 - rho2]])
    Ytilda = corruption(Y, T)

    my_shape = (nrep, len(nvector))

    # cleanlab method
    tot_time, tot_err, tot_ble = [np.zeros(my_shape) for _ in range(3)]

    for rep in range(nrep):

        seed = rep
        print("Rep:", rep)

        X_train, X_test, y_tilda_train, y_tilda_test, \
            train_indx, test_indx = train_test_split(X_feat, Ytilda, range(n_samples), test_size=0.2,
                                                     random_state=seed, stratify=Ytilda)

        # retrieve the clean and cleansed test sets:
        y_test = Y[test_indx]
        cl_rep, y_cleansed_test = clean_labels(X_test, y_tilda_test)

        for i, ntrain in enumerate(nvector):

            print("training size: ", ntrain)

            X_tr = X_train[0:ntrain, :]
            y_tr = y_tilda_train[0:ntrain]

            # METHOD OF CLEANLAB
            start_time = time.time()
            cl = CleanLearning(clf=LogisticRegression(max_iter=400))  # any sklearn-compatible classifier
            cl.fit(X_tr, y_tr)
            end_time = time.time()
            elapsed_time = end_time - start_time
            y_pred = cl.predict(X_test)
            cl.time = elapsed_time
            cl.y_pred = y_pred
            cl.error = np.average(y_pred != y_test)
            cl.ble = np.average(y_pred != y_cleansed_test)

            tot_time[rep, i] = cl.time
            tot_err[rep, i] = cl.error
            tot_ble[rep, i] = cl.ble

            results_models = results_models._append(
                {'rep': rep,
                 'ntrain': ntrain,
                 'cl': {'model': cl, 'noise_matrix_used_in_learning': None,
                        'noise_matrix_used_in_corruption': T, 'predicted_labels': [cl.y_pred],
                        'true_labels': [y_test], 'noisy_labels': [y_tilda_test], 'cleansed_labels': [y_cleansed_test]}
                 }, ignore_index=True
            )

    summary_data = pd.DataFrame(
        data={
            'n_samples': '%d' % n_samples,
            'n_attributes': '%d' % n_features,
            'phi': phi,
            'rho1': '%f' % rho1,
            'rho2': '%f' % rho2,
            'noise_matrix': [T],
            'ntrain_vector': [nvector],
        }
    )
    summary_results = pd.DataFrame(
        data={
            'cl': {'biased_loss': tot_ble, 'unbiased_loss': [], 'errors': tot_err, 'times': tot_time}
        }
    )

    return summary_data, summary_results, results_models


def simulation_perfeval_lrcleansed(X, Y, nvector, nrep, phi='fourier', rho1=0.2, rho2=0.2, **phi_kwargs):
    """
    Simulation for different training size and using cleansed labels proivided by cleanlab.

    Here we cleanse the labels using library cleanlab (https://docs.cleanlab.ai/stable/index.html) then
    we train on those a classic logistic regression and we evaluate the performance of this lr_cleansed
    on cleansed test labels (metric: ble). We use it as an estiamte of the true error lr_cleansed.error
    computed on truly clean test-data that in real-life are not provided.

    Parameters
    ----------
    X : `array`-like of shape (`n_samples` * `n_features`)
        matrix representing the data (features)

    Y : `array`-like of shape (`n_samples` * 1)
        matrix representing the labels

    nvector : `array`-like of shape (`number_training_size` * 1)
            vector representing different training size

    nrep : `int`
        number of repetitions for the simulations

    phi : `str` or `BasePhi` instance, default = 'fourier'
        Type of feature mapping function to use for mapping the input data.
        The currently available feature mapping methods are
        'fourier', 'relu', 'threshold' and 'linear'.

    rho1 : `float`, default = 0.2
        Noise rate on class 1
        :math:` \\mathbb{P}(ytilde = 2 | ytrue = 1)`

    rho2 : `float`, default = 0.2
        Noise rate on class 2
        :math:` \\mathbb{P}(ytilde = 1 | ytrue = 2)`


    Returns
    -------
    summary_data:   Pandas Dataframe
                    summary of all the simulation's setting
                    (loss, feat. mapp, rhos)

    summary_results: Pandas Dataframe
                     summary of the main results
                     (errors of the methods, times and eventually bounds)

    results_models:  Pandas Dataframe
                     containing the detailed information regarding each modela of each repetition and training size

    """

    n_samples, n_features = X.shape
    classes_ = np.unique(Y)
    n_classes = len(classes_)

    # TRASFORMA I DATI CON FEAT. MAPPING
    # Feature mappings
    fit_intercept = False
    random_state = False
    if phi == 'fourier':
        phi = RandomFourierPhi(n_classes=n_classes,
                               fit_intercept=fit_intercept,
                               random_state=random_state,
                               **phi_kwargs)
    elif phi == 'linear':
        phi = BasePhi(n_classes=n_classes,
                      fit_intercept=fit_intercept,
                      **phi_kwargs)
    elif phi == 'threshold':
        phi = ThresholdPhi(n_classes=n_classes,
                           fit_intercept=fit_intercept,
                           **phi_kwargs)
    elif phi == 'relu':
        phi = RandomReLUPhi(n_classes=n_classes,
                            fit_intercept=fit_intercept,
                            random_state=random_state,
                            **phi_kwargs)
    elif not isinstance(phi, BasePhi):
        raise ValueError('Unexpected feature mapping type ... ')

    # Fit the feature mappings
    phi.fit(X, Y)
    X_feat = phi.transform(X)

    results_models = pd.DataFrame(
        {'rep': [],
         'ntrain': [],
         'lr_cleansed': [],
         }
    )

    T = np.array([[1 - rho1, rho2], [rho1, 1 - rho2]])
    Ytilda = corruption(Y, T)

    my_shape = (nrep, len(nvector))

    # cleanlab method
    tot_time, tot_err, tot_ble = [np.zeros(my_shape) for _ in range(3)]

    for rep in range(nrep):

        # cleanse the labels and use only these:
        cl_rep, Ycleansed = clean_labels(X_feat, Ytilda)

        seed = rep
        print("Rep:", rep)

        X_train, X_test, y_cleansed_train, y_cleansed_test, \
            train_indx, test_indx = train_test_split(X_feat, Ycleansed, range(n_samples), test_size=0.2,
                                                     random_state=seed, stratify=Ycleansed)

        # retrieve the clean and cleansed test sets:
        y_test = Y[test_indx]

        for i, ntrain in enumerate(nvector):

            print("training size: ", ntrain)

            X_tr = X_train[0:ntrain, :]
            y_tr = y_cleansed_train[0:ntrain]

            # CLASSIC LOGISTIC REGRESSION ON CLEANSED LABELS
            lr = my_lr(X_tr, y_tr, X_test, y_test, max_iter=300)
            lr.ble = np.average(lr.y_pred != y_cleansed_test)

            tot_time[rep, i] = lr.time
            tot_err[rep, i] = lr.error
            tot_ble[rep, i] = lr.ble

            results_models = results_models._append(
                {'rep': rep,
                 'ntrain': ntrain,
                 'lr_cleansed': {'model': lr, 'noise_matrix_used_in_learning': None,
                                 'noise_matrix_used_in_corruption': T, 'predicted_labels': [lr.y_pred],
                                 'true_labels': [y_test], 'cleansed_labels': [y_cleansed_test]}
                 }, ignore_index=True
            )

    summary_data = pd.DataFrame(
        data={
            'n_samples': '%d' % n_samples,
            'n_attributes': '%d' % n_features,
            'phi': phi,
            'rho1': '%f' % rho1,
            'rho2': '%f' % rho2,
            'noise_matrix': [T],
            'ntrain_vector': [nvector],
        }
    )
    summary_results = pd.DataFrame(
        data={
            'lr_cleansed': {'biased_loss': tot_ble, 'unbiased_loss': [], 'errors': tot_err, 'times': tot_time}
        }
    )

    return summary_data, summary_results, results_models


def simulation_perfeval_cv_mrc(X, Y, nrep, nfolds, test_size, det, lambda0, phi='fourier', loss='0-1', rho1=0.2, rho2=0.2):
    """
    Simulation for MRCs models, with cross validation on noisy test data.

    Parameters
    ----------

    X : `array`-like of shape (`n_samples` * `n_features`)
        matrix representing the data (features)

    Y : `array`-like of shape (`n_samples` * 1)
        matrix representing the labels

    nrep : `int`
        number of repetitions for the simulations

    nfolds : `int`
        number of folds used in the k-fold cross-validation split

    test_size : `float`
        percentage of samples kept out for test
        (for mortality dataset: 0.6 recommended)

    lambda0 : `float`
        hyperparameter of MRC that controls the size of the set U:
        lambda = lambda0 * std/ sqrt(ntrain)
        Typically: lambda0 = 0.3, 0.7, 1

    det : `bool`
        True if we want to use the deterministic rule to predict
        False if we want to use the probabilistic rule to predict

    phi : `str` or `BasePhi` instance, default = 'fourier'
        Type of feature mapping function to use for mapping the input data.
        The currently available feature mapping methods are
        'fourier', 'relu', 'threshold' and 'linear'.

    loss : `str` {'0-1', 'log'}, default = '0-1'
        Type of loss function to use for the risk minimization.

    rho1 : `float`, default = 0.2
        Noise rate on class 1
        :math:` \\mathbb{P}(ytilda = 2 | ytrue = 1)`

    rho2 : `float`, default = 0.2
        Noise rate on class 2
        :math:` \\mathbb{P}(ytilda = 1 | ytrue = 2)`


    Returns
    -------
    summary_data:   Pandas Dataframe
                    summary of all the simulation's setting
                    (loss, feat. mapp, rhos)

    summary_results: Pandas Dataframe
                     summary of the main results
                     (errors of the methods, times and eventually bounds)

    results_models:  Pandas Dataframe
                     containing the detailed information regarding each modela of each repetition and training size

    """

    n_samples, n_features = X.shape

    T = np.array([[1 - rho1, rho2], [rho1, 1 - rho2]])

    # corrupt the data with fixed seed
    Ytilda = corruption(Y, T, seed=1)

    my_shape = (nrep, 1)
    mrc1_BLE, mrc2_BLE = [np.zeros(my_shape) for _ in range(2)]
    mrc1_ULE, mrc2_ULE = [np.zeros(my_shape) for _ in range(2)]
    mrc1_err, mrc2_err = [np.zeros(my_shape) for _ in range(2)]
    mrc1_opt, mrc2_opt = [np.zeros(my_shape) for _ in range(2)]

    for rep in range(nrep):

        seed = rep
        print('Rep ', rep)
        # split data into training and test sets:
        X_train, X_test, y_tilda_train, y_tilda_test, \
            train_indx, test_indx = train_test_split(X, Ytilda, range(n_samples), test_size=test_size, random_state=seed,
                                                     stratify=Ytilda)

        # retrieve the clean test and training set:
        y_test = Y[test_indx]

        mrc1 = my_mrc(phi, loss, det, lambda0, seed, 'classic', X_train, y_tilda_train, X_test, y_test)  # w/out correction
        mrc2 = my_mrc(phi, loss, det, lambda0, seed, 'backward', X_train, y_tilda_train, X_test, y_test, T, projection=True)

        skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)

        BLE1_cv, BLE2_cv = [np.zeros(shape=(nfolds, 1)) for _ in range(2)]
        ULE1_cv, ULE2_cv = [np.zeros(shape=(nfolds, 1)) for _ in range(2)]

        for k, (tr_indx, val_indx) in enumerate(skf.split(X_train, y_tilda_train)):

            print('Fold ', k, ' of ', nfolds)
            X_tr, X_val = X_train[tr_indx], X_train[val_indx]
            y_tr, y_val = y_tilda_train[tr_indx], y_tilda_train[val_indx]
            print('training size ', y_tr.shape)
            print('validation size ', y_val.shape)

            mrc1_cv = my_mrc(phi, loss, det, lambda0, seed, 'classic', X_tr, y_tr, X_val, y_val)
            mrc2_cv = my_mrc(phi, loss, det, lambda0, seed, 'backward', X_tr, y_tr, X_val, y_val, T, projection=True)

            mrc1_cv.T = T
            BLE1_cv[k] = mrc1_cv.error
            ULE1_cv[k] = unbiased_loss(mrc1_cv, y_val)

            BLE2_cv[k] = mrc2_cv.error
            ULE2_cv[k] = unbiased_loss(mrc2_cv, y_val)

        mrc1_BLE[rep] = np.mean(BLE1_cv)
        mrc1_ULE[rep] = np.mean(ULE1_cv)
        mrc1_opt[rep] = mrc1.upper_
        mrc1_err[rep] = mrc1.error  # we cannot do this in practice

        mrc2_BLE[rep] = np.mean(BLE2_cv)
        mrc2_ULE[rep] = np.mean(ULE2_cv)
        mrc2_opt[rep] = mrc2.upper_
        mrc2_err[rep] = mrc2.error  # we cannot do this in practice

    summary_data = pd.DataFrame(
        data={
            'n_samples': '%d' % n_samples,
            'n_attributes': '%d' % n_features,
            'phi': phi,
            'loss': loss,
            'deterministic_rule': det,
            'lambda0': lambda0,
            'rho1': '%f' % rho1,
            'rho2': '%f' % rho2,
            'noise_matrix': [T],
            'n_folds': nfolds
        }
    )

    summary_results = pd.DataFrame(
        {'mrc_nocorr': {'biased_loss': mrc1_BLE, 'unbiased_loss': mrc1_ULE, 'bounds': mrc1_opt, 'errors': mrc1_err},
         'mrc_back': {'biased_loss': mrc2_BLE, 'unbiased_loss': mrc2_ULE, 'bounds': mrc2_opt, 'errors': mrc2_err},
         }
    )

    return summary_data, summary_results


def simulation_perfeval_cv_lr_nat(X, Y, nrep, nfolds, test_size, phi='fourier', rho1=0.2, rho2=0.2, **phi_kwargs):
    """
    Simulation for Logistic Regssion and Natarajan models, with cross validation on noisy test data.

    Parameters
    ----------

    X : `array`-like of shape (`n_samples` * `n_features`)
        matrix representing the data (features)

    Y : `array`-like of shape (`n_samples` * 1)
        matrix representing the labels

    nrep : `int`
        number of repetitions for the simulations

    nfolds : `int`
        number of folds used in the k-fold cross-validation split

    test_size : `float`
        percentage of samples kept out for test
        (for mortality dataset: 0.6 recommended)

    phi : `str` or `BasePhi` instance, default = 'fourier'
        Type of feature mapping function to use for mapping the input data.
        The currently available feature mapping methods are
        'fourier', 'relu', 'threshold' and 'linear'.

    loss : `str` {'0-1', 'log'}, default = '0-1'
        Type of loss function to use for the risk minimization.

    rho1 : `float`, default = 0.2
        Noise rate on class 1
        :math:` \\mathbb{P}(ytilda = 2 | ytrue = 1)`

    rho2 : `float`, default = 0.2
        Noise rate on class 2
        :math:` \\mathbb{P}(ytilda = 1 | ytrue = 2)`


    Returns
    -------
    summary_data:   Pandas Dataframe
                    summary of all the simulation's setting
                    (loss, feat. mapp, rhos)

    summary_results: Pandas Dataframe
                     summary of the main results
                     (errors of the methods, times and eventually bounds)

    results_models:  Pandas Dataframe
                     containing the detailed information regarding each modela of each repetition and training size

    """

    n_samples, n_features = X.shape
    classes_ = np.unique(Y)
    n_classes = len(classes_)

    # TRASFORMA I DATI CON FEAT. MAPPING
    # Feature mappings
    fit_intercept = False
    random_state = False
    if phi == 'fourier':
        phi = RandomFourierPhi(n_classes=n_classes,
                               fit_intercept=fit_intercept,
                               random_state=random_state,
                               **phi_kwargs)
    elif phi == 'linear':
        phi = BasePhi(n_classes=n_classes,
                      fit_intercept=fit_intercept,
                      **phi_kwargs)
    elif phi == 'threshold':
        phi = ThresholdPhi(n_classes=n_classes,
                           fit_intercept=fit_intercept,
                           **phi_kwargs)
    elif phi == 'relu':
        phi = RandomReLUPhi(n_classes=n_classes,
                            fit_intercept=fit_intercept,
                            random_state=random_state,
                            **phi_kwargs)
    elif not isinstance(phi, BasePhi):
        raise ValueError('Unexpected feature mapping type ... ')

    # Fit the feature mappings
    phi.fit(X, Y)
    X_feat = phi.transform(X)

    T = np.array([[1 - rho1, rho2], [rho1, 1 - rho2]])
    Ytilda = corruption(Y, T)

    my_shape = (nrep, 1)
    lr_BLE, nat_BLE = [np.zeros(my_shape) for _ in range(2)]
    lr_ULE, nat_ULE = [np.zeros(my_shape) for _ in range(2)]
    lr_err, nat_err = [np.zeros(my_shape) for _ in range(2)]

    for rep in range(nrep):

        seed = rep
        print('Rep ', rep)
        # split data into training and test sets:
        X_train, X_test, y_tilda_train, y_tilda_test, \
            train_indx, test_indx = train_test_split(X_feat, Ytilda, range(n_samples), test_size=test_size, random_state=seed,
                                                     stratify=Ytilda)

        # retrieve the clean test and training set:
        y_test = Y[test_indx]

        lr = my_lr(X_train, y_tilda_train, X_test, y_test, max_iter=300)
        nat = my_natarajan(X_train, y_tilda_train, X_test, y_test, T, regularization='ridge')

        skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)

        lr_BLE_cv, nat_BLE_cv = [np.zeros(shape=(nfolds, 1)) for _ in range(2)]
        lr_ULE_cv, nat_ULE_cv = [np.zeros(shape=(nfolds, 1)) for _ in range(2)]

        for k, (tr_indx, val_indx) in enumerate(skf.split(X_train, y_tilda_train)):

            print('Fold ', k, ' of ', nfolds)
            X_tr, X_val = X_train[tr_indx], X_train[val_indx]
            y_tr, y_val = y_tilda_train[tr_indx], y_tilda_train[val_indx]
            print('training size ', y_tr.shape)
            print('validation size ', y_val.shape)

            lr_cv = my_lr(X_tr, y_tr,  X_val, y_val, max_iter=300)
            nat_cv = my_natarajan(X_tr, y_tr,  X_val, y_val, T, regularization='ridge')

            lr_cv.T = T
            lr_BLE_cv[k] = lr_cv.error
            nat_BLE_cv[k] = nat_cv.error

            lr_ULE_cv[k] = unbiased_loss(lr_cv, y_val)
            nat_ULE_cv[k] = unbiased_loss(nat_cv, y_val)

        lr_BLE[rep] = np.mean(lr_BLE_cv)
        lr_ULE[rep] = np.mean(lr_ULE_cv)
        lr_err[rep] = lr.error  # we cannot do this in practice

        nat_BLE[rep] = np.mean(nat_BLE_cv)
        nat_ULE[rep] = np.mean(nat_ULE_cv)
        nat_err[rep] = nat.error  # we cannot do this in practice

    summary_data = pd.DataFrame(
        data={
            'n_samples': '%d' % n_samples,
            'n_attributes': '%d' % n_features,
            'phi': phi,
            'rho1': '%f' % rho1,
            'rho2': '%f' % rho2,
            'noise_matrix': [T],
            'n_folds': nfolds
        }
    )

    summary_results = pd.DataFrame(
        {'lr_nocorr': {'biased_loss': lr_BLE, 'unbiased_loss': lr_ULE, 'bounds': [], 'errors': lr_err},
         'natarajan': {'biased_loss': nat_BLE, 'unbiased_loss': nat_ULE, 'bounds': [], 'errors': nat_err},
         }
    )

    return summary_data, summary_results


def simulation_perfeval_cv_cleanlab(X, Y, nrep, nfolds, test_size, phi='fourier', rho1=0.2, rho2=0.2, **phi_kwargs):
    """
    Simulation for CleanLab model, with cross validation on noisy test data.

    Parameters
    ----------

    X : `array`-like of shape (`n_samples` * `n_features`)
        matrix representing the data (features)

    Y : `array`-like of shape (`n_samples` * 1)
        matrix representing the labels

    nrep : `int`
        number of repetitions for the simulations

    nfolds : `int`
        number of folds used in the k-fold cross-validation split

    test_size : `float`
        percentage of samples kept out for test
        (for mortality dataset: 0.6 recommended)

    phi : `str` or `BasePhi` instance, default = 'fourier'
        Type of feature mapping function to use for mapping the input data.
        The currently available feature mapping methods are
        'fourier', 'relu', 'threshold' and 'linear'.

    loss : `str` {'0-1', 'log'}, default = '0-1'
        Type of loss function to use for the risk minimization.

    rho1 : `float`, default = 0.2
        Noise rate on class 1
        :math:` \\mathbb{P}(ytilda = 2 | ytrue = 1)`

    rho2 : `float`, default = 0.2
        Noise rate on class 2
        :math:` \\mathbb{P}(ytilda = 1 | ytrue = 2)`

    Returns
    -------
    summary_data:   Pandas Dataframe
                    summary of all the simulation's setting
                    (loss, feat. mapp, rhos)

    summary_results: Pandas Dataframe
                     summary of the main results
                     (errors of the methods, times and eventually bounds)

    results_models:  Pandas Dataframe
                     containing the detailed information regarding each modela of each repetition and training size

    """

    n_samples, n_features = X.shape
    classes_ = np.unique(Y)
    n_classes = len(classes_)

    # TRANSFORM DATA WITH FEAT. MAPPING
    # Feature mappings
    fit_intercept = False
    random_state = False
    if phi == 'fourier':
        phi = RandomFourierPhi(n_classes=n_classes,
                               fit_intercept=fit_intercept,
                               random_state=random_state,
                               **phi_kwargs)
    elif phi == 'linear':
        phi = BasePhi(n_classes=n_classes,
                      fit_intercept=fit_intercept,
                      **phi_kwargs)
    elif phi == 'threshold':
        phi = ThresholdPhi(n_classes=n_classes,
                           fit_intercept=fit_intercept,
                           **phi_kwargs)
    elif phi == 'relu':
        phi = RandomReLUPhi(n_classes=n_classes,
                            fit_intercept=fit_intercept,
                            random_state=random_state,
                            **phi_kwargs)
    elif not isinstance(phi, BasePhi):
        raise ValueError('Unexpected feature mapping type ... ')

    # Fit the feature mappings
    phi.fit(X, Y)
    X_feat = phi.transform(X)

    T = np.array([[1 - rho1, rho2], [rho1, 1 - rho2]])
    Ytilda = corruption(Y, T)

    my_shape = (nrep, 1)
    lr_ULE, lr_err = [np.zeros(my_shape) for _ in range(2)]  # Classic Loss Estimator

    for rep in range(nrep):

        seed = rep
        print('Rep ', rep)
        # split data into training and test sets:
        X_train, X_test, y_tilda_train, y_tilda_test, \
            train_indx, test_indx = train_test_split(X_feat, Ytilda, range(n_samples), test_size=test_size, random_state=seed,
                                                     stratify=Ytilda)

        # retrieve the clean test and training set:
        y_test = Y[test_indx]
        # cleanse the labels at training:
        cl_rep, y_cleansed_train = clean_labels(X_train, y_tilda_train)

        # trained on cleansed labels and tested on truly clean labels:
        lr = my_lr(X_train, y_cleansed_train, X_test, y_test, max_iter=300)

        skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)

        lr_ULE_cv = np.zeros(shape=(nfolds, 1))

        for k, (tr_indx, val_indx) in enumerate(skf.split(X_train, y_cleansed_train)):

            print('Fold ', k, ' of ', nfolds)
            X_tr, X_val = X_train[tr_indx], X_train[val_indx]
            y_tr, y_val = y_cleansed_train[tr_indx], y_cleansed_train[val_indx]
            print('training size ', y_tr.shape)
            print('validation size ', y_val.shape)

            # trained and tested on cleansed labels
            lr_cv = my_lr(X_tr, y_tr,  X_val, y_val, max_iter=300)

            lr_cv.T = T
            lr_ULE_cv[k] = lr_cv.error

        lr_ULE[rep] = np.mean(lr_ULE_cv)
        lr_err[rep] = lr.error  # we cannot do this in practice

    summary_data = pd.DataFrame(
        data={
            'n_samples': '%d' % n_samples,
            'n_attributes': '%d' % n_features,
            'phi': phi,
            'rho1': '%f' % rho1,
            'rho2': '%f' % rho2,
            'noise_matrix': [T],
            'n_folds': nfolds
        }
    )

    summary_results = pd.DataFrame(
        {'lr_cleansed': {'biased_loss': lr_ULE, 'unbiased_loss': [], 'bounds': [], 'errors': lr_err},
         }
    )

    return summary_data, summary_results
