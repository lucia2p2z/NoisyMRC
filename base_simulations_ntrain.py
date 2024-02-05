from sklearn.model_selection import train_test_split

from general_utilities import *
from sota_utilities import *

# to transform the data
from MRCpy.phi import \
    BasePhi, \
    RandomFourierPhi, \
    RandomReLUPhi, \
    ThresholdPhi

# to use cleanlab
from sklearn.linear_model import LogisticRegression
from cleanlab.classification import CleanLearning
from cleanlab.count import estimate_noise_matrices


def simulation_ntrain_mrc(X, Y, nvector, nrep, det, lambda0, phi='fourier', loss='0-1', rho1=0.2, rho2=0.2):
    """
    Simulation for different training size for MRCs models.

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

    results_models = pd.DataFrame(
        {'rep': [],
         'ntrain': [],
         'mrc_clean': [],
         'mrc_nocorr': [],
         'mrc_back': []
         }
    )
    T = np.array([[1 - rho1, rho2], [rho1, 1 - rho2]])

    # corrupt the data with fixed seed
    Ytilda = corruption(Y, T)

    my_shape = (nrep, len(nvector))
    tot_time1, tot_time2, tot_time3 = [np.zeros(my_shape) for _ in range(3)]
    tot_err1, tot_err2, tot_err3 = [np.zeros(my_shape) for _ in range(3)]
    tot_bd1, tot_bd2, tot_bd3 = [np.zeros(my_shape) for _ in range(3)]

    for rep in range(nrep):

        seed = rep
        print("Rep:", rep)

        # split data into training and test sets:
        X_train, X_test, y_tilda_train, y_tilda_test, \
            train_indx, test_indx = train_test_split(X, Ytilda, range(n_samples), test_size=0.2, random_state=seed,
                                                     stratify=Ytilda)

        # retrieve the clean test and training set:
        y_train = Y[train_indx]
        y_test = Y[test_indx]

        for i, ntrain in enumerate(nvector):

            X_tr = X_train[0:ntrain, :]
            y_tr = y_train[0:ntrain]
            y_ti = y_tilda_train[0:ntrain]

            mrc1 = my_mrc(phi, loss, det, lambda0, seed, 'classic', X_tr, y_tr, X_test, y_test)
            mrc2 = my_mrc(phi, loss, det, lambda0, seed, 'classic', X_tr, y_ti, X_test, y_test)
            mrc3 = my_mrc(phi, loss, det, lambda0, seed, 'backward', X_tr, y_ti, X_test, y_test, T, projection=True)

            tot_time1[rep, i] = mrc1.time
            tot_time2[rep, i] = mrc2.time
            tot_time3[rep, i] = mrc3.time

            tot_bd1[rep, i] = mrc1.upper_
            tot_bd2[rep, i] = mrc2.upper_
            tot_bd3[rep, i] = mrc3.upper_

            tot_err1[rep, i] = mrc1.error
            tot_err2[rep, i] = mrc2.error
            tot_err3[rep, i] = mrc3.error

            results_models = results_models._append(
                {'rep': rep,
                 'ntrain': ntrain,
                 'mrc_clean': {'model': mrc1,
                               'noise_matrix_used_in_learning': None, 'noise_matrix_used_in_corruption': None,
                               'predicted_labels': [mrc1.y_pred], 'true_labels': [y_test]},
                 'mrc_nocorr': {'model': mrc2,
                                'noise_matrix_used_in_learning': None, 'noise_matrix_used_in_corruption': T,
                                'predicted_labels': [mrc2.y_pred], 'true_labels': [y_test]},
                 'mrc_back': {'model': mrc3,
                              'noise_matrix_used_in_learning': T, 'noise_matrix_used_in_corruption': T,
                              'predicted_labels': [mrc3.y_pred], 'true_labels': [y_test]}
                 }, ignore_index=True
            )

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
            'ntrain_vector': [nvector],
        }
    )
    summary_results = pd.DataFrame(
        {'mrc_clean': {'errors': tot_err1, 'bounds': tot_bd1, 'times': tot_time1},
         'mrc_nocorr':  {'errors': tot_err2, 'bounds': tot_bd2, 'times': tot_time2},
         'mrc_back':  {'errors': tot_err3, 'bounds': tot_bd3, 'times': tot_time3},
         }
    )

    return summary_data, summary_results, results_models


def simulation_ntrain_lr(X, Y, nvector, nrep, phi='fourier', rho1=0.2, rho2=0.2, **phi_kwargs):
    """
    Simulation for different training size for Logistic regression models.

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
         'lr_clean': [],
         'lr_nocorr': [],
         }
    )

    T = np.array([[1 - rho1, rho2], [rho1, 1 - rho2]])
    Ytilda = corruption(Y, T)

    my_shape = (nrep, len(nvector))

    tot_time1, tot_time2 = [np.zeros(my_shape) for _ in range(2)]
    tot_err1, tot_err2 = [np.zeros(my_shape) for _ in range(2)]

    for rep in range(nrep):

        seed = rep
        print("Rep:", rep)

        # split data into training and test sets:
        X_train, X_test, y_tilda_train, y_tilda_test, \
            train_indx, test_indx = train_test_split(X_feat, Ytilda, range(n_samples), test_size=0.2, random_state=seed,
                                                     stratify=Ytilda)

        # retrieve the clean test and training set:
        y_train = Y[train_indx]
        y_test = Y[test_indx]

        for i, ntrain in enumerate(nvector):

            print("training size: ", ntrain)
            X_tr = X_train[0:ntrain, :]
            y_tr = y_train[0:ntrain]
            y_ti = y_tilda_train[0:ntrain]

            # LR on clean labels
            lr1 = my_lr(X_tr, y_tr, X_test, y_test, max_iter=300)

            # LR on noisy labels without correction
            lr2 = my_lr(X_tr, y_ti, X_test, y_test, max_iter=300)

            tot_time1[rep, i] = lr1.time
            tot_time2[rep, i] = lr2.time
            tot_err1[rep, i] = lr1.error
            tot_err2[rep, i] = lr2.error

            results_models = results_models._append(
                {'rep': rep,
                 'ntrain': ntrain,
                 'lr_clean': {'model': lr1, 'noise_matrix_used_in_learning': None, 'noise_matrix_used_in_corruption': None, 'predicted_labels': [lr1.y_pred],
                              'true_labels': [y_test]},
                 'lr_nocorr': {'model': lr2, 'noise_matrix_used_in_learning': None, 'noise_matrix_used_in_corruption': [T], 'predicted_labels': [lr2.y_pred],
                               'true_labels': [y_test]},
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
        {'lr_clean': {'errors': tot_err1, 'times': tot_time1},
         'lr_nocorr': {'errors': tot_err2, 'times': tot_time2},
         }
    )

    return summary_data, summary_results, results_models


def simulation_ntrain_natarajan(X, Y, nvector, nrep, phi='fourier', rho1=0.2, rho2=0.2, **phi_kwargs):
    """
    Simulation for different training size for Natarajan Model.

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
         'natarajan': [],
         }
    )

    T = np.array([[1 - rho1, rho2], [rho1, 1 - rho2]])
    Ytilda = corruption(Y, T)

    my_shape = (nrep, len(nvector))

    tot_time, tot_err = [np.zeros(my_shape) for _ in range(2)]

    for rep in range(nrep):

        seed = rep
        print("Rep:", rep)

        # split data into training and test sets:
        X_train, X_test, y_tilda_train, y_tilda_test, \
            train_indx, test_indx = train_test_split(X_feat, Ytilda, range(n_samples), test_size=0.2, random_state=seed,
                                                     stratify=Ytilda)

        # retrieve the clean test and training set:
        y_train = Y[train_indx]
        y_test = Y[test_indx]

        for i, ntrain in enumerate(nvector):

            print("training size: ", ntrain)
            X_tr = X_train[0:ntrain, :]
            y_tr = y_train[0:ntrain]
            y_ti = y_tilda_train[0:ntrain]

            # NATARAJAN METHOD: modified logistic regression
            nat = my_natarajan(X_tr, y_ti, X_test, y_test, T, regularization='ridge')

            tot_time[rep, i] = nat.time
            tot_err[rep, i] = nat.error

            results_models = results_models._append(
                {'rep': rep,
                 'ntrain': ntrain,
                 'natarajan': {'model': nat,
                               'noise_matrix_used_in_learning': [T], 'noise_matrix_used_in_corruption': [T],
                               'predicted_labels': [nat.y_pred], 'true_labels': [y_test]}
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
        {'natarajan': {'errors': tot_err, 'times': tot_time},
         }
    )

    return summary_data, summary_results, results_models


def simulation_ntrain_cleanlab(X, Y, nvector, nrep, phi='fourier', rho1=0.2, rho2=0.2, **phi_kwargs):
    """
    Simulation for different training size and using T_estimated provided
    by the library cleanlab (https://docs.cleanlab.ai/stable/index.html)

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
    tot_time_cl, tot_err_cl = [np.zeros(my_shape) for _ in range(2)]

    for rep in range(nrep):

        seed = rep
        print("Rep:", rep)

        X_train, X_test, y_tilda_train, y_tilda_test, \
            train_indx, test_indx = train_test_split(X_feat, Ytilda, range(n_samples), test_size=0.2,
                                                     random_state=seed, stratify=Ytilda)

        # retrieve the clean training and test sets:
        y_train = Y[train_indx]
        y_test = Y[test_indx]

        for i, ntrain in enumerate(nvector):

            print("training size: ", ntrain)

            X_tr = X_train[0:ntrain, :]
            y_tr = y_train[0:ntrain]
            y_ti = y_tilda_train[0:ntrain]

            # METHOD OF CLEANLAB
            start_time = time.time()
            cl = CleanLearning(clf=LogisticRegression(max_iter=400))  # any sklearn-compatible classifier
            cl.fit(X_tr, y_ti)
            end_time = time.time()
            elapsed_time = end_time - start_time
            y_pred = cl.predict(X_test)
            cl.y_pred = y_pred
            cl.error = np.average(y_pred != y_test)
            cl.time = elapsed_time

            tot_time_cl[rep, i] = cl.time
            tot_err_cl[rep, i] = cl.error

            results_models = results_models._append(
                {'rep': rep,
                 'ntrain': ntrain,
                 'cl': {'model': cl, 'noise_matrix_used_in_learning': None,
                        'noise_matrix_used_in_corruption': T, 'predicted_labels': [cl.y_pred],
                        'true_labels': [y_test]}
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
            'cl': {'errors': tot_err_cl, 'times': tot_time_cl}
        }
    )

    return summary_data, summary_results, results_models


def simulation_ntrain_mrc_est(X, Y, nvector, nrep, det, lambda0, phi='fourier', loss='0-1', rho1=0.2, rho2=0.2):
    """
    Simulation for different training size and using MRC + T_estimated.

    T_estimated is provided by the library cleanlab (https://docs.cleanlab.ai/stable/index.html)

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

    results_models = pd.DataFrame(
        {'rep': [],
         'ntrain': [],
         'mrc_back_est': [],
         }
    )

    T = np.array([[1 - rho1, rho2], [rho1, 1 - rho2]])
    Ytilda = corruption(Y, T)

    my_shape = (nrep, len(nvector))

    # mrc + T estimated
    tot_time_mrc, tot_err_mrc, tot_bd_mrc = [np.zeros(my_shape) for _ in range(3)]

    train_ratio = 0.6
    validation_ratio = 0.30
    test_ratio = 0.10

    for rep in range(nrep):

        seed = rep
        print("Rep:", rep)

        # train is now 75% of the entire data set
        X_train, X_rem, y_tilda_train, y_tilda_rem, \
            train_indx, rem_indx = train_test_split(X, Ytilda, range(n_samples), test_size=1 - train_ratio,
                                                    random_state=seed, stratify=Ytilda)

        # test is now 10% of the initial data set & validation is now 15% of the initial data set
        X_val, X_test, y_tilda_val, y_tilda_test, \
            val_indx, test_indx = train_test_split(X_rem, y_tilda_rem, range(len(y_tilda_rem)),
                                                   test_size=test_ratio / (test_ratio + validation_ratio),
                                                   random_state=seed, stratify=y_tilda_rem)

        # Retrieve the clean training and test sets:
        y_rem = Y[rem_indx]
        y_test = y_rem[test_indx]

        # Estimate T = P(obs_lab|true_lab) using cleanlab on the validation set
        T_est = estimate_noise_matrices(X_val, y_tilda_val, clf=LogisticRegression(max_iter=500), cv_n_folds=5,
                                        thresholds=None, converge_latent_estimates=True, seed=None, clf_kwargs={},
                                        validation_func=None)[0]
        print('Estimated T: ', T_est)

        for i, ntrain in enumerate(nvector):

            X_tr = X_train[0:ntrain, :]
            y_ti = y_tilda_train[0:ntrain]

            # MRC on noisy labels with backward correction on T estimated
            mrc = my_mrc(phi, loss, det, lambda0, seed, 'backward', X_tr, y_ti, X_test, y_test, T=T_est, projection=True)

            tot_time_mrc[rep, i] = mrc.time
            tot_bd_mrc[rep, i] = mrc.upper_
            tot_err_mrc[rep, i] = mrc.error

            results_models = results_models._append(
                {'rep': rep,
                 'ntrain': ntrain,
                 'mrc_back_est': {'model': mrc, 'noise_matrix_used_in_learning': T_est,
                                  'noise_matrix_used_in_corruption': T, 'predicted_labels': [mrc.y_pred],
                                  'true_labels': [y_test]},
                 }, ignore_index=True
            )

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
            'ntrain_vector': [nvector],
        }
    )
    summary_results = pd.DataFrame(
        data={
            'mrc_back_est': {'errors': tot_err_mrc, 'times': tot_time_mrc, 'bounds': tot_bd_mrc},
        }
    )

    return summary_data, summary_results, results_models


def simulation_ntrain_lrcleansed(X, Y, nvector, nrep, phi='fourier', rho1=0.2, rho2=0.2, **phi_kwargs):
    """
    Simulation for different training size and using cleansed labels for Logistic Regression.

    The cleansed labels are obtained using the library cleanlab (https://docs.cleanlab.ai/stable/index.html)

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

    loss : `str` {'0-1', 'log'}, default = '0-1'
        Type of loss function to use for the risk minimization.

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

    results_cleanlab: Pandas Dataframe
                      containing the detailed information regarding the cleaning process of the labels, of each rep.
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

    # log regression + cleansed labels
    tot_time_lr, tot_err_lr = [np.zeros(my_shape) for _ in range(2)]

    for rep in range(nrep):

        seed = rep
        print("Rep:", rep)

        # train is now 75% of the entire data set
        X_train, X_test, y_tilda_train, y_tilda_test, \
            train_indx, test_indx = train_test_split(X_feat, Ytilda, range(n_samples), test_size=0.2,
                                                     random_state=seed, stratify=Ytilda)

        # retrieve the clean test and training  sets:
        y_test = Y[test_indx]
        y_train = Y[train_indx]

        # clean the train noisy labels using cleanlab:
        cl_rep, y_cleansed_train = clean_labels(X_train, y_tilda_train)

        number_true_issues = np.count_nonzero(y_train != y_tilda_train)
        print(f'The exact number of label errors is {number_true_issues}')

        for i, ntrain in enumerate(nvector):

            X_tr = X_train[0:ntrain, :]
            y_cl = y_cleansed_train[0:ntrain]

            # LR on cleansed labels
            lr = my_lr(X_tr, y_cl, X_test, y_test, max_iter=300)

            tot_time_lr[rep, i] = lr.time
            tot_err_lr[rep, i] = lr.error

            results_models = results_models._append(
                {'rep': rep,
                 'ntrain': ntrain,
                 'lr_cleansed': {'model': lr, 'noise_matrix_used_in_learning': None,
                                 'noise_matrix_used_in_corruption': T, 'predicted_labels': [lr.y_pred],
                                 'true_labels': [y_test]},
                 }, ignore_index=True
            )

        results_cleanlab = pd.DataFrame(
            data={
                'rep': rep,
                'number_true_issues': number_true_issues,
                'number_found_issues': cl_rep.number_found_issues,
                'noisy_labels': [y_tilda_train],
                'true_labels': [y_train],
                'cleansed_labels': [y_cleansed_train],
                'labels_issues': cl_rep
            }
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
            'lr_cleansed': {'errors': tot_err_lr, 'times': tot_time_lr, 'bounds': None},
        }
    )

    return summary_data, summary_results, results_models, results_cleanlab


def simulation_ntrain_mrccleansed(X, Y, nvector, nrep, det, lambda0, phi='fourier', loss='0-1', rho1=0.2, rho2=0.2):
    """
    Simulation for different training size and using cleansed labels on MRC.

    The cleansed labels are obtained using the library cleanlab (https://docs.cleanlab.ai/stable/index.html)

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

    results_cleanlab: Pandas Dataframe
                      containing the detailed information regarding the cleaning process of the labels, of each rep.
    """

    n_samples, n_features = X.shape
    classes_ = np.unique(Y)
    n_classes = len(classes_)

    results_models = pd.DataFrame(
        {'rep': [],
         'ntrain': [],
         'mrc_cleansed': [],
         }
    )

    T = np.array([[1 - rho1, rho2], [rho1, 1 - rho2]])
    Ytilda = corruption(Y, T)

    my_shape = (nrep, len(nvector))

    # mrc + cleansed labels
    tot_time_mrc, tot_err_mrc, tot_bd_mrc = [np.zeros(my_shape) for _ in range(3)]

    for rep in range(nrep):

        seed = rep
        print("Rep:", rep)

        # train is now 75% of the entire data set
        X_train, X_test, y_tilda_train, y_tilda_test, \
            train_indx, test_indx = train_test_split(X, Ytilda, range(n_samples), test_size=0.2,
                                                     random_state=seed, stratify=Ytilda)

        # retrieve the clean test and training  sets:
        y_test = Y[test_indx]
        y_train = Y[train_indx]

        # clean the train noisy labels using cleanlab:
        cl_rep, y_cleansed_train = clean_labels(X_train, y_tilda_train)

        number_true_issues = np.count_nonzero(y_train != y_tilda_train)
        print(f'The exact number of label errors is {number_true_issues}')

        for i, ntrain in enumerate(nvector):

            X_tr = X_train[0:ntrain, :]
            y_cl = y_cleansed_train[0:ntrain]

            # MRC on cleansed labels
            mrc = my_mrc(phi, loss, det, lambda0, seed, 'classic', X_tr, y_cl, X_test, y_test)

            tot_time_mrc[rep, i] = mrc.time
            tot_bd_mrc[rep, i] = mrc.upper_
            tot_err_mrc[rep, i] = mrc.error

            results_models = results_models._append(
                {'rep': rep,
                 'ntrain': ntrain,
                 'mrc_cleansed': {'model': mrc, 'noise_matrix_used_in_learning': None,
                                  'noise_matrix_used_in_corruption': T, 'predicted_labels': [mrc.y_pred],
                                  'true_labels': [y_test]},
                 }, ignore_index=True
            )

        results_cleanlab = pd.DataFrame(
            data={
                'rep': rep,
                'number_true_issues': number_true_issues,
                'number_found_issues': cl_rep.number_found_issues,
                'noisy_labels': [y_tilda_train],
                'true_labels': [y_train],
                'cleansed_labels': [y_cleansed_train],
                'labels_issues': cl_rep
            }
        )

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
            'ntrain_vector': [nvector],
        }
    )
    summary_results = pd.DataFrame(
        data={
            'mrc_cleansed': {'errors': tot_err_mrc, 'times': tot_time_mrc, 'bounds': tot_bd_mrc},
        }
    )

    return summary_data, summary_results, results_models, results_cleanlab


def simulation_ntrain_nata_est(X, Y, nvector, nrep, phi='fourier', rho1=0.2, rho2=0.2,  **phi_kwargs):
    """
    Simulation for different training size and using Natarajan + T_estimated.

    T_estimated is provided by the library cleanlab (https://docs.cleanlab.ai/stable/index.html)

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

    loss : `str` {'0-1', 'log'}, default = '0-1'
        Type of loss function to use for the risk minimization.

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
         'natarajan_est': [],
         }
    )

    T = np.array([[1 - rho1, rho2], [rho1, 1 - rho2]])
    Ytilda = corruption(Y, T)

    my_shape = (nrep, len(nvector))

    # nata + T estimated
    tot_time_nata, tot_err_nata = [np.zeros(my_shape) for _ in range(2)]

    train_ratio = 0.6
    validation_ratio = 0.30
    test_ratio = 0.10

    for rep in range(nrep):

        seed = rep
        print("Rep:", rep)

        # train is now 75% of the entire data set
        X_train, X_rem, y_tilda_train, y_tilda_rem, \
            train_indx, rem_indx = train_test_split(X_feat, Ytilda, range(n_samples), test_size=1 - train_ratio,
                                                    random_state=seed, stratify=Ytilda)

        # test is now 10% of the initial data set & validation is now 15% of the initial data set
        X_val, X_test, y_tilda_val, y_tilda_test, \
            val_indx, test_indx = train_test_split(X_rem, y_tilda_rem, range(len(y_tilda_rem)),
                                                   test_size=test_ratio / (test_ratio + validation_ratio),
                                                   random_state=seed, stratify=y_tilda_rem)

        # Retrieve the clean training and test sets:
        y_rem = Y[rem_indx]
        y_test = y_rem[test_indx]

        # Estimate T = P(obs_lab|true_lab) using cleanlab on the validation set
        T_est = estimate_noise_matrices(X_val, y_tilda_val, clf=LogisticRegression(max_iter=500), cv_n_folds=5,
                                        thresholds=None, converge_latent_estimates=True, seed=None, clf_kwargs={},
                                        validation_func=None)[0]
        print('Estimated T: ', T_est)

        for i, ntrain in enumerate(nvector):

            X_tr = X_train[0:ntrain, :]
            y_ti = y_tilda_train[0:ntrain]

            # Natarajan on noisy labels with T estimated
            nat = my_natarajan(X_tr,y_ti, X_test, y_test, T=T_est, regularization='ridge')

            tot_time_nata[rep, i] = nat.time
            tot_err_nata[rep, i] = nat.error

            results_models = results_models._append(
                {'rep': rep,
                 'ntrain': ntrain,
                 'natarajan_est': {'model': nat,
                                   'noise_matrix_used_in_learning': [T_est], 'noise_matrix_used_in_corruption': [T],
                                   'predicted_labels': [nat.y_pred], 'true_labels': [y_test]},
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
            'nata_est': {'errors': tot_err_nata, 'times': tot_time_nata},
        }
    )

    return summary_data, summary_results, results_models

