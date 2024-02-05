'''
.. _base_mrc:
Super class for Minimax Risk Classifiers.
'''

import cvxpy as cvx
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

# Import the feature mapping
from MRCpy.phi import \
    BasePhi, \
    RandomFourierPhi, \
    RandomReLUPhi, \
    ThresholdPhi


class BaseMRC(BaseEstimator, ClassifierMixin):
    '''
    Base class for different minimax risk classifiers

    This class is a parent class for different MRCs
    implemented in the library.
    It defines the different parameters and the layout.

    .. seealso:: For more information about MRC, one can refer to the
    following resources:

                    [1] `Mazuelas, S., Zanoni, A., & Pérez, A. (2020).
                    Minimax Classification with
                    0-1 Loss and Performance Guarantees.
                    Advances in Neural Information Processing
                    Systems, 33, 302-312. <https://arxiv.org/abs/2010.07964>`_

                    [2] `Mazuelas, S., Shen, Y., & Pérez, A. (2020).
                    Generalized Maximum
                    Entropy for Supervised Classification.
                    arXiv preprint arXiv:2007.05447.
                    <https://arxiv.org/abs/2007.05447>`_

                    [3] `Bondugula, K., Mazuelas, S., & Pérez,
                    A. (2021). MRcvxy: A Library for Minimax Risk Classifiers.
                    arXiv preprint arXiv:2108.01952.
                    <https://arxiv.org/abs/2108.01952>`_

    Parameters
    ----------
    loss : `str`, default = '0-1'
        Type of loss function to use for the risk
        minimization.
        The options are 0-1 loss and logaritmic loss.
        '0-1'
            0-1 loss quantifies the probability of classification
            error at a certain example for a certain rule.
        'log'
            Log-loss quantifies the minus log-likelihood at a
            certain example for a certain rule.

    s : `float`, default = `0.3`
        Parameter that tunes the estimation of expected values
        of feature mapping function. It is used to calculate :math:`\lambda`
        (variance in the mean estimates
        for the expectations of the feature mappings) in the following way

        .. math::
            \\lambda = s * \\text{std}(\\phi(X,Y)) / \\sqrt{\\left| X \\right|}

        where (X,Y) is the dataset of training samples and their labels
        respectively and :math:`\\text{std}(\\phi(X,Y))` stands for
        standard deviation of :math:`\\phi(X,Y)` in the supervised
        dataset (X,Y).

    deterministic : `bool`, default = `True`
        Whether the prediction of the labels
        should be done in a deterministic way (given a fixed `random_state`
        in the case of using random Fourier or random ReLU features).

    random_state : `int`, RandomState instance, default = `None`
        Random seed used when 'fourier' and 'relu' options for feature mappings
        are used to produce the random weights.

    fit_intercept : `bool`, default = `True`
            Whether to calculate the intercept for MRCs
            If set to false, no intercept will be used in calculations
            (i.e. data is expected to be already centered).

    phi : `str` or `BasePhi` instance, default = 'linear'
        Type of feature mapping function to use for mapping the input data.
        The currenlty available feature mapping methods are
        'fourier', 'relu', 'threshold' and 'linear'.
        The users can also implement their own feature mapping object
        (should be a `BasePhi` instance) and pass it to this argument.
        Note that when using 'fourier' or 'relu' feature mappings,
        training and testing instances are expected to be normalized.
        To implement a feature mapping, please go through the
        :ref:`Feature Mapping` section.

        'linear'
            It uses the identity feature map referred to as Linear feature map.
            See class `BasePhi`.

        'fourier'
            It uses Random Fourier Feature map. See class `RandomFourierPhi`.

        'relu'
            It uses Rectified Linear Unit (ReLU) features.
            See class `RandomReLUPhi`.

        'threshold'
            It uses Feature mappings obtained using a threshold.
            See class `ThresholdPhi`.

    **phi_kwargs : Additional parameters for feature mappings.
                Groups the multiple optional parameters
                for the corresponding feature mappings (`phi`).

                For example in case of fourier features,
                the number of features is given by `n_components`
                parameter which can be passed as argument -
                `MRC(loss='log', phi='fourier', n_components=500)`

                The list of arguments for each feature mappings class
                can be found in the corresponding documentation.

    Attributes
    ----------
    is_fitted_ : `bool`
        Whether the classifier is fitted i.e., the parameters are learnt.

    tau_ : `array`-like of shape (`n_features`)
        Mean estimates for the expectations of feature mappings.

    lambda_ : `array`-like of shape (`n_features`)
        Variance in the mean estimates for the expectations
        of the feature mappings.

    classes_ : `array`-like of shape (`n_classes`)
        Labels in the given dataset.
        If the labels Y are not given during fit
        i.e., tau and lambda are given as input,
        then this array is None.
    '''

    def __init__(self,
                 loss='0-1',
                 s=0.3,
                 deterministic=True,
                 random_state=None,
                 fit_intercept=True,
                 phi='linear',
                 **phi_kwargs):

        self.loss = loss
        self.s = s
        self.deterministic = deterministic
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        # Feature mapping and its parameters
        self.phi = phi
        self.phi_kwargs = phi_kwargs
        # Solver list for cvxpy
        self.cvx_solvers = ['SCS', 'ECOS', 'GUROBI']

    def fit(self, X, Y, X_=None):
        '''
        Fit the MRC model.

        Computes the parameters required for the minimax risk optimization
        and then calls the `minimax_risk` function to solve the optimization.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
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

        X_ : array-like of shape (`n_samples2`, `n_dimensions`), default = None
            These instances are optional and
            when given, will be used in the minimax risk optimization.
            These extra instances are generally a smaller set and
            give an advantage in training time.

        Returns
        -------
        self :
            Fitted estimator

        '''

        X, Y = check_X_y(X, Y, accept_sparse=True)

        # Check if separate instances are given for the optimization
        if X_ is not None:
            X_opt = check_array(X_, accept_sparse=True)
            not_all_instances = False
        else:
            # Use the training samples used to compute estimate
            # for the optimization.
            X_opt = X

            # # If the labels are not given, then these instances
            # # are assumed to be given for optimization only and
            # # hence all the instances will be used.
            # if Y is None:
            #     not_all_instances = False
            # else:
            not_all_instances = True

        # Obtaining the number of classes and mapping the labels to integers
        origY = Y
        self.classes_ = np.unique(origY)
        n_classes = len(self.classes_)
        Y = np.zeros(origY.shape[0], dtype=int)

        # Map the values of Y from 0 to n_classes-1
        for i, y in enumerate(self.classes_):
            Y[origY == y] = i

        # Feature mappings
        if self.phi == 'fourier':
            self.phi = RandomFourierPhi(n_classes=n_classes,
                                        fit_intercept=self.fit_intercept,
                                        random_state=self.random_state,
                                        **self.phi_kwargs)
        elif self.phi == 'linear':
            self.phi = BasePhi(n_classes=n_classes,
                               fit_intercept=self.fit_intercept,
                               **self.phi_kwargs)
        elif self.phi == 'threshold':
            self.phi = ThresholdPhi(n_classes=n_classes,
                                    fit_intercept=self.fit_intercept,
                                    **self.phi_kwargs)
        elif self.phi == 'relu':
            self.phi = RandomReLUPhi(n_classes=n_classes,
                                     fit_intercept=self.fit_intercept,
                                     random_state=self.random_state,
                                     **self.phi_kwargs)
        elif not isinstance(self.phi, BasePhi):
            raise ValueError('Unexpected feature mapping type ... ')

        # Fit the feature mappings
        self.phi.fit(X, Y)

        # Compute the expectation estimates
        tau_ = self.phi.est_exp(X, Y)
        lambda_ = (self.s * self.phi.est_std(X, Y)) / \
            np.sqrt(X.shape[0])

        # Limit the number of training samples used in the optimization
        # for large datasets
        # Reduces the training time and use of memory
        n_max = 5000
        n = X_opt.shape[0]
        if not_all_instances and n_max < n:
            n = n_max

        # Fit the MRC classifier
        self.minimax_risk(X_opt[:n], tau_, lambda_, n_classes)

        return self


    def tau_projection_cvx(self, X, T, tau):
        '''
        Compute the vector that minimises the distance between the original tau
        and the convex hull defined by the set of distributions U.
        (the vector that satisfies this definition is simply the projection of original tau into U).

        Projection solved suing cvx

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Training instances used for solving
            the minimax risk optimization problem.

        T : `array`-like of shape (`n_classes` * `n_classes`)
            transition probability matrix describing the noise rates on the labels

        tau : `array`-like of shape (`n_features`* `n_classes`)
                    original vector tau, to be projected

        Returns
        -------
        proj_tau : `array`-like of shape (`n_features`* `n_classes`)
            projected tau

        '''

        n, d = X.shape
        phi = self.phi.eval_x(X)
        n_classes = T.shape[0]
        F = np.reshape(phi, (n * n_classes, self.phi.len_))

        m = int(self.phi.len_ / n_classes)
        I = np.eye(m)
        Ttot = np.kron(T, I)
        A = Ttot @ F.T

        # Variables
        pcvx = cvx.Variable(n_classes * n)

        #Objective function and constraints
        objective = cvx.Minimize(cvx.sum_squares(tau - A @ pcvx))
        constraints = self.compute_projection_constraints(pcvx)

        # problem = cvx.Problem(objective, constraints)
        # problem.solve()

        print('Solving Projection using : ', self.cvx_solvers[0])
        prob = cvx.Problem(objective, constraints)

        try:
            prob.solve(solver= self.cvx_solvers[0], verbose=False)
            pcvx_ = pcvx.value
        except:
            print('Error occured while using ' + self.cvx_solvers[0] + ' solver.\n' + \
                  'Trying with the following solvers ' + ' '.join(self.cvx_solvers[1:]))
            pcvx_ = None

        # if the solver could not find values of mu for the given solver
        if pcvx_ is None:

            # try with a different solver for solution
            for s in self.cvx_solvers[1:]:

                # Solve the problem
                try:
                    prob.solve(solver=s, verbose=False)
                    pcvx_ = pcvx.value
                except:
                    print('Error occured while using ' + self.cvx_solvers[0] + ' solver.')
                    pcvx_ = None

                # Check the values
                pcvx_ = pcvx.value

                # Break the loop once the solution is obtained
                if pcvx_ is not None:
                    break

        # If no solution can be found for the optimization.
        if pcvx_ is None:
            raise ValueError('CVXpy solver couldn\'t find a solution .... ' +
                             'The problem is ', prob.status)

        objective_value = prob.value
        # return mu_, objective_value

        proj_tau = np.dot(A, pcvx_)

        return proj_tau

    def compute_projection_constraints(self):
        '''
        Computes the constraints of the projection cvx problem (needed to compute the projected vecotr tau)
        for an MRC or a CMRC.
        '''
        raise NotImplementedError('BaseMRC is not an implemented classifier.' +
                                  ' It is base class for different MRCs.' +
                                  ' This functions needs to be implemented' +
                                  ' by a sub-class implementing a MRC.')

    def fit_noise(self, X, Y, T, projection=False, X_=None):
        '''
        Fit the MRC model.

        Computes the parameters required for the minimax risk optimization
        and then calls the `minimax_risk` function to solve the optimization.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
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

        T : `array`-like of shape (`n_classes`, `n_classes`)
            transition probability matrix describing the label noise.
            The general entry t_ij corresponds to the probability P(ytilda = i ! ytreu = j)
            where ytilda represents the currupted version of ytrue

        X_ : array-like of shape (`n_samples2`, `n_dimensions`), default = None
            These instances are optional and
            when given, will be used in the minimax risk optimization.
            These extra instances are generally a smaller set and
            give an advantage in training time.

        Returns
        -------
        self :
            Fitted estimator

        '''

        X, Y = check_X_y(X, Y, accept_sparse=True)

        # Check if separate instances are given for the optimization
        if X_ is not None:
            X_opt = check_array(X_, accept_sparse=True)
            not_all_instances = False
        else:
            # Use the training samples used to compute estimate
            # for the optimization.
            X_opt = X

            # # If the labels are not given, then these instances
            # # are assumed to be given for optimization only and
            # # hence all the instances will be used.
            # if Y is None:
            #     not_all_instances = False
            # else:
            not_all_instances = True

        # Obtaining the number of classes and mapping the labels to integers
        origY = Y
        self.classes_ = np.unique(origY)
        n_classes = len(self.classes_)
        Y = np.zeros(origY.shape[0], dtype=int)

        # Map the values of Y from 0 to n_classes-1
        for i, y in enumerate(self.classes_):
            Y[origY == y] = i

        # Feature mappings
        if self.phi == 'fourier':
            self.phi = RandomFourierPhi(n_classes=n_classes,
                                        fit_intercept=self.fit_intercept,
                                        random_state=self.random_state,
                                        **self.phi_kwargs)
        elif self.phi == 'linear':
            self.phi = BasePhi(n_classes=n_classes,
                               fit_intercept=self.fit_intercept,
                               **self.phi_kwargs)
        elif self.phi == 'threshold':
            self.phi = ThresholdPhi(n_classes=n_classes,
                                    fit_intercept=self.fit_intercept,
                                    **self.phi_kwargs)
        elif self.phi == 'relu':
            self.phi = RandomReLUPhi(n_classes=n_classes,
                                     fit_intercept=self.fit_intercept,
                                     random_state=self.random_state,
                                     **self.phi_kwargs)
        elif not isinstance(self.phi, BasePhi):
            raise ValueError('Unexpected feature mapping type ... ')

        # Fit the feature mappings
        self.phi.fit(X, Y)
        m = int(self.phi.len_ / n_classes)
        I = np.eye(m)
        tinv = np.linalg.inv(T)
        tinvtot = np.kron(tinv, I)

        # Compute the expectation estimates
        orig_tau = self.phi.est_exp(X, Y)
        self.orig_tau = orig_tau
        if projection:
            proj_tau = self.tau_projection_cvx(X, T, orig_tau)
            self.proj_tau = proj_tau
            tau_ = np.dot(tinvtot, proj_tau)
        else:
            tau_ = np.dot(tinvtot, orig_tau)

        lambda_ = (self.s * self.phi.est_std_noise(X, Y, tinvtot)) / \
            np.sqrt(X.shape[0])

        # Limit the number of training samples used in the optimization
        # for large datasets
        # Reduces the training time and use of memory
        n_max = 5000
        n = X_opt.shape[0]
        if not_all_instances and n_max < n:
            n = n_max

        # Fit the MRC classifier
        self.minimax_risk(X_opt[:n], tau_, lambda_, n_classes)

        return self

    def minimax_risk(self, X, tau_, lambda_, n_classes):
        '''
        Abstract function for sub-classes implementing
        the different MRCs.

        Solves the minimax risk optimization problem
        for the corresponding variant of MRC.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Training instances used for solving
            the minimax risk optimization problem.

        tau_ : `array`-like of shape (`n_features` * `n_classes`)
            Mean estimates
            for the expectations of feature mappings.

        lambda_ : `array`-like of shape (`n_features` * `n_classes`)
            Variance in the mean estimates
            for the expectations of the feature mappings.

        n_classes : `int`
            Number of labels in the dataset.

        Returns
        -------
        self :
            Fitted estimator

        '''

        # Variants of MRCs inheriting from this class should
        # extend this function to implement the solution to their
        # minimax risk optimization problem.

        raise NotImplementedError('BaseMRC is not an implemented classifier.' +
                                  ' It is base class for different MRCs.' +
                                  ' This functions needs to be implemented' +
                                  ' by a sub-class implementing a MRC.')

    def predict_proba(self, X):
        '''
        Abstract function for sub-classes implementing
        the different MRCs.

        Computes conditional probabilities corresponding
        to each class for the given unlabeled instances.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Testing instances for which
            the prediction probabilities are calculated for each class.

        Returns
        -------
        hy_x : `array`-like of shape (`n_samples`, `n_classes`)
            Conditional probabilities (:math:`p(y|x)`)
            corresponding to each class.
        '''

        # Variants of MRCs inheriting from this class
        # implement this function to compute the conditional
        # probabilities using the classifier obtained from minimax risk

        raise NotImplementedError('BaseMRC is not an implemented classifier.' +
                                  ' It is base class for different MRCs.' +
                                  ' This functions needs to be implemented' +
                                  ' by a sub-class implementing a MRC.')

    def predict(self, X):
        '''
        Predicts classes for new instances using a fitted model.

        Returns the predicted classes for the given instances in `X`
        using the probabilities given by the function `predict_proba`.

        Parameters
        ----------
        X : `array`-like of shape (`n_samples`, `n_dimensions`)
            Test instances for which the labels are to be predicted
            by the MRC model.

        Returns
        -------
        y_pred : `array`-like of shape (`n_samples`)
            Predicted labels corresponding to the given instances.

        '''

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")

        # Get the prediction probabilities for the classifier
        proba = self.predict_proba(X)

        if self.deterministic:
            y_pred = np.argmax(proba, axis=1)
        else:
            y_pred = [np.random.choice(self.n_classes, size=1, p=pc)[0]
                      for pc in proba]

        # Check if the labels were provided for fitting
        # (labels might be omitted if fitting is done through minimax_risk)
        # Otherwise return the default labels i.e., from 0 to n_classes-1.
        if hasattr(self, 'classes_'):
            y_pred = np.asarray([self.classes_[label] for label in y_pred])

        return y_pred

