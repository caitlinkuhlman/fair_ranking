"""Weight Boosting
This module contains weight boosting estimators for both classification and
regression.
The module structure is the following:
- The ``BaseWeightBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ from each other in the loss function that is optimized.
- ``AdaBoostClassifier`` implements adaptive boosting (AdaBoost-SAMME) for
  classification problems.
- ``AdaBoostRegressor`` implements adaptive boosting (AdaBoost.R2) for
  regression problems.
"""

# Authors: Noel Dawe <noel@dawe.me>
#          Gilles Louppe <g.louppe@gmail.com>
#          Hamzeh Alsalhi <ha258@cornell.edu>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod

import numpy as np
import math
from numpy.core.umath_tests import inner1d
from sklearn.ensemble.base import BaseEnsemble
from sklearn.base import ClassifierMixin, RegressorMixin, is_regressor, is_classifier
from sklearn.externals import six
from sklearn.externals.six.moves import zip
from sklearn.externals.six.moves import xrange as range
from sklearn.ensemble import *
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree.tree import BaseDecisionTree
from sklearn.tree._tree import DTYPE
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.utils import extmath
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.utils.validation import has_fit_parameter, check_is_fitted

class FairBoostRegressor(weight_boosting.BaseWeightBoosting, RegressorMixin):
    """An AdaBoost regressor.
    An AdaBoost [1] regressor is a meta-estimator that begins by fitting a
    regressor on the original dataset and then fits additional copies of the
    regressor on the same dataset but where the weights of instances are
    adjusted according to the error of the current prediction. As such,
    subsequent regressors focus more on difficult cases.
    This class implements the algorithm known as AdaBoost.R2 [2].
    Read more in the :ref:`User Guide <adaboost>`.
    Parameters
    ----------
    base_estimator : object, optional (default=DecisionTreeRegressor)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required.
    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each regressor by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    loss : {'linear', 'square', 'exponential'}, optional (default='linear')
        The loss function to use when updating the weights after each
        boosting iteration.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.
    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.
    estimator_errors_ : array of floats
        Regression error for each estimator in the boosted ensemble.
    feature_importances_ : array of shape = [n_features]
        The feature importances if supported by the ``base_estimator``.
    See also
    --------
    AdaBoostClassifier, GradientBoostingRegressor, DecisionTreeRegressor
    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.
    .. [2] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.
    """
    
    
    #we pass group matrices to method: g1,g2
    #g1:[k][n] matrix - each row is indices of points in this bin
    #g2:[k][m] matrix - numberof bins is the same but number of instances in each bin may vary between groups
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 loss='linear',
                 random_state=None):
        

        super(FairBoostRegressor, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.loss = loss
        self.random_state = random_state
        
    def setGroups(self, g):
        self.g0 = [i for i,x in enumerate(g) if x==0]
        self.g1 = [i for i,x in enumerate(g) if x==1]

    def fit(self, X, y, sample_weight=None):
        """Build a boosted regressor from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        y : array-like of shape = [n_samples]
            The target values (real numbers).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check loss
        if self.loss not in ('linear', 'square', 'exponential'):
            raise ValueError(
                "loss must be 'linear', 'square', or 'exponential'")

        # Fit
        return super(FairBoostRegressor, self).fit(X, y)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(FairBoostRegressor, self)._validate_estimator(
            default=DecisionTreeRegressor(max_depth=3))
        

    def recomputeBins(self, y_predict, g0, g1, nbins):
        #g0 has indexes of objects in X
        # get indexes of sorted predictions for group
        sorted0 = np.argsort([y_predict[x] for x in g0])
        binSize=int(np.ceil(float(len(g0))/nbins))
        bins0=[]
        b=[]
        i=0
        j=binSize-1
        for n in range(nbins):
            k=int(min(j,len(sorted0)-1))
            bins0.append([g0[x] for x in sorted0[i:k]])
            i+=binSize
            j+=binSize
           
        #g1 has indexes of objects in X
        # get indexes of sorted predictions for group
        sorted1 = np.argsort([y_predict[x] for x in g1])
        binSize=int(np.ceil(float(len(g1))/nbins))
        bins1=[]
        b=[]
        i=0
        j=binSize-1
        for n in range(nbins):
            k=int(min(j,len(sorted0)-1))
            bins1.append([g1[x] for x in sorted1[i:k]])
            i+=binSize
            j+=binSize
        return bins0,bins1
        
    #########################################################################################
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Perform a single boost according to the AdaBoost.R2 algorithm and return the updated sample weights.
        Parameters
        ----------
        iboost : int
        The index of the current boost iteration.
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
        The training input samples. Sparse matrix can be CSC, CSR, COO,
        DOK, or LIL. DOK and LIL are converted to CSR.
        y : array-like of shape = [n_samples]
        The target values (class labels in classification, real numbers in
        regression).
        sample_weight : array-like of shape = [n_samples]
        The current sample weights.
        random_state : numpy.RandomState
        The current random number generator
        Returns
        -------
        sample_weight : array-like of shape = [n_samples] or None
        The reweighted sample weights.
        If None then boosting has terminated early.
        estimator_weight : float
        The weight for the current boost.
        If None then boosting has terminated early.
        estimator_error : float
        The regression error for the current boost.
        If None then boosting has terminated early.
        """
        estimator = self._make_estimator(random_state=random_state)

        # Weighted sampling of the training set with replacement
        # For NumPy >= 1.7.0 use np.random.choice
        
        cdf = np.cumsum(sample_weight, axis=None, dtype=np.float64)
        
        cdf /= cdf[-1]
        uniform_samples = random_state.random_sample(X.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)
        
        #print()
        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        y_predict = estimator.predict(X)
        #print ("Iteration " + str(iboost) + ": "+str(mean_squared_error(y, y_predict)))
        
        error_vect = np.abs(y_predict - y)
        
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        ###########  REPLACE ERROR VECT WITH OUR OWN   ########
        #         error_vect = np.abs(y_predict - y)
        
        ################ compute bin-wise group error #####################
        #print("before")
        #print(error_vect)
        nbins=10
        bins0,bins1  = self.recomputeBins(y_predict, self.g0, self.g1, nbins)
        # sum the error for items in bin for each group
        e0=[np.sum([error_vect[i] for i in b]) for b in bins0]
        e1=[np.sum([error_vect[i] for i in b]) for b in bins1]
    
        bin_error = np.subtract(e0, e1)
        fair_error = np.mean(np.abs(bin_error))
        #print ("errors: ", fair_error)
        ################# Update weights based on binned error ############

        # weight proportional to difference between the error in each group in the bin. 
        # but could be based off of the individual error for each term
        # or some combination
        for i,e in enumerate(bin_error):
            if e < 0:
                for x in bins0[i]:
                   error_vect[x]=np.abs(bin_error[i])
                for x in bins1[i]:
                    error_vect[x]=0
            else:
                for x in bins1[i]:
                    error_vect[x]=np.abs(bin_error[i])
                for x in bins0[i]:
                    error_vect[x]=0

    ################ continue #########################################
        
        #print("after")
        #print(error_vect)
        # Calculate the average loss
        estimator_error = (sample_weight * error_vect).sum()

        #if estimator_error <= 0:
        if fair_error <= 0:
            # Stop if fit is perfect
            #print("Fit is perfect!")
            return sample_weight, 1., 0.

        #elif estimator_error >= 0.5:
        elif fair_error >= 10:
            # Discard current estimator only if it isn't the only one
            #print("error too high!")
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None
            
        beta = estimator_error / (1. - estimator_error)

        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1. / beta)
        #if this isnt the last iteration,
        if not iboost == self.n_estimators - 1:
            sample_weight *= np.power(
                beta,
                (1. - error_vect) * self.learning_rate)

        return sample_weight, estimator_weight, estimator_error
    
    #########################################################################################

    def _get_median_predict(self, X, limit):
        # Evaluate predictions of all estimators
        predictions = np.array([
            est.predict(X) for est in self.estimators_[:limit]]).T
        
        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)
        
        # Find index of median prediction for each sample
        weight_cdf = np.cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)
        
        median_estimators = sorted_idx[np.arange(X.shape[0]), median_idx]
        
        # Return median predictions
        return predictions[np.arange(X.shape[0]), median_estimators]
        
    def predict(self, X):
        """Predict regression value for X.
        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
        The training input samples. Sparse matrix can be CSC, CSR, COO,
        DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        y : array of shape = [n_samples]
        The predicted regression values.
        """
        check_is_fitted(self, "estimator_weights_")
        X = self._validate_X_predict(X)
        
        return self._get_median_predict(X, len(self.estimators_))

    def staged_predict(self, X):
        """Return staged predictions for X.
        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.
        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
        The training input samples. Sparse matrix can be CSC, CSR, COO,
        DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        y : generator of array, shape = [n_samples]
        The predicted regression values.
        """
        check_is_fitted(self, "estimator_weights_")
        X = self._validate_X_predict(X)
    
        for i, _ in enumerate(self.estimators_, 1):
            yield self._get_median_predict(X, limit=i)
            
