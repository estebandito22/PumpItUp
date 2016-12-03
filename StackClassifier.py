import numpy as np
import math
from sklearn.cross_validation import KFold
import itertools
from sklearn.metrics import accuracy_score

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
import six
from joblib import Parallel, delayed
from sklearn.utils.validation import has_fit_parameter, check_is_fitted

def _parallel_fit_estimator(estimator, X, y):
    """Private function used to fit an estimator within a job."""
    estimator.fit(X, y)
    return estimator

class StackClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Stacked classifier for unfitted estimators.
    .. versionadded:: 0.1
    Parameters
    ----------
    base_estimators : list of (string, estimator) tuples
        Invoking the ``fit`` method on the ``StackClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        `self.base_estimators_`.
    stack_estimator : sklearn estimator
        Invoking the ``fit`` method on the ``StackClassifier`` will fit a clone
        of the original stack_estimator that will be stored in the class attribute
        `self.stack_estimator_`.
    n_folds : int, optional (default = 2)
        The number of folds to use for base estimators during ``fit``.
    ovr_fold : bool, optional (default = False)
        If True, then the base estimators will fit an additional fold using
        all samples.
    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for ``fit``.
        If -1, then the number of jobs is set to the number of cores.
    
    Attributes
    ----------
    base_estimators_ : list of classifiers
        The collection of fitted base-estimators.
    base_predictions_ : list of predictions
        The collection of predictions from fitted base-estimators.
    stack_estimator_ : classifier
        The fitted stacker estimator.
    predictions_ : list of predictions
        The predictions from the fitted stacker estimator.
    classes_ : array-like, shape = [n_predictions]
        The classes labels.
    Examples
    --------
    
    """
    
    def __init__(self, base_estimators, stack_estimator, n_folds=2, ovr_fold = False, n_jobs=1):       
        self.base_estimators=base_estimators
        self.named_base_estimators = dict(base_estimators)
        self.stack_estimator=stack_estimator
        self.n_folds = n_folds
        self.ovr_fold = ovr_fold
        self.n_jobs = n_jobs
    
    def fit(self, X, y=None):
        """ Fit the estimators.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : object
        """
            
        if self.base_estimators is None or len(self.base_estimators) == 0:
            raise AttributeError('Invalid `base_estimators` attribute, `base_estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')
            
        if self.stack_estimator is None:
            raise AttributeError('Invalid `stack_estimator` attribute, `stack_estimator`'
                                 ' should be an estimator')
            
        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        self.base_estimators_ = []
        self.base_predictions_ = []
        self.base_scores_ = []
        self.stack_estimator_ = None
        
        transformed_y = self.le_.transform(y)
        
        stack_sets = KFold(n=X.shape[0], n_folds=self.n_folds)
        base_predictions = []
        
        if self.ovr_fold == True:
            self.base_estimators_ += [Parallel(n_jobs=self.n_jobs)(
                    delayed(_parallel_fit_estimator)(clone(clf), X, transformed_y)
                        for _, clf in self.base_estimators)]        
            base_predictions += [[clf.predict_proba(X) for clf in self.base_estimators_[0]]]
            self.base_scores_ += [[accuracy_score(clf.predict(X), y) for clf in self.base_estimators_[0]]]
        
        for idx, (train_index, test_index) in enumerate(stack_sets):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = transformed_y[train_index], transformed_y[test_index]
            self.base_estimators_ += [Parallel(n_jobs=self.n_jobs)(
                    delayed(_parallel_fit_estimator)(clone(clf), X_train, y_train)
                        for _, clf in self.base_estimators)]
            if self.ovr_fold == True:
                base_predictions += [[clf.predict_proba(X_test) for clf in self.base_estimators_[idx+1]]]
                self.base_scores_ += [[accuracy_score(clf.predict(X), y) for clf in self.base_estimators_[idx+1]]]
            else:
                base_predictions += [[clf.predict_proba(X_test) for clf in self.base_estimators_[idx]]]
                self.base_scores_ += [[accuracy_score(clf.predict(X), y) for clf in self.base_estimators_[idx]]]
        
        if self.ovr_fold == True:
            self.base_predictions_ = np.hstack((np.hstack(base_predictions[0]),np.vstack([np.hstack(z) for z in base_predictions[1:]])))
        else:
            self.base_predictions_ = np.vstack([np.hstack(z) for z in base_predictions])
        
        self.stack_estimator_ = clone(self.stack_estimator).fit(self.base_predictions_, transformed_y)
        
        return self
    
    def predict(self, X):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        predictions_ : array-like, shape = [n_samples]
            Predicted class labels.
        """       
        
        base_predictions = self._blend_probas(X)
        
        self.predictions_ = self.le_.inverse_transform(self.stack_estimator_.predict(base_predictions))
        
        return self.predictions_
    
    def _blend_probas(self, X):
        """Return collected predicted class probabilities from fitted base_estimators_"""

        base_predictions = []
        blend_predictions_j = {}
        blended_predictions = []
        
        if self.ovr_fold == True:
            base_predictions += [[clf.predict_proba(X) for clf in self.base_estimators_[0]]]
            fold_idx_range = range(1,self.n_folds+1)
        else:
            fold_idx_range = range(1,self.n_folds)
        
        for estimator_idx in range(len(self.base_estimators)):
            for fold_idx in fold_idx_range:
                if estimator_idx in blend_predictions_j:
                    blend_predictions_j[estimator_idx] += [[self.base_estimators_[fold_idx][estimator_idx].predict_proba(X)]]
                else:
                    blend_predictions_j[estimator_idx] = []
                    blend_predictions_j[estimator_idx] += [[self.base_estimators_[fold_idx][estimator_idx].predict_proba(X)]]
        
        for estimator_idx in range(len(self.base_estimators)):
            zipped_predictions = list(zip(*blend_predictions_j[estimator_idx]))
            blended_predictions += [sum(y) / len(y) for y in zipped_predictions]
        
        base_predictions += [blended_predictions]
        
        if self.ovr_fold == True:
            base_predictions = np.hstack((np.hstack(base_predictions[0]),np.hstack(base_predictions[1])))
        else:
            base_predictions = np.hstack(base_predictions[0])
        
        return base_predictions
            
    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support"""
        
        if not deep:
            return super(StackClassifier, self).get_params(deep=False)
        else:
            out = super(StackClassifier, self).get_params(deep=True)
            out.update(self.base_estimators.copy())
            return out
        
    def score(self, X, y):
        """Return accuracy score of predictions"""
        return accuracy_score(self.predict(X),y)
