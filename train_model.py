############################# Imports #########################################

import pandas as pd
import numpy as np
import scipy as sp
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import KFold
from sklearn.grid_search import RandomizedSearchCV
import datetime
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from StackClassifier import *

######################### Training and Test Data ##############################

train_values = pd.read_csv(
    """https://s3.amazonaws.com/drivendata/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv""")
train_labels = pd.read_csv(
    """https://s3.amazonaws.com/drivendata/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv""")
test_values = pd.read_csv(
    """https://s3.amazonaws.com/drivendata/data/7/public/702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv""")

########################## Derive Features ####################################

derived_features = lambda x: pd.Series({
    'datetime_recorded': datetime.datetime.strptime(x['date_recorded'], "%Y-%m-%d"),
    'date_recorded_num': (datetime.datetime.strptime(x['date_recorded'], "%Y-%m-%d") - datetime.datetime(1950, 1, 1)).days,
    'construction_year_num': x['construction_year'] - 1950 if x['construction_year'] > 0 else np.nan,
    'dayofyear_recorded': datetime.datetime.strptime(x['date_recorded'], "%Y-%m-%d").timetuple().tm_yday,
    'dayofweek_recorded': datetime.datetime.strptime(x['date_recorded'], "%Y-%m-%d").weekday(),
    'quarterofyear_recorded': ((datetime.datetime.strptime(x['date_recorded'], "%Y-%m-%d")).month - 1) // 3 + 1
})
train_values = train_values.join(train_values.apply(derived_features, axis=1))
test_values = test_values.join(test_values.apply(derived_features, axis=1))

######################### Remove Features #####################################

train_values = train_values.drop(['id', 'region_code', 'district_code', 'num_private', 'recorded_by', 'wpt_name',
                                  'subvillage', 'lga', 'ward', 'scheme_name', 'date_recorded', 'datetime_recorded', 'construction_year'], axis=1)
test_values = test_values.drop(['region_code', 'district_code', 'num_private', 'recorded_by', 'wpt_name',
                                'subvillage', 'lga', 'ward', 'scheme_name', 'date_recorded', 'datetime_recorded', 'construction_year'], axis=1)

#################### Transformers for Pipeline ###############################


class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, data_types=list, sparse=False):
        self.data_types = data_types
        self.sparse = sparse

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        if self.sparse == True:
            return sp.sparse.csr_matrix(data_dict.select_dtypes(include=self.data_types))
        else:
            return data_dict.select_dtypes(include=self.data_types)


class PrepForVect(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict.fillna('NA').T.to_dict().values()


########################## Pipeline for Model ################################

def build_pipeline(estimator):
    pipeline = Pipeline([
        # Use FeatureUnion to combine categorical and numerical features
        ('union', FeatureUnion(
            transformer_list=[
                # Pipeline for categorical data
                ('categorical', Pipeline([
                        ('selector', ItemSelector(
                            data_types=['bool', 'object'])),
                        ('prepare', PrepForVect()),
                        ('vect', DictVectorizer(sparse=True)),
                        ])),
                # Pipeline for numerical data
                ('numerical', Pipeline([
                    ('selector', ItemSelector(data_types=[
                        'int64', 'float64'], sparse=True)),
                    ('impute', Imputer()),
                    # ('normalize', Normalizer())
                ]))
            ]
        )),
        # Select Features
        # ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
        ('classification', estimator)
    ])

    return pipeline

############################ Grid Search ######################################

model_params = {
    # 'union__numerical__normalize__norm':['l1','l2','max'],
    # 'feature_selection__estimator__C':sp.stats.uniform(0.165,0.164),
    'classification__AdaBoost__base_estimator__max_depth': sp.stats.randint(2, 20),
    'classification__AdaBoost__base_estimator__min_samples_leaf': sp.stats.randint(2, 20),
    'classification__AdaBoost__base_estimator__max_features': sp.stats.uniform(0.5, 0.4),
    'classification__AdaBoost__algorithm': ['SAMME', 'SAMME.R'],
    'classification__AdaBoost__learning_rate': sp.stats.uniform(0.05, 0.049),
    'classification__AdaBoost__base_estimator__criterion': ['gini', 'entropy'],
    'classification__RandomForestClassifier__min_samples_leaf': sp.stats.randint(2, 20),
    'classification__RandomForestClassifier__max_depth': sp.stats.randint(2, 20),
    'classification__RandomForestClassifier__max_features': sp.stats.uniform(0.5, 0.4),
    'classification__RandomForestClassifier__criterion': ['gini', 'entropy'],
    'classification__MLPClassifier__hidden_layer_sizes': [(10, 2), (9, 2), (8, 2), (7, 2), (6, 2)],
    'classification__MLPClassifier__learning_rate_init': sp.stats.uniform(0.05, 0.049),
    'classification__MLPClassifier__alpha': sp.stats.uniform(0.05, 0.049),
    'classification__MLPClassifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
    'classification__ExtraTreesClassifier__min_samples_leaf': sp.stats.randint(2, 20),
    'classification__ExtraTreesClassifier__max_depth': sp.stats.randint(2, 20),
    'classification__ExtraTreesClassifier__max_features': sp.stats.uniform(0.5, 0.4),
    'classification__ExtraTreesClassifier__criterion': ['gini', 'entropy'],
    'classification__SVC__C': sp.stats.uniform(0.165, 0.164),
    'classification__SVC__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'classification__LogisticRegression__C': sp.stats.uniform(0.165, 0.164),
    'classification__LogisticRegression__class_weight': [None, 'balanced'],
    'classification__LogisticRegression__multi_class': ['ovr', 'multinomial'],
    'classification__stack_estimator__subsample': sp.stats.uniform(0.5, 0.3),
    'classification__stack_estimator__min_samples_leaf': sp.stats.randint(2, 20),
    'classification__stack_estimator__max_depth': sp.stats.randint(2, 20),
    'classification__stack_estimator__max_features': sp.stats.uniform(0.5, 0.4),
    'classification__stack_estimator__loss': ['deviance'],
    'classification__stack_estimator__learning_rate': sp.stats.uniform(0.05, 0.049)
}

base_estimator_list = [
    ['AdaBoost', AdaBoostClassifier(
        DecisionTreeClassifier(), n_estimators=1000)],
    ['RandomForestClassifier', RandomForestClassifier(n_estimators=1000)],
    ['MLPClassifier', MLPClassifier(max_iter=1000, early_stopping=True)],
    ['ExtraTreesClassifier', ExtraTreesClassifier(n_estimators=1000)],
    ['SVC', SVC(max_iter=1000, probability=True)],
    ['LogisticRegression', LogisticRegression(
        max_iter=1000, solver='lbfgs', penalty='l2')]
]

sclf = StackClassifier(base_estimators=base_estimator_list, stack_estimator=GradientBoostingClassifier(
    n_estimators=500), n_folds=5, ovr_fold=False)
pipeline = build_pipeline(sclf)
kfold = KFold(train_values.shape[0], n_folds=5)
grid_search = RandomizedSearchCV(
    pipeline, param_distributions=model_params, cv=kfold, verbose=10, n_iter=5)
grid_search.fit(train_values, train_labels['status_group'].ravel())

############################ Submission #######################################

predictions = grid_search.predict(test_values.drop('id', axis=1))
submission = pd.DataFrame({'id': test_values['id'].values,
                           'status_group': predictions}, columns=['id', 'status_group'])
