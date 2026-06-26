# SPARE module to train a misc biomarker using undefined set

"""
SPARE-CL Pipeline Module

This module contains functions for training and inference of SPARE-CL models.
"""

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from ..data_analysis import (
	report_classification_metrics
)

from ..util import (
    get_hyperparameter_tuning
)

from ..svm import (
    get_svm_hyperparameter_grids
)

# Accepts dataframe and target_column as input along with other parameters to perform an svc training
def train_svc_model(
    X,
    y,
    kernel: str = 'linear', # linear_fast, linear, rbf, poly, sigmoid 
    tune_hyperparameters: bool = False,
    cv_fold: int = 5,
    class_balancing: bool = True,
    get_cv_scores: bool = True,
    train_whole_set: bool = True,
    random_state: int = 42, # for replication
    verbose: int = 1,
    **svc_params
    ):
    # Items to return
    model = None
    grid_search = None
    cv_results = None
    best_cv_model = None
    best_cv_score = 0
    
    # Initialize base parameters
    if kernel == 'linear_fast':
        print(f"Training model with LinearSVC...")
        base_params = {'fit_intercept':True,
                       'random_state': random_state,
                       'verbose' : verbose > 1,
                       'max_iter' : 1000000
                       }
    else:
        print(f"Training model with default SVC with {kernel} kernel...")
        base_params = {'kernel': kernel,
                       'probability':True, 
                       'random_state': random_state,
                       'verbose' : verbose > 1}
    
    # Overwrite base parameters with svc_params
    base_params.update(svc_params)
    
    # Enable class_weight='balanced' if class_balancing parameter is passed and True
    if class_balancing:
        base_params.update({'class_weight':'balanced'})
    
    # Perform hyperparameter tuning when asked
    hyperparameter_tuning={}
    if tune_hyperparameters:
        print(f"Hyperparameter selection initated...")
        param_grids = get_svm_hyperparameter_grids()['classification'][kernel]
             
        # Create base model
        if kernel == 'linear_fast':
            base_model = LinearSVC(**base_params)
        else:
            base_model = SVC(**base_params)
    
        # Perform grid search with 5-fold CV
        cv = RepeatedStratifiedKFold(n_splits=cv_fold,
                                     n_repeats=1, 
                                     random_state=random_state)
        
        grid_search = GridSearchCV(
            base_model,
            param_grids,
            cv=cv,
            scoring='average_precision' if class_balancing == True else 'balanced_accuracy',
            n_jobs=-1,
            verbose=verbose
        )
        
        grid_search.fit(X, y)
    
        # Get best parameters and CV score & Update the svc_params
        # cv_score = grid_search.best_score_
        base_params.update(grid_search.best_params_)

        print(f"Best parameters: {base_params}")
        print(f"Best CV {grid_search.scorer_}: {grid_search.best_score_:.3f}")

        hyperparameter_tuning = get_hyperparameter_tuning(grid_search, base_params, param_grids)

    else:
        print(f"Hyperparameter selection skipped...")
        # Use default parameters
        svc_params.setdefault('random_state', random_state)

    # Perform another CV using the best parameter if get_cv_score parameter is True
    if get_cv_scores:
        repeat=3
        print(f"Initiating {repeat} repeated {cv_fold}-fold CV")

        cv = RepeatedStratifiedKFold(n_splits=cv_fold, 
                                     n_repeats=repeat, 
                                     random_state=random_state)

        # Define the schema
        cv_results = dict.fromkeys(["Repeat_%d"%r for r in range(repeat)],
                                   dict.fromkeys(["Fold_%d" % i for i in range(cv_fold)]))

        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            cv_result={} # model, 

            rep_num=str(i//cv.cvargs['n_splits'])
            fold_num=str(i%cv.cvargs['n_splits'])

            print(f"CV iteration {i} (Repeat: {rep_num} Fold: {fold_num})")

            # CV tr/ts sets
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            # Train model with current parameters
            model=None
            if kernel == 'linear_fast':
                model = LinearSVC(**base_params)
            else:
                model = SVC(**base_params)
            model.fit(X_train, y_train.values)

            cv_result['model']=model
            # Predict
            y_pred = model.predict(X_test)
            # Get decision function
            mdf = model.decision_function(X_test)

            # Archieve the testing outcomes
            df_cv_result_per_fold = pd.DataFrame()
            df_cv_result_per_fold['test_reference'] = y_test
            df_cv_result_per_fold['test_prediction'] = y_pred
            df_cv_result_per_fold['test_decision_function'] = mdf
            df_cv_result_per_fold['fold'] = int(fold_num)

            cv_result['cv_validation']=df_cv_result_per_fold
            
            # Get validation metrics
            cv_metric = report_classification_metrics(y_test, y_pred, mdf)
            # print the cv metric
            print(f"CV Validation: {cv_metric}")
            # save
            cv_result['cv_scores']=cv_metric
            

            cv_results[f"Repeat_{rep_num}"][f"Fold_{fold_num}"]=cv_result

            # # Update the best performing model based off of ROC-AUC
            # if 'ROC-AUC' in cv_metric.keys():
            #     if cv_metric['ROC-AUC'] > best_cv_score:
            #         best_cv_model = model
            #         best_cv_score = cv_metric['ROC-AUC']
            # elif 'Accuracy' in cv_metric.keys():
            #     if cv_metric['Accuracy'] > best_cv_score:
            #         best_cv_model = model
            #         best_cv_score = cv_metric['Accuracy']
            

    # Train model using the best parameter and whole set
    if train_whole_set:
        print("Training the wholeset.")
        if kernel == 'linear_fast':
            model = LinearSVC(**base_params)
        else:
            model = SVC(**base_params)
        model.fit(X, y)
    
    else:
        if tune_hyperparameters:
            model = grid_search.best_estimator_
        elif get_cv_scores:
            model = best_cv_model
    
    # Return model and the CV scores
    return model, hyperparameter_tuning, cv_results

