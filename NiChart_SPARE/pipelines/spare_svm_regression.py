# SPARE module to train a misc biomarker using undefined set

"""
SPARE-CL Pipeline Module

This module contains functions for training and inference of SPARE-CL models.
"""

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR, SVR
from sklearn.model_selection import GridSearchCV, RepeatedKFold

from ..data_analysis import (
	report_regression_metrics
)

from ..util import (
    get_hyperparameter_tuning
)

from ..svm import (
    get_svm_hyperparameter_grids,
    correct_linearsvr_bias
)

# Accepts dataframe and target_column as input along with other parameters to perform an svc training
def train_svr_model(
    X,
    y,
    kernel: str = 'linear', # linear_fast, linear, rbf, poly, sigmoid 
    tune_hyperparameters: bool = False,
    cv_fold: int = 5,
    get_cv_scores: bool = True,
    train_whole_set: bool = True,
    random_state: int = 42, # for replication
    verbose: int = 1,
    **svc_params
    ):
    # Items to return
    model = None
    grid_search = None
    cv_scores = None
    best_cv_model = None
    best_cv_score = 0
    
    # Initialize base parameters
    if kernel == 'linear_fast':
        print(f"Training model with LinearSVR...")
        base_params = {'max_iter': 100000,
                       'verbose' : verbose > 1}
    else:
        print(f"Training model with default SVR with {kernel} kernel...")
        base_params = {'kernel': kernel, 
                       #'random_state': random_state,
                       'verbose' : verbose > 1}
    # Overwrite base parameters with svc_params
    base_params.update(svc_params)
        
    # Perform hyperparameter tuning when asked
    hyperparameter_tuning={}
    if tune_hyperparameters:
        print(f"Hyperparameter selection initated...")
        param_grids = get_svm_hyperparameter_grids()['regression'][kernel]
             
        # Create base model
        if kernel == 'linear_fast':
            base_model = LinearSVR(**base_params)
        else:
            base_model = SVR(**base_params)
    
        # Perform grid search with 5-fold CV
        cv = RepeatedKFold(n_splits=cv_fold,
                           n_repeats=1, 
                           random_state=random_state)
        
        grid_search = GridSearchCV(
            base_model,
            param_grids,
            cv=cv,
            scoring='r2',
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
        # # Use default parameters
        # svc_params.setdefault('random_state', random_state)

    # Perform another CV using the best parameter if get_cv_score parameter is True
    cv_scores = {}
    if get_cv_scores:
        print(f"Initiating {cv_fold}-fold CV")
        
        cv = RepeatedKFold(n_splits=cv_fold, 
                           n_repeats=1, 
                           random_state=random_state)
        
        for i, (train_index, test_index) in enumerate(cv.split(X, y)):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            
            # Train model with current parameters
            if kernel == 'linear_fast':
                model = LinearSVR(**base_params)
                print("Correcting bias")
                model = correct_linearsvr_bias(model,X,y)
                model.fit(X_train, y_train)
            else:
                model = SVR(**base_params)
                model.fit(X_train, y_train)
            
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            # Get validation metrics
            cv_metric = report_regression_metrics(y_test, y_pred)
            print(f"Iteration {i+1} Repeat {(i+1)//cv_fold} Fold {i % cv.n_repeats} metrics: {cv_metric}")
            # Save the scores
            cv_scores["Fold_%d" % (i % cv.n_repeats)] = cv_metric
            # Update the best performing model based off of ROC-AUC
            if cv_metric['MSE'] > best_cv_score:
                best_cv_model = model
                best_cv_score = cv_metric['MSE']
            

    # Train model using the best parameter and whole set
    if train_whole_set:
        print("Training the wholeset.")
        if kernel == 'linear_fast':
            model = LinearSVR(**base_params)
            model.fit(X, y)
            print("Correcting bias")
            model = correct_linearsvr_bias(model,X,y)
        else:
            model = SVR(**base_params)
            model.fit(X, y)
    
    else:
        if tune_hyperparameters:
            model = grid_search.best_estimator_
        elif get_cv_scores:
            model = best_cv_model

    # Return model and the CV scores
    return model, hyperparameter_tuning, cv_scores

# def train_svr_model(
#     X,
#     y,
#     kernel: str = 'linear',
#     tune_hyperparameters: bool = False,
#     cv_fold: int = 5,
#     get_cv_scores: bool = True,
#     train_whole_set: bool = True,
#     random_state: int = 42,
#     **svc_params
# ):
#     """Train an SVR model to predict the target column from a dataframe."""
    
#     # Initialize base parameters
#     base_params = {'kernel': kernel, 'random_state': random_state}
#     base_params.update(svc_params)  # overwrite base_params
        
#     # Train SVR model
#     if tune_hyperparameters:
#         print(f"Hyperparameter selection initated...")
#         param_grids = get_svm_hyperparameter_grids()['regression']
        
#         # Get parameter grid for the specified kernel
#         param_grid = param_grids.get(kernel, {})
#         if param_grid:
#             # Remove any parameters that are already set in svc_params
#             for param in list(param_grid.keys()):
#                 if param in svc_params:
#                     del param_grid[param]
        
#         # Create base model
#         base_model = SVR(**base_params)
        
#         # Perform grid search with 5-fold CV
#         cv = RepeatedKFold(n_splits=cv_fold)
        
#         grid_search = GridSearchCV(
#             base_model,
#             param_grid,
#             cv=cv,
#             scoring='r2',
#             n_jobs=-1,
#             verbose=3
#         )
        
#         grid_search.fit(X, y)
#         print(f"Hyperparameter selection with {cv_fold} fold CV completed.")
        
#         # Get best parameters and CV score & Update the svc_params
#         cv_score = grid_search.best_score_
#         svr_params = grid_search.best_params_
        
#         print(f"Best parameters: {svr_params}")
#         print(f"CV balanced accuracy: {cv_score:.3f}")

#     else:
#         print(f"Hyperparameter selection skipped...")
#         # Use default parameters
#         svc_params.setdefault('random_state', random_state)
#         cv_score = None
    
#     if get_cv_scores:
#         print(f"Initiating {cv_fold}-fold CV")
#         cv_scores = []
#         cv = RepeatedKFold(n_splits=cv_fold, n_repeats=10, random_state=random_state)
        
#         for i, (train_index, test_index) in enumerate(cv.split(X)):
#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = y[train_index], y[test_index]
            
#             # Train model with current parameters
#             model = SVR(kernel=kernel, **svc_params)
#             model.fit(X_train, y_train)
            
#             # Predict and calculate accuracy
#             y_pred = model.predict(X_test)
#             score = r2_score(y_test, y_pred)
#             cv_scores.append(score)
            
#             print(f"Fold {i+1}: R2 = {score:.3f}")
        
#         print(f"Mean CV R2: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

#     # Train model using the best parameter and whole set
#     if train_whole_set:
#         model = SVR(kernel=kernel, **svc_params)
#         model.fit(X, y)
#         return model
    
#     else:
#         if tune_hyperparameters:
#             return grid_search.best_estimator_
#         else:
#             return None