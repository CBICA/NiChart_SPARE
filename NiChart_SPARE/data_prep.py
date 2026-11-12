# Functions for data preprocessing for training and inference.

# FEATURES ADDED

# Data selection for 95% confidence interval
# SVR Correction

import os
import pandas as pd
import numpy as np
# import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.utils import column_or_1d


# ############# DATA CHECK ##############
def validate_dataframe(df: pd.DataFrame, target_column: str) -> None:
    """Validate input dataframe and target column."""
    if df.empty:
        raise ValueError("Dataframe is empty")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")


# ############# DATA PREPROCESSING ##############
def load_csv_data(file_path, drop_columns=None):
    """Load CSV data and return dataframe, optionally dropping specified columns"""
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded data: {len(df)} samples, {len(df.columns)} columns")
    if drop_columns:
        for col in drop_columns:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"Dropped column: {col}")
            else:
                print(f"Warning: Column to drop not found: {col}")
    return df


# Encode features
def encode_feature_df(df: pd.DataFrame):
    if (df.dtypes == 'object').any():
        columns_in_order = df.columns
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        print(f"Encoding the following columns: {object_cols}")
        df_oc = df[object_cols]
        encoders = {}
        for oc in object_cols:
            encoder = LabelEncoder()
            df_oc.loc[:, oc] = encoder.fit_transform(df_oc[oc].to_numpy())
            
            encoders[oc] = encoder
        #df_encoded = pd.DataFrame(df_oc, columns=object_cols, index=df.index)
        print(df.shape, df_oc.shape)
        return pd.concat([df[df.columns[~df.columns.isin(object_cols)]], df_oc], axis=1)[columns_in_order], encoders
    else:
        return df, None

# Perform ICV correction
def correct_icv(df: pd.DataFrame, icv_col: str = "DL_MUSE_Volume_702", roi_col_keyword: str = "DL_MUSE_Volume_"):
    roi_columns = [col for col in df.columns if roi_col_keyword in col and col != roi_col_keyword]
    if icv_col in df.columns and len(roi_columns) > 0:
        non_roi_columns = [col for col in df.columns if col not in roi_columns and col != roi_col_keyword]
        df_roi_icv_corrected = df[roi_columns] / df[icv_col]
        if non_roi_columns != []:
            df_roi_icv_corrected = pd.concat([df[non_roi_columns],df_roi_icv_corrected],axis=1)
        return df_roi_icv_corrected[[c for c in df.columns if c != icv_col]]
    else:
        raise("Failed to perform ICV correction as ROI column or ICV column does not exist")

# Scale features
def scale_feature_df(df: pd.DataFrame):
    X = df.copy()
    scalers={}
    for c in X.columns:
        if c not in ['Sex','Sex_M']:
            scaler = StandardScaler()
            scaler.fit(X[c].to_numpy().reshape(-1,1))
            scalers[c] = scaler
            X[c] = scaler.transform(X[c].to_numpy().reshape(-1,1))
        else:
            X[c] = X[c].astype(float)
    return X, scalers


# Prepare data for regressor training and testing
# df: dataframe containing only essential columns + target column
def preprocess_regression_data(
    df: pd.DataFrame, 
    target_column: str,
    encode_categorical_features: bool = True,
    scale_features: bool = True,
    # icv_correction: bool = False,
    # icv_column: str = '',
    for_training: bool = True,
    feature_encoder: LabelEncoder = None, # for inference
    feature_scaler: StandardScaler = None, # for inference
):
    X, y = (None, None)
    
    if for_training:
        """Preprocess data for training: handle missing values and encode categorical featurs & targets."""
        feature_encoder, feature_scaler = (None, None)
        df = df.dropna(subset=[target_column]) # Remove rows with missing target values
        
        # Separate features and target
        print(f"Dropped target column: {target_column}")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode feature labels if they're not numeric and encoding is requested
        if encode_categorical_features:
            print(f"Encoding the categorical features.")
            X, feature_encoder = encode_feature_df(X)
        # Scale features if requested
        if scale_features:
            print(f"Scaling the features.")
            X, feature_scaler = scale_feature_df(X)

    else:
        """Preprocess data for inference: handle missing values and encode categorical features."""
        # Check if ground truth is provided in the df, if so, drop it.
        if target_column in df.columns:
            print(f"Dropped target column: {target_column}")
            X = df.drop([target_column],axis=1)
            y = df[target_column]
        else:
            X = df
            y = None
        
        if feature_encoder != None:
            for ec in feature_encoder.keys():
                X[ec] = feature_encoder[ec].transform(X[ec])
        
        if feature_scaler != None:
            for fs in feature_scaler.keys():
                X[fs] = feature_scaler[fs].transform(X[fs].to_numpy().reshape(-1,1))
    
    return X, y, feature_encoder, feature_scaler


# Prepare data for classifier training and testing
# df: dataframe containing only essential columns + target column
def preprocess_classification_data(
    df: pd.DataFrame, 
    target_column: str,
    encode_categorical_features: bool = True,
    scale_features: bool = True,
    # icv_correction: bool = False,
    # icv_column: str = '',
    for_training: bool = True,
    feature_encoder: LabelEncoder = None, # for inference
    feature_scaler: StandardScaler = None, # for inference
):
    X, y = (None, None)
    print("In preprocessing of classification data.")
    if for_training == True:
        """Preprocess data for training: handle missing values and encode categorical featurs & targets."""
        feature_encoder, feature_scaler = (None, None)
        print("Dropping rows with target column NaNs")
        df = df.dropna(subset=[target_column]) # Remove rows with missing target values
        
        # Separate features and target
        print("Separating feature and target columns.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode feature labels if they're not numeric and encoding is requested
        if encode_categorical_features:
            print(f"Encoding the categorical features.")
            X, feature_encoder = encode_feature_df(X)
        # Scale features if requested
        if scale_features:
            print(f"Scaling the features.")
            X, feature_scaler = scale_feature_df(X)

    else:
        """Preprocess data for inference: handle missing values and encode categorical features."""
        # Check if ground truth is provided in the df, if so, drop it.
        print(f"Dropping target column {target_column} if present.")
        if target_column in df.columns:
            X = df.drop([target_column],axis=1)
            y = df[target_column]
        else:
            print("Target column not found in dataframe.")
            X = df
            y = None
        
        if feature_encoder != None:
            print("Running feature encoding.")
            for ec in feature_encoder.keys():
                X[ec] = feature_encoder[ec].fit_transform(X[ec])
        
        if feature_scaler != None:
            print("Running feature scaling.")
            for fs in feature_scaler.keys():
                X[fs] = feature_scaler[fs].transform(X[fs].to_numpy().reshape(-1,1))

    return X, y, feature_encoder, feature_scaler

import warnings

def apply_cvm_residualization(df: pd.DataFrame, 
                              age_col='Age', 
                              sex_col='Sex', 
                              dlicv_col='DL_MUSE_Volume_702'):
    
    df = df.rename(columns={dlicv_col:'DLICV'})

    all_columns = df.columns.tolist()

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df_params = pd.read_csv(os.path.join(current_dir, 'reference', 'covparams_scaler_sparecvms_dl2.csv'))

    # df_params = pd.read_csv("/home/kylebaik/Packages/NiChart_SPARE/NiChart_SPARE/reference/covparams_scaler_sparecvms_dl2.csv")
    # df_params = df_params.rename(columns={'DLICV':dlicv_col})

    df['Age_Original'] = df[age_col]#.copy(deep=True)

    # Centre age on the training mean held in the residualization map, not on
    # the mean of whoever happens to be in this batch.
    #
    # The coefficients below were fitted against mean-centred age, so centring
    # on the batch shifts every prediction by
    # (training_mean - batch_mean) * coefficient: the same scan scores
    # differently depending on which other subjects were submitted with it. On
    # 561 harmonized IXI subjects, re-running the 60 youngest on their own
    # (mean age 23.8) rather than inside the full cohort (mean age 48.6)
    # flipped the label for 1 of 60 on both SPARE-CVM-T2D and
    # SPARE-CVM-HYPERLIPIDEMIA.
    #
    # The map's Scaler_Mean for age is that training mean -- it is what
    # standardizes the age column a few lines down. A map without an age row
    # cannot be centred correctly, and falling back to the batch mean is the
    # bug being fixed here, so say so instead.
    age_params = df_params.loc[df_params['Features'] == age_col, 'Scaler_Mean']
    if len(age_params) != 1:
        raise ValueError(
            f"Residualization map has {len(age_params)} rows for the age column "
            f"'{age_col}'; exactly one is needed to centre age on the training "
            f"mean. Check that reference/covparams_scaler_sparecvms_dl2.csv "
            f"carries an '{age_col}' row."
        )
    meanage = float(age_params.iloc[0])

    df['Mean_centered_age'] = df['Age_Original']-meanage
    df['DLICV_Original'] = df['DLICV'].copy(deep=True)

    # Map Sex column
    if 'Sex_M' not in df.columns:
        df['Sex_M'] = df[sex_col].copy(deep=True)
        if 0 not in df[sex_col].unique() and 'M' in df[sex_col].unique() and 'F' in df[sex_col].unique():
            df['Sex_M'] = df[sex_col].map({'F':0,'M':1})

    rois = df_params.loc[df_params['Features'].str.contains('_Volume_'),'Features']

    features = [age_col, 'DLICV'] + [roi for roi in rois if roi in df.columns]
    confounds = ['Sex_M', 'Mean_centered_age', 'DLICV']
    
    for roi in rois:
        if roi in df.columns:
            df['Orig_' + roi] = df[roi].copy(deep = True)
            df['Pred_' + roi] = df_params.loc[df_params['Features'] == roi, 'Intercept'].values + np.matmul(df[confounds], df_params.loc[df_params['Features'] == roi, confounds].to_numpy().reshape(-1))
            df[roi] =  df['Orig_' + roi] - df['Pred_' + roi] 

    for ft in features:
        df[ft] = (df[ft] - df_params.loc[df_params['Features'] == ft, 'Scaler_Mean'].values) / np.sqrt(df_params.loc[df_params['Features'] == ft, 'Scaler_Var'].values)
    
    

    df = df[all_columns]
    
    #features = features + [sex_col,]

    df = df.rename(columns={'DLICV': dlicv_col})
    
    return df #, features


