import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets.utils.data_utils import ITSDataset


def get_OU_data(t_val=4, max_val_samples=1):
    """
    Reads OU data and formats it in ITS Dataset

    Keyword arguments:
    t_val -- how many time steps will be in the prediction (int)
    max_val_samples -- maximum number of predicted events (int)

    Returns:
    train, val -- training and validation data (ITSDataset)
    dataset_name -- name of dataset (string)
    """
    full_data = pd.read_csv('data/double_OU.csv', index_col=0)
    
    val_options = {'T_val': t_val, 'max_val_samples': max_val_samples}
    train = ITSDataset(in_df=full_data.reset_index(), validation=True, val_options=val_options)
    return train, train, '2DOU'


def get_mimic_data(norm, small_ds=False, val_size=0.4, random_state=42):
    """
    Reads MIMIC4 data and labels stored as csv and formats it into ITS Dataset

    Keyword arguments:
    norm -- normalize data to gaussian or do minmax
    small_ds -- if the dataset only consists of the first 8000 patients (bool)
    val_size -- how many percent of patient ids are used for validation (float)
    random_state -- seed used for train-val-split (int)

    Returns:
    train, val -- training and validation data (ITSDataset)
    dataset_name -- name of dataset (string)
    """
    # read patient data and labels into dataframe
    full_data = pd.read_csv(os.path.join(os.getcwd(), 'data/full_dataset.csv'), index_col=0, na_filter=False)
    full_labels = pd.read_csv(os.path.join(os.getcwd(), 'data/complete_death_tags.csv'))
    full_labels.drop(['unique_id'], axis=1, inplace=True)
    covariates = pd.read_csv(os.path.join(os.getcwd(), 'data/covariates.csv'), index_col=0, na_filter=False,
                             low_memory=False)
    covariates.drop([''], inplace=True)
    covariates.index = covariates.index.astype(float).astype(int)
    
    joined_idx = np.intersect1d(full_labels.index.unique(), covariates.index.unique())
    full_data = full_data.loc[joined_idx]
    covariates = covariates.loc[joined_idx]
    
    # only take the first 8000 patient ids and their measurements
    if small_ds:
        full_data = full_data.loc[full_data.index.unique()[:3500]]
        covariates = covariates.loc[full_data.index.unique()[:3500]]
        
    assert len(full_data.index.unique()) == len(covariates.index.unique())
    print("# of patients:", len(full_data.index.unique()))

    # normalize data to N(0, 1)
    value_cols = [c.startswith('Value') for c in full_data.columns]
    value_cols = full_data.iloc[:, value_cols]
    mask_cols = [('Mask' + x[5:]) for x in value_cols]
    
    if norm == "minmax":
        for item in zip(value_cols, mask_cols):
            temp = full_data.loc[full_data[item[1]].astype('bool'), item[0]]
            full_data.loc[full_data[item[1]].astype('bool'), item[0]] = (temp - temp.min()) / (temp.max() - temp.min())
    elif norm == "gaussian":
        for item in zip(value_cols, mask_cols):
            temp = full_data.loc[full_data[item[1]].astype('bool'), item[0]]
            full_data.loc[full_data[item[1]].astype('bool'), item[0]] = 0.5 * ((temp - temp.mean()) / temp.std())
    else:
        raise NotImplementedError()

    full_data.dropna(inplace=True)
    
    # normalize also covariates
    value_cols = [c.startswith('Value') for c in covariates.columns]
    value_cols = covariates.iloc[:, value_cols]
    covariates.replace(0, np.nan, inplace=True)
    if norm == "minmax":
        for col in value_cols:
            temp = covariates.loc[:, col]
            covariates.loc[:, col] = (temp - temp.min()) / (temp.max() - temp.min())
    elif norm == "gaussian":
        for col in value_cols:
            temp = covariates.loc[:, col]
            covariates.loc[:, col] = 0.5 * ((temp - temp.mean()) / temp.std())
    else:
        raise NotImplementedError()
        
    covariates.fillna(0, inplace=True)
    
    full_data.loc[:, 'Time'] = full_data['Time'] / 1000
    covariates.loc[:, 'Time'] = covariates['Time'] / 1000

    # split into train and validation by patient ids
    train_idx, val_idx = train_test_split(full_data.index.unique(), test_size=val_size, random_state=random_state)
    val_idx, test_idx = train_test_split(val_idx, test_size=0.5, random_state=random_state)    

    # for forecasting, specify point up to which measurements are used,
    val_options = {'T_val': 2.16, 'max_val_samples': 3}
    train = ITSDataset(in_df=full_data.loc[train_idx].reset_index(),
                       label_df=full_labels.loc[train_idx].reset_index(), 
                       cont_cov_df=covariates.loc[train_idx].reset_index())
    val = ITSDataset(in_df=full_data.loc[val_idx].reset_index(),
                     label_df=full_labels.loc[val_idx].reset_index(), 
                     cont_cov_df=covariates.loc[val_idx].reset_index(), 
                     validation=True, val_options=val_options)
    test = ITSDataset(in_df=full_data.loc[test_idx].reset_index(),
                      label_df=full_labels.loc[test_idx].reset_index(),
                      cont_cov_df=covariates.loc[test_idx].reset_index(),
                      validation=True, val_options=val_options)
    
    return train, val, test, 'MIMIC'
