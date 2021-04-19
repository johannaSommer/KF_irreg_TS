import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class ITSDataset(Dataset):
    """
    Dataset class for irregular data, originally taken from
    https://github.com/edebrouwer/gru_ode_bayes
    and modified
    """
    def __init__(self, in_df, init_cov_df=None, cont_cov_df=None, label_df=None, validation=False, val_options=None):
        """
        Keyword arguments:
        in_df -- patient data (pd.DataFrame)
        init_cov_df -- covariates used for hidden state initialization (pd.DataFrame)
        cont_cov_df -- covariates used throughout the series (pd.DataFrame)
        label_df -- death tags for each patient (pd.DataFrame)
        validation -- if the constructed dataset is used for validation (bool)
        val_options -- options to specify forecasting time period (dict)
        """
        self.validation = validation
        self.df = in_df
        self.init_cov_df = init_cov_df
        self.cont_cov_df = cont_cov_df
        self.label_df = label_df
        self.cov_available = True

        # Create Dummy covariates and labels if they are not fed.
        num_unique = np.zeros(self.df['ID'].nunique())
        if self.init_cov_df is None:
            self.init_cov_df = pd.DataFrame({'ID': self.df['ID'].unique(), 'Cov': num_unique})
        if self.label_df is None:
            self.label_df = pd.DataFrame({'ID': self.df['ID'].unique(), 'label': num_unique})
        if self.cont_cov_df is None:
            self.cont_cov_df = pd.DataFrame({'ID': self.df['ID'].unique(), 'label': num_unique})
            self.cov_available = False
        
        # If validation : consider only the data with a least one observation before T_val and one observation after:
        self.store_last = False
        if self.validation:
            before_idx = self.df.loc[self.df['Time'] <= val_options['T_val'], 'ID'].unique()
            # Validation samples only after some time.
            if val_options.get('T_val_from'):
                after_idx = self.df.loc[self.df['Time'] >= val_options['T_val_from'], 'ID'].unique()
                # Dataset get will return a flag for the collate to compute the last sample before T_val
                self.store_last = True
            else:
                after_idx = self.df.loc[self.df['Time'] > val_options['T_val'], 'ID'].unique()
            
            valid_idx = np.intersect1d(before_idx, after_idx)
            self.df = self.df.loc[self.df['ID'].isin(valid_idx)].copy()
            self.label_df = self.label_df.loc[self.label_df['ID'].isin(valid_idx)].copy()
            self.init_cov_df = self.init_cov_df.loc[self.init_cov_df['ID'].isin(valid_idx)].copy()

        # reset indices so that they start at 0
        map_dict = dict(zip(self.df.loc[:, 'ID'].unique(), np.arange(self.df.loc[:, 'ID'].nunique())))
        self.df.loc[:, 'ID'] = self.df.loc[:, 'ID'].map(map_dict)
        self.init_cov_df.loc[:, 'ID'] = self.init_cov_df.loc[:, 'ID'].map(map_dict)
        self.label_df.loc[:, 'ID'] = self.label_df['ID'].map(map_dict)
        self.cont_cov_df.loc[:, 'ID'] = self.cont_cov_df['ID'].map(map_dict)
        
        assert self.init_cov_df.shape[0] == self.df['ID'].nunique()

        # number of variables in the dataset
        self.variable_num = sum([c.startswith('Value') for c in self.df.columns])
        self.init_cov_dim = self.init_cov_df.shape[1] - 1
        self.init_cov_df = self.init_cov_df.astype(np.float32)
        self.init_cov_df.set_index('ID', inplace=True)
        self.cont_cov_df.set_index('ID', inplace=True)
        self.label_df.set_index('ID', inplace=True)
        self.df = self.df.astype(np.float32)

        if self.validation:
            assert val_options is not None, 'Validation set options should be fed'
            self.df_before = self.df.loc[self.df['Time'] <= val_options['T_val']].copy()
            if val_options.get('T_val_from'):  # Validation samples only after some time.
                self.df_after = self.df.loc[self.df['Time'] >= val_options['T_val_from']].sort_values('Time').copy()
            else:
                self.df_after = self.df.loc[self.df['Time'] > val_options['T_val']].sort_values('Time').copy()
            if val_options.get('T_closest') is not None:
                df_after_temp = self.df_after.copy()
                df_after_temp['Time_from_target'] = (df_after_temp['Time'] - val_options['T_closest']).abs()
                df_after_temp.sort_values(by=['Time_from_target', 'Value_0'], inplace=True, ascending=True)
                df_after_temp.drop_duplicates(subset=['ID'], keep='first', inplace=True)
                self.df_after = df_after_temp.drop(columns=['Time_from_target'])
            else:
                self.df_after = self.df_after.groupby('ID').head(val_options['max_val_samples']).copy()
            self.df = self.df_before  # We remove observations after T_val
            self.df_after.ID = self.df_after.ID.astype(np.int)
            self.df_after.sort_values('Time', inplace=True)
        else:
            self.df_after = None

        self.length = self.df['ID'].nunique()
        self.df.ID = self.df.ID.astype(np.int)
        self.df.set_index('ID', inplace=True)
        self.df.sort_values('Time', inplace=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        subset = self.df.loc[idx]
        if len(subset.shape) == 1:
            subset = self.df.loc[[idx]]
        cont_covs = self.cont_cov_df.loc[idx]
        if len(cont_covs.shape) == 1:
            cont_covs = self.cont_cov_df.loc[[idx]]
        init_covs = self.init_cov_df.loc[idx].values
        covs_av = self.cov_available
        tag = self.label_df.loc[idx].astype(np.float32).values
        if self.validation:
            val_samples = self.df_after.loc[self.df_after['ID'] == idx]
        else:
            val_samples = None
        # returning also idx to allow empty samples
        return {'idx': idx, 'y': tag, 'path': subset, 'init_cov': init_covs, 'val_samples': val_samples,
                'param_cov': cont_covs, 'covs_av': covs_av, 'store_last': self.store_last}


def collate_KF(batch):
    """
    Collate function used in the DataLoader to format data for KF
    Note: all samples are concatenated to one single series so that the KF only
    needs to iterate through time once
    """
    # unify sample dfs into one big df and reset sample IDs
    df = pd.concat([b['path'] for b in batch], axis=0)
    df.reset_index(inplace=True)
    map_dict = dict(zip(df['ID'].unique(), np.arange(df['ID'].nunique())))
    df['ID'] = df['ID'].map(map_dict)
    df.set_index('ID', inplace=True)
    df.sort_values(by=['ID', 'Time'], inplace=True)
    
    times = [df.loc[i].Time.values if isinstance(df.loc[i].Time, pd.Series) 
             else np.array([df.loc[i].Time]) for i in df.index.unique()]
    num_observations = [len(x) for x in times]

    value_cols = [c.startswith('Value') for c in df.columns]
    mask_cols = [c.startswith('Mask') for c in df.columns]
    z = torch.Tensor(df.loc[:, value_cols].to_numpy())
    mask = torch.Tensor(df.loc[:, mask_cols].to_numpy())
        
    # validation data    
    if batch[0]['val_samples'] is not None:   
        val_df = pd.concat([b['val_samples'] for b in batch], axis=0)
        val_df.reset_index(inplace=True)
        val_df['ID'] = val_df['ID'].map(map_dict)
        val_df.set_index('ID', inplace=True)
        val_df.drop('index', inplace=True, axis=1)
        val_df.sort_values(by=['ID', 'Time'], inplace=True)
        val_times = [val_df.loc[i].Time.values if isinstance(val_df.loc[i].Time, pd.Series)
                     else np.array([val_df.loc[i].Time]) for i in val_df.index.unique()]
        val_z = torch.Tensor(val_df.loc[:, value_cols].to_numpy())
        val_mask = torch.Tensor(val_df.loc[:, mask_cols].to_numpy())    
    else:
        val_times = None
        val_z = None
        val_mask = None
        
    # covariates
    if batch[0]['covs_av']:
        covariates = pd.concat([b['param_cov'] for b in batch], axis=0)
        covariates.reset_index(inplace=True)
        covariates['ID'] = covariates['ID'].map(map_dict)
        covariates.set_index('ID', inplace=True)
        covariates.sort_values(by=['ID', 'Time'], inplace=True)
        cov_times = [covariates.loc[i].Time.values if isinstance(covariates.loc[i].Time, pd.Series) 
                     else np.array([covariates.loc[i].Time]) for i in covariates.index.unique()]
        cov_value_cols = [c.startswith('Value') for c in covariates.columns]
        cov_values = [torch.Tensor(covariates.loc[i, cov_value_cols].to_numpy()) 
                      if len(torch.Tensor(covariates.loc[i, cov_value_cols].to_numpy()).shape) > 1 
                      else torch.Tensor(covariates.loc[i, cov_value_cols].to_numpy()).unsqueeze(0) 
                      for i in covariates.index.unique()]
    else: 
        cov_values = None
        cov_times = None
        
    res = dict()
    res['ids'] = np.unique(df.index.values)
    res['z'] = z
    res['mask'] = mask
    res['times'] = times
    res['numobs'] = num_observations
    
    res['val_z'] = val_z
    res['val_mask'] = val_mask
    res['val_times'] = val_times
    res['cov_values'] = cov_values
    res['cov_times'] = cov_times

    return res
