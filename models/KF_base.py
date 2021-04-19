import nf
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from itertools import compress
from models.KF_cells import *


def masked_likelihood(batch, pred_means, pred_sigmas, flow):
    """
    Computes masked likelihood of observed values in predicted distributions

    Keyword arguments:
    batch -- batch like constructed in datasets.utils.data_utils.collate_KF (dict)
    pred_means -- predicted means (torch.Tensor)
    pred_sigmas -- predicted sigmas (torch.Tensor)
    flow -- instance of flow (nf.Flow)

    Returns:
    loss -- mean negative log-likelihood
    """
    loss = torch.Tensor([]).cuda()
    obs = batch['z']
    mask = batch['mask'].to(torch.bool)
    # no other implementation possible, calc now scales in number of dimensions & not in number of observations
    for d in range(1, mask.shape[1]+1):
        mm = (torch.sum(mask, dim=1) == d)
        if mm.sum() > 0:
            m = mask[mm]
            means = torch.stack(torch.chunk(pred_means[mm][m], len(m), dim=0))
            sigmas = torch.stack(torch.chunk(pred_sigmas[mm][m], len(m), dim=0)).transpose(2, 1)
            sigmas = torch.stack(torch.chunk(sigmas[m], len(m), dim=0)).transpose(2, 1)
            # special case: only one dim observed -> normal distribution
            if d == 1:
                flow.base_dist = torch.distributions.Normal(means.squeeze(-1), torch.sqrt(sigmas.squeeze(-1).squeeze(-1)))
                transformed_obs, log_jac_diag = flow.inverse(obs[mm], mask=m.to(torch.int))
                res = flow.base_dist.log_prob(transformed_obs[mask[mm]].squeeze(0)) + log_jac_diag.sum(-1)
                res = res if len(res.shape) > 0 else res.unsqueeze(0)
            else:
                flow.base_dist = torch.distributions.MultivariateNormal(means, sigmas)
                transformed_obs, log_jac_diag = flow.inverse(obs[mm], mask=m.to(torch.int))
                res = (flow.base_dist.log_prob(torch.stack(torch.chunk(transformed_obs[m], len(m), dim=0))) + log_jac_diag.sum(-1))
            loss = torch.cat((loss, res))
    return -torch.sum(loss)/mask.sum()


def discretize(F, Q, t, device):
    """
    Discretize matrices for continuous update step with matrix exponential and matrix fraction decomposition
    
    Keyword arguments:
    F, Q -- KF matrices for state update (torch.Tensor)
    t -- time delta to last observation (torch.Tensor)
    device -- device of model (torch.device)

    Returns:
    A, L -- discretized update matrices
    """
    if len(F.shape) == 3:
        m = F.shape[0]
        n = F.shape[1]
        M = torch.zeros(m, 2*n, 2*n)
        A = torch.matrix_exp(F * t.unsqueeze(-1).unsqueeze(-1).to(device))
        M[:, :n, :n] = F
        M[:, :n, n:] = Q
        M[:, n:, n:] = -F.transpose(1, 2)
        M = torch.matrix_exp(M * t.unsqueeze(-1).unsqueeze(-1)) @ torch.cat([torch.zeros(n, n), torch.eye(n, n)])
    else:
        n = F.shape[0]
        M = torch.zeros(2*n, 2*n)
        A = torch.matrix_exp(F * t.view(-1, 1, 1).to(device))
        M[:n, :n] = F
        M[:n, n:] = Q
        M[n:, n:] = -F.T
        M = torch.matrix_exp(M * t.view(-1, 1, 1)) @ torch.cat([torch.zeros(n, n), torch.eye(n, n)])
    C, D = M[:, :n], M[:, n:]
    L = C @ torch.inverse(D)
    return A.to(device), L.to(device)


class KalmanFilter(nn.Module):
    """
    Kalman Filter in continuous time

    Keyword arguments:
    dim -- dimension of the data (int)
    latent_dim -- dimension of the latent space (int)
    fixed_params -- whether the KF matrices are fixed or predicted over time (bool)
    hidden_dim -- hidden size of parameter mapping function(int)
    normflow -- flow used for emission, can be identity for no nfs (nf.Flow)
    cov_dim -- dimension of covariates (int)
    use_covs -- whether to use covariates to predict parameters (bool)
    use_ode -- use ODE or fixed function to predict parameters (bool)
    device -- device to cast tensors to (string)
    discretize -- discretize KF matrices or use torchdiffeq (bool)
    """
    def __init__(self, seed, dim, latent_dim, fixed_params, hidden_dim, normflow, cov_dim, use_covs, use_ode,
                 device, discretize=True):
        super().__init__()
        torch.manual_seed(seed)
        self.dim = dim
        self.latent_dim = latent_dim
        self.device = device
        self.disc = discretize
        self.flow = normflow
        self.fixed_params = fixed_params
        self.use_covs = use_covs
        self.use_ode = use_ode
        self.ode_opts = {
            'method': 'dopri5',
            't': torch.Tensor([0, 1]).to(self.device),
            'options': dict(dtype=torch.float32)
        }
        
        if fixed_params:
            # State update: x_{k} = A @ x_{k-1} + q
            if discretize:
                self.F = nn.Parameter(torch.randn(latent_dim, latent_dim, device=torch.device(self.device)) / latent_dim)
                self.pre_Q = nn.Parameter(torch.ones(1, latent_dim, device=torch.device(self.device)))
            else:
                self.F = None
                self.pre_Q = None
                self.ode_cell = ODECell(dim, latent_dim, device)
                self.ode_cell.to(self.device)

            # Emission: y_{k} = H @ x_{k} + r
            self.H = nn.Parameter(torch.randn(dim, latent_dim, device=torch.device(self.device)) / latent_dim)
            self.pre_R = nn.Parameter(torch.ones(1, dim, device=torch.device(self.device)))

        else:
            if not discretize:
                self.ode_cell = ODECell_dynamic(dim, latent_dim, device)
                self.ode_cell.to(self.device)

            if use_ode:
                self.rnn_prop = RNNprop(latent_dim, hidden_dim, device)
                self.rnn_prop.to(self.device)
            else: 
                assert not use_covs
                self.param_func = ParamFunc(hidden_dim)
                self.param_func.to(self.device)
                
            self.rnn_map = RNNmap(dim, latent_dim, hidden_dim, device)
            self.rnn_map.to(self.device)        
            self.hidden_dim = hidden_dim

            if use_covs:
                self.rnn_update = RNNupdate(latent_dim, cov_dim, hidden_dim)                

    @property
    def R(self):
        """
        Calculate observation noise covariance
        """
        return torch.eye(self.dim, self.dim, device=torch.device(self.device)) * self.pre_R.clone()**2
    
    def Q(self):
        """
        Calculate measurement noise covariance
        """
        return torch.eye(self.latent_dim, self.latent_dim, device=torch.device(self.device)) * self.pre_Q**2
    
    def initialize_params(self, bsize):
        """
        Initialize parameter for every sample in batch
        """
        x = torch.zeros(self.latent_dim, 1, device=torch.device(self.device)).repeat((bsize, 1, 1))
        P = torch.eye(self.latent_dim, self.latent_dim, device=torch.device(self.device)).repeat((bsize, 1, 1))    
        
        if not self.fixed_params:    
            self.hidden_state = torch.zeros(1, self.hidden_dim, device=self.device).repeat((bsize, 1, 1))
            self.H = torch.zeros((bsize, self.dim, self.latent_dim), device=self.device)
            self.pre_R = torch.zeros((bsize, 1, self.dim), device=self.device)
            self.F = torch.zeros((bsize, self.latent_dim, self.latent_dim), device=self.device)
            self.pre_Q = torch.zeros((bsize, 1, self.latent_dim), device=self.device)
        
        return x, P
    
    def set_params(self, diff, idx, t):
        """
        Evolve hidden state and set parameters for current time step
        """
        if self.use_ode:
            y0 = (self.hidden_state[idx], diff.to(self.device))
            solution = odeint(self.rnn_prop, y0, adjoint_options=dict(norm=make_norm(y0)), **self.ode_opts)
            self.hidden_state[idx] = solution[0][1]
        else:
            self.hidden_state[idx] = self.param_func(t.to(self.device)).unsqueeze(1)
        self.F[idx], self.H[idx], self.pre_R[idx], self.pre_Q[idx] = self.rnn_map(self.hidden_state[idx].clone())

    def set_params_wcov(self, idx_covs, cov_values, idx):
        """
        Evolve hidden state and set parameters for current time step, update hidden state
        every time a covariate is observed
        """
        if len(cov_values) > 0:
            cov_values = torch.stack(cov_values)
            hs = self.hidden_state[idx].clone()   
            hs[idx_covs] = self.rnn_update(cov_values.to(self.device)).unsqueeze(1)
            self.hidden_state[idx] = hs.clone()            
        self.F[idx], self.H[idx], self.pre_R[idx], self.pre_Q[idx] = self.rnn_map(self.hidden_state[idx].clone())

    def update(self, x, P, z, mask, idx):
        """
        Update state x and P after the observation, outputs filtered state and covariance
        Masking done according to formulas from
        https://www.stat.pitt.edu/stoffer/tsa4/tsa4.pdf, Chapter 6.4 Page 311
        https://www.stat.pitt.edu/stoffer/dss_files/em.pdf, Chapter 3
        """
        if self.fixed_params:
            # create one parameter matrix for each observed sample in batch
            pre_R_masked = self.pre_R.clone()
            pre_R_masked = pre_R_masked.repeat(mask.shape[0], 1)
            identities = torch.eye(self.dim, self.dim, device=self.device).repeat(mask.shape[0], 1, 1)
            H_masked = self.H.clone().repeat(mask.shape[0], 1, 1)    
        else:
            # create one parameter matrix for each observed sample in batch
            pre_R_masked = self.pre_R[idx].clone().squeeze(1)
            identities = torch.eye(self.dim, self.dim, device=self.device).repeat(mask.shape[0], 1, 1)
            H_masked = self.H[idx].clone()

        # mask parameter matrices for partial measurements
        pre_R_masked[~mask.to(torch.bool)] = 1
        R_masked = identities * (pre_R_masked**2).unsqueeze(-1)
        H_masked[~mask.to(torch.bool)] = 0
        # masked emission
        masked_sigma = H_masked @ P @ H_masked.transpose(1, 2) + R_masked
        # Kalman gain, a more stable implementation than naive P @ H^T @ y_sigma^{-1}
        L = torch.cholesky(masked_sigma)
        K = torch.triangular_solve(H_masked @ P.transpose(1, 2), L, upper=False)[0]
        K = torch.triangular_solve(K, L.transpose(1, 2))[0].transpose(1, 2)
        z = (self.flow.inverse(z.squeeze(-1))[0] * mask).unsqueeze(-1)
        # Update state mean and covariance p(x | y)
        v = z - H_masked @ x
        x = x + K @ v  
        identity = torch.eye(*P.shape[1:], device=torch.device(self.device))
        # Joseph Form for numerical stability
        P = (identity - K @ H_masked) @ P @ (identity - K @ H_masked).transpose(1, 2) + K @ R_masked @ K.transpose(1, 2)

        return x, P

    def emission(self, x, P, idx):
        """
        emission from state space m & P to observed space mean & sigma
        """
        if self.fixed_params:
            pred_mean = self.H @ x
            pred_sigma = self.H @ P @ self.H.transpose(0, 1) + self.R
        else:
            pred_mean = self.H[idx] @ x
            pred_sigma = self.H[idx] @ P @ self.H[idx].transpose(1, 2) + self.R[idx]            
        return pred_mean.squeeze(-1), pred_sigma
    
    def iterate_cont_sequence(self, batch, x, P):
        """
        Iterate input data in case of continuous time
        """
        # Initialization & casting
        batch['z'] = batch['z'].to(self.device)
        batch['mask'] = batch['mask'].to(self.device)
        batch['numobs'] = torch.Tensor(batch['numobs']).to(self.device)
        if len(batch['times']) == 1:
            batch['times'] = torch.Tensor(batch['times'])
        else:
            batch['times'] = np.array(batch['times'], dtype=object)
        if self.use_covs:
            cov_times = np.array(batch['cov_times'], dtype=object)
            cov_values = np.array(batch['cov_values'], dtype=object)
        pred_means, pred_sigmas, z_reord, mask_reord = [], [], [], []
        last_times = torch.zeros((len(batch['numobs'])))
        
        for ind in range(0, int(torch.max(batch['numobs']).item())):
            # get ids of the samples observed at this time step slice 
            idx = batch['numobs'] > ind
            # calculate time difference to last step
            current_times = torch.Tensor([x[ind] for x in batch['times'][idx.cpu()]])
            diff = current_times - last_times[idx]
            
            if not self.fixed_params:
                if self.use_covs:
                    evals, vals = [], []
                    for k, time in enumerate(cov_times[idx.cpu()]):
                        mask = torch.Tensor(time) <= current_times[k]
                        evals.append(mask.sum() > 0)
                        if mask.sum() > 0:
                            vals.append(cov_values[idx.cpu()][k][mask][-1])
                    self.set_params_wcov(torch.stack(evals), vals, idx)
                else:
                    self.set_params(diff, idx, current_times)

            if self.disc:
                if self.fixed_params:
                    A, L = discretize(self.F, self.Q(), diff, self.device)
                else:
                    Q = torch.eye(self.latent_dim, self.latent_dim, device=torch.device(self.device)).repeat(self.pre_Q[idx].shape[0], 1, 1)
                    Q *= self.pre_Q[idx]**2
                    A, L = discretize(self.F[idx], Q, diff, self.device)
                x[idx] = A @ x[idx]
                P[idx] = A @ P[idx] @ A.transpose(1, 2) + L
            else:
                y0 = (x[idx].to(self.device), P[idx].to(self.device), diff.to(self.device))
                solution = odeint(self.ode_cell, y0, adjoint_options=dict(norm=make_norm(y0)), **self.ode_opts)
                # assign prediction to states               
                x[idx] = solution[0][1]
                P[idx] = solution[1][1]        

            # 'convert' x and P to observation space by emission
            last_times[idx] = current_times
            pred_mean, pred_sigma = self.emission(x[idx], P[idx], idx)
            pred_means.append(pred_mean)
            pred_sigmas.append(pred_sigma)

            # get observations and masks in correct order, update states
            zero_tens = torch.Tensor([0]).to(self.device)
            z_slice = batch['z'][(torch.cat((zero_tens, torch.cumsum(batch['numobs'], dim=0)))[:-1][idx] + ind).long()]
            m_slice = batch['mask'][(torch.cat((zero_tens, torch.cumsum(batch['numobs'], dim=0)))[:-1][idx] + ind).long()]
            x[idx], P[idx] = self.update(x[idx], P[idx], z_slice.unsqueeze(-1), m_slice, idx)
            z_reord.append(z_slice)
            mask_reord.append(m_slice)

        return pred_means, pred_sigmas, x, P, z_reord, mask_reord

    def forward(self, inp):
        """
        Forward pass for the Kalman Filter
        
        Keyword arguments:
        inp -- batch like constructed in datasets.utils.data_utils.collate_KF (dict)

        Returns:
        loss - NLL of observed sequence in predicted probability dist (torch.Tensor)
        """
        x, P = self.initialize_params(inp['ids'].shape[0])
        pred_means, pred_sigmas, _, _, new_z, new_mask = self.iterate_cont_sequence(inp, x, P)
        pred_means = torch.cat(pred_means)
        pred_sigmas = torch.cat(pred_sigmas)
        syn_batch = dict()
        syn_batch['z'] = torch.cat(new_z)
        syn_batch['mask'] = torch.cat(new_mask)
        loss = masked_likelihood(syn_batch, pred_means, pred_sigmas, self.flow)
        return loss
    
    def forecasting(self, times, batch, x, P):
        """
        forecast means and sigmas over given time period
        
        Keyword arguments:
        times --  times of observations for every sample, first element
                   has to be time of last state update (torch.Tensor)
        batch -- batch like constructed in datasets.utils.data_utils.collate_KF (dict)
        x, P - states at times[:, 0] (torch.Tensor)

        Returns:
        pred_means, pred_sigmas
        """
        pred_means = torch.Tensor([]).to(self.device)
        pred_sigmas = torch.Tensor([]).to(self.device)
        x = x.to(self.device)
        P = P.to(self.device)
        
        val_numobs = torch.Tensor([len(x) for x in times])
        if self.use_covs:
            cov_times = np.array(batch['cov_times'], dtype=object)
            cov_values = np.array(batch['cov_values'], dtype=object)
        
        for ind in range(1, int(torch.max(val_numobs).item())):
            idx = val_numobs > ind
            current_times = torch.Tensor([x[ind] for x in list(compress(times, idx.cpu()))])
            last_times = torch.Tensor([x[ind-1] for x in list(compress(times, idx.cpu()))])
            diff = current_times - last_times
            if not self.fixed_params:
                if self.use_covs:
                    evals, vals = [], []
                    for k, time in enumerate(cov_times[idx.cpu()]):
                        mask = torch.Tensor(time) <= current_times[k]
                        evals.append(mask.sum() > 0)
                        if mask.sum() > 0:
                            vals.append(cov_values[idx.cpu()][k][mask][-1])
                    self.set_params_wcov(torch.stack(evals), vals, idx)
                else:
                    self.set_params(diff, idx, current_times)     

            if self.disc:
                if self.fixed_params:
                    A, L = discretize(self.F, self.Q(), diff, self.device)
                else:
                    Q = torch.eye(self.latent_dim, self.latent_dim, device=torch.device(self.device)).repeat(self.pre_Q[idx].shape[0], 1, 1)
                    Q *= self.pre_Q[idx]**2
                    A, L = discretize(self.F[idx], Q, diff, self.device)
                x[idx] = A @ x[idx]
                P[idx] = A @ P[idx] @ A.transpose(1, 2) + L
            else:
                # no need to have 12LOC solution here, as it only speeds up backward pass
                y0 = (x[idx].to(self.device), P[idx].to(self.device), diff.to(self.device), self.F[idx], self.pre_Q[idx])
                solution = odeint(self.ode_cell, y0, **self.ode_opts)
                x[idx] = solution[0][1]
                P[idx] = solution[1][1]   

            # 'convert' x and P to observation space by emission
            pred_mean, pred_sigma = self.emission(x[idx], P[idx], idx)
            pred_means = torch.cat([pred_means, pred_mean])
            pred_sigmas = torch.cat([pred_sigmas, pred_sigma])
        return pred_means, pred_sigmas  

    
def rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()


def make_norm(state):
    state_size = 0
    for x in state:
        state_size += x.numel()

    def norm(aug_state):
        y = aug_state[1:1 + state_size]
        adj_y = aug_state[1 + state_size:1+state_size*2]
        return max(rms_norm(y), rms_norm(adj_y))
    return norm
