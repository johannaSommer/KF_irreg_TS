import torch
import torch.nn as nn


class DiscreteKalmanFilter(nn.Module):
    """
    Kalman Filter over discrete time steps

    Keyword arguments:
    dim -- dimension of the data (int)
    latent_dim -- dimension of the latent space (int)
    """
    def __init__(self, dim, latent_dim):
        super().__init__()
        torch.manual_seed(0)
        self.dim = dim
        self.latent_dim = latent_dim

        # State update: x_{k} = A @ x_{k-1} + q
        self.F = nn.Parameter(torch.randn(latent_dim, latent_dim) / latent_dim)
        self.pre_Q = nn.Parameter(torch.ones(1, latent_dim))
        
        # Emission: y_{k} = H @ x_{k} + r
        self.H = nn.Parameter(torch.randn(dim, latent_dim) / latent_dim)
        self.pre_R = nn.Parameter(torch.ones(1, dim))
            
        # Priors
        self.x0 = torch.zeros(latent_dim, 1)
        self.P0 = torch.eye(latent_dim, latent_dim)
        
        # Bounds for numerical instability
        self.noise_upper = 1.0
        self.noise_lower = 1e-6
        self.prior_upper = 1.0
        self.prior_lower = 1e-6
        self.innovation_upper = 1
        self.innovation_lower = 0.01

    @property
    def R(self):
        """
        Calculate observation noise covariance
        """
        # pre_R_norm = torch.sigmoid(self.pre_R) * (self.noise_upper - self.noise_lower) + self.noise_lower
        return torch.eye(self.dim, self.dim) * self.pre_R**2

    @property
    def Q(self):
        """
        Calculate process noise covariance
        """
        return torch.eye(self.latent_dim, self.latent_dim) * self.pre_Q**2

    def predict(self, x, P):
        """
        Update state mean and covariance p(x_{k} | x_{k-1}) and calculate mean and
        covariance in the observation space in the case of discrete time steps
        """
        x = self.F @ x
        P = self.F @ P @ self.F.transpose(0, 1) + self.Q

        pred_mean, pred_sigma = self.emission(x, P)
        
        assert x.shape == (self.latent_dim, 1)
        assert P.shape == (self.latent_dim, self.latent_dim)
        return x, P, pred_mean, pred_sigma

    def update(self, x, P, z, pred_sigma):
        """
        Update state x and P after the observation, 
        outputs filtered state and covariance
        """
        # Update state mean and covariance p(x | y), Joseph Form      
        # Kalman gain, a more stable implementation than naive P @ H^T @ y_sigma^{-1}
        L = torch.cholesky(pred_sigma)
        K = torch.triangular_solve(self.H @ P.transpose(0, 1), L, upper=False)[0]
        K = torch.triangular_solve(K, L.transpose(0, 1))[0].transpose(0, 1)
        
        v = z - self.H @ x
        x = x + K @ v
        P = (torch.eye(*P.shape) - K @ self.H) @ P @ (torch.eye(*P.shape) - K @ self.H).T + K @ self.R @ K.T
        
        assert x.shape == (self.latent_dim, 1)
        assert P.shape == (self.latent_dim, self.latent_dim)
        return x, P
    
    def emission(self, x, P):
        """
        emission from state space m & P to observed space mean & sigma
        """
        pred_mean = self.H @ x
        pred_sigma = self.H @ P @ self.H.transpose(0, 1) + self.R
        return pred_mean.squeeze(-1), pred_sigma

    def iterate_disc_sequence(self, z):
        """
        Iterate input data in case of discrete time steps
        """
        # Initialization
        x, P = self.x0, self.P0
        pred_means, pred_sigmas = [], []

        # Iterate through sequence performing predict-update steps
        for i in range(z.shape[0]):
            if i > 0:
                x_prio, P_prio, pred_mean, pred_sigma = self.predict(x, P)
                x, P = self.update(x_prio, P_prio, z[i].unsqueeze(-1), pred_sigma)
            else:
                pred_mean, pred_sigma = self.emission(x, P)
                x, P = self.update(x, P, z[i].unsqueeze(-1), pred_sigma)
            pred_means.append(pred_mean)
            pred_sigmas.append(pred_sigma)

        pred_means = torch.stack(pred_means)
        pred_sigmas = torch.stack(pred_sigmas)
        return pred_means, pred_sigmas, x, P

    def forward(self, z):
        """
        Forward pass for the Kalman Filter
        
        Keyword arguments:
        z -- observed values (torch.Tensor)

        Returns:
        loss - NLL of observed sequence in predicted probability dist (torch.Tensor)
        """
        assert isinstance(z, torch.Tensor)
        pred_means, pred_sigmas, x, P = self.iterate_disc_sequence(z)
            
        # evaluate observed sequence in predicted distribution
        dist = torch.distributions.MultivariateNormal(pred_means, pred_sigmas)
        loss = -dist.log_prob(z).mean()   
        loss /= len(z[0])
        return loss
    
    def forecasting(self, T, x, P):
        """
        forecast means and sigmas over given time period
        
        Keyword arguments:
        T -- observed values (int or torch.Tensor)
        x, P - last states before forecasting window

        Returns:
        pred_means, pred_sigmas
        """
        pred_means = torch.Tensor([])
        pred_sigmas = torch.Tensor([])
        assert isinstance(T, int)
        assert T > 0
        for i in range(T):
            x, P, pred_mean, pred_sigma = self.predict(x, P)
            pred_means = torch.cat([pred_means, pred_mean.unsqueeze(0)])
            pred_sigmas = torch.cat([pred_sigmas, pred_sigma.unsqueeze(0)])
                
        return pred_means, pred_sigmas
