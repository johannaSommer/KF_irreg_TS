import torch
import torch.nn as nn


class ODECell_fixed(nn.Module):
    """
    KF state update in continuous time, formulated as nn.Module so that
    odeint_adjoint can be used
    """
    def __init__(self, dim, latent_dim, device):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.device = device
        
        self.F = nn.Parameter(torch.randn(latent_dim, latent_dim, device=torch.device(self.device)))
        self.pre_Q = nn.Parameter(torch.ones(1, latent_dim, device=torch.device(self.device)))
        
    def forward(self, t, inp):
        """
        dx/dt that will be handed to ODE solver as func, continuous time
        """
        x, P, diff = inp[0], inp[1], inp[2]
        dx = (self.F @ x) * diff.unsqueeze(-1).unsqueeze(-1)
        Q = torch.eye(self.latent_dim, self.latent_dim, device=torch.device(self.device)) * self.pre_Q**2
        dP = (self.F @ P + P @ self.F.T + Q) * diff.unsqueeze(-1).unsqueeze(-1)
        return dx, dP, torch.zeros_like(diff)


class ODECell_dynamic(nn.Module):
    """
    KF state update in continuous time, formulated as nn.Module so that
    odeint_adjoint can be used
    """
    def __init__(self, dim, latent_dim, device):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.device = device
        
    def forward(self, t, inp):
        """
        dx/dt that will be handed to ODE solver as func, continuous time
        """
        # parameter F and Q are used as inputs to the ODE but with dx/dt=0 to stay constant
        # https://github.com/rtqichen/torchdiffeq/issues/64
        x, P, diff = inp[0], inp[1], inp[2]
        F, pre_Q = inp[3], inp[4]
        
        dx = (F @ x) * diff.unsqueeze(-1).unsqueeze(-1)
        Q = torch.cat(pre_Q.shape[0]*[torch.eye(self.latent_dim, self.latent_dim, device=torch.device(self.device)).unsqueeze(0)]) * pre_Q**2
        dP = (F @ P + P @ F.transpose(1, 2) + Q) * diff.unsqueeze(-1).unsqueeze(-1)
        
        return dx, dP, torch.zeros_like(diff), torch.zeros_like(F), torch.zeros_like(pre_Q)
    

class ParamFunc(nn.Module):
    def __init__(self, nhidden):
        super().__init__()
        self.nhidden = nhidden
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(1, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)

    def forward(self, t):
        out = self.fc1(t.unsqueeze(-1))
        out = self.elu(out)
        out = self.fc2(out)
        return out
    

class RNNprop(nn.Module):
    """
    ODE dynamics to propagate parameter hidden state through
    """
    def __init__(self, latent_dim, nhidden, device):
        super().__init__()
        self.nhidden = nhidden
        self.latent_dim = latent_dim
        self.device = device
        
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(nhidden, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, nhidden)
        
    def forward(self, t, inp):
        x, diff = inp[0], inp[1]
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        # force elementwise multiply in batch
        out = out * diff.unsqueeze(-1).unsqueeze(-1)
        return out, torch.zeros_like(diff)
        
        
class RNNupdate(nn.Module):
    """
    Update parameter hidden state when covariates are observed
    """
    def __init__(self, latent_dim, obs_dim, nhidden):
        super().__init__()
        self.nhidden = nhidden
        self.i2h = nn.Linear(obs_dim, nhidden)
        self.h2h = nn.Linear(nhidden, nhidden)

    def forward(self, x):
        h = torch.tanh(self.i2h(x))
        h = torch.tanh(self.h2h(h))
        return h
        

class RNNmap(nn.Module):
    """
    Map parameter hidden state to KF matrices
    """
    def __init__(self, dim, latent_dim, nhidden, device):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.dim = dim
        self.fc1 = nn.Linear(nhidden, latent_dim * latent_dim)
        self.fc2 = nn.Linear(nhidden, dim * latent_dim)
        self.fc3 = nn.Linear(nhidden, dim)
        self.fc4 = nn.Linear(nhidden, latent_dim)
    
    def forward(self, h):
        F = torch.reshape(self.fc1(h).squeeze(-2), (h.shape[0], self.latent_dim, self.latent_dim))
        H = torch.reshape(self.fc2(h).squeeze(-2), (h.shape[0], self.dim, self.latent_dim))
        R = self.fc3(h)
        Q = self.fc4(h)
        return F, H, R, Q
