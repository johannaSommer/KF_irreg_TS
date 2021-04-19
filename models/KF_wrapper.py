import nf
import torch
from models.KF_base import KalmanFilter

"""
Provide these parameters to the model, the wrappers will take care of the rest:
seed, dim, latent_dim, 
For RKF_F, RKF_O, RCKF additionally provide:
hidden_dim 
For RCKF additionally provide:
cov_dim
"""


def get_identity_flow():
    transforms = [nf.Identity()]
    flow = nf.Flow(torch.distributions.Normal(0, 1), transforms)
    flow.to("cuda")
    return flow


def KF(**kwargs):
    flow = get_identity_flow()
    return KalmanFilter(**kwargs, device='cuda', discretize=True, hidden_dim=0, cov_dim=0, use_covs=False,
                        use_ode=False, normflow=flow, fixed_params=True)


def NKF(**kwargs):
    torch.manual_seed(0)
    hdim = 4
    transforms = [nf.Coupling(nf.Affine(kwargs["dim"], hdim), nf.net.MLP(kwargs["dim"], [hdim] * 2, hdim),
                              mask='ordered_right_half'),
                  nf.Coupling(nf.Affine(kwargs["dim"], hdim), nf.net.MLP(kwargs["dim"], [hdim] * 2, hdim),
                              mask='ordered_left_half')]
    flow = nf.Flow(torch.distributions.Normal(0, 1), transforms)
    flow.to("cuda")
    return KalmanFilter(**kwargs, device='cuda', discretize=True, hidden_dim=0, cov_dim=0, use_covs=False,
                        use_ode=False, normflow=flow, fixed_params=True)

       
def RKF_F(**kwargs):
    flow = get_identity_flow()
    return KalmanFilter(**kwargs, cov_dim=0, use_covs=False, use_ode=False, device='cuda', discretize=True,
                        normflow=flow, fixed_params=False)
    
    
def RKF_O(**kwargs):
    flow = get_identity_flow()
    return KalmanFilter(**kwargs, cov_dim=0, use_covs=False, use_ode=True, device='cuda', discretize=True,
                        normflow=flow, fixed_params=False)
    
    
def RCKF(**kwargs):
    flow = get_identity_flow()
    return KalmanFilter(**kwargs, use_covs=True, use_ode=True, device='cuda', discretize=True,
                        normflow=flow, fixed_params=False)
