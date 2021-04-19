from models.KF_wrapper import *
from training.KF_train_utils import train_loop_KF
from datasets.utils.get_data import get_mimic_data
from sacred import Experiment
import seml


ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(epochs, device, dim, ldim, batch_size, lr, seed, cov_dim, hiddendim, model_name, norm, reg_lambda, ds_seed):

    train, val, _, name = get_mimic_data(norm=norm, small_ds=False, random_state=ds_seed)
    
    if model_name == "KF":
        model = KF(seed=seed, dim=dim, latent_dim=ldim)
    elif model_name == "NKF":
        model = NKF(seed=seed, dim=dim, latent_dim=ldim)
    elif model_name == "RKF-O":
        model = RKF_O(seed=seed, dim=dim, latent_dim=ldim)
    elif model_name == "RKF-F":
        model = RKF_F(seed=seed, dim=dim, latent_dim=ldim, hidden_dim=hiddendim)
    elif model_name == "RCKF":
        model = RCKF(seed=seed, dim=dim, latent_dim=ldim, hidden_dim=hiddendim, cov_dim=cov_dim)
        
    experiment_name = train_loop_KF(model, model_name, name, train, val, epochs=epochs, lr=lr, batch_size=batch_size,
                                    device=device, reg_lambda=reg_lambda)
    
    return experiment_name
