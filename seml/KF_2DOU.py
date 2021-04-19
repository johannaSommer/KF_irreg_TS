from sacred import Experiment
import seml
from models.KF_wrapper import *
from training.KF_train_utils import train_loop_KF
from datasets.utils.get_data import get_OU_data


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
def run(seed, epochs, device, dim, batch_size, lr, ldim, hiddendim, model_name, reg_lambda):

    #  do your processing here
    train, val, name = get_OU_data()
    if model_name == "KF":
        model = KF(seed=seed, dim=dim, latent_dim=ldim)
    if model_name == "NKF":
        model = NKF(seed=seed, dim=dim, latent_dim=ldim)
    elif model_name == "RKF-F":
        model = RKF_F(seed=seed, dim=dim, latent_dim=ldim, hidden_dim=hiddendim)
        
    experiment_name = train_loop_KF(model, model_name, name, train, val, epochs=epochs, lr=lr, batch_size=batch_size,
                                    device=device, reg_lambda=reg_lambda)

    # the returned result will be written into the database
    return experiment_name
