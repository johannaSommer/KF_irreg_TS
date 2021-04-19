import numpy as np
import time
import torch
from datetime import datetime
from matplotlib import pyplot as plt
from datasets.utils import data_utils
from torch.utils.data import DataLoader
from models.KF_base import masked_likelihood


def train_loop_KF(model, model_name, dataset_name, train, val, epochs, lr, batch_size, device, reg_lambda):
    """
    Executes training loop and evaluation with the Kalman Filter

    Keyword arguments:
    model, model_name -- model to train (torch model, string)
    dataset_name -- name of the dataset to be trained on (string)
    train, val -- training and validation data (ITSDataset)
    lr -- learning rate (int)
    epochs, batch_size -- epochs, batch_size (int)
    device -- device the model is casted to (string)
    reg_lambda -- regularization factor (float)
    """
    if isinstance(train, data_utils.ITSDataset) and isinstance(val, data_utils.ITSDataset):
        dl_train = DataLoader(dataset=train, collate_fn=data_utils.collate_KF,
                              shuffle=True, batch_size=batch_size, num_workers=0)
        dl_val = DataLoader(dataset=val, collate_fn=data_utils.collate_KF,
                            shuffle=False, batch_size=batch_size, num_workers=0)
    else:
        raise ValueError('Data is not provided as ITS Dataset.')
        
    simulation_name = model_name + '_' + dataset_name + '_' + datetime.now().strftime("%d:%m-%H:%M:%S.%f")

    negll_batch, loss_batch, loss_epoch, negll_epoch, val_negll = [], [], [], [], []
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    val_metric_prev = 100
    for epoch in range(1, epochs+1):
        total_loss, total_negll = 0, 0
        start_time = time.time()

        for j, b in enumerate(dl_train):
            negll = model(b) 
            loss = negll + reg_lambda * regularize(model, model_name)
            loss.backward()
            optim.step()
            loss = loss.detach()
            total_loss += loss
            total_negll += negll
            negll_batch.append(negll.item())
            loss_batch.append(loss.item())
            optim.zero_grad()

        loss_epoch.append((total_loss / len(dl_train)).item())
        negll_epoch.append((total_negll / len(dl_train)).item())
        print(f'Average loss is {(total_loss / len(dl_train)):.4f} in Epoch {epoch}', flush=True)
        print(f'Average NegLL is {(total_negll / len(dl_train)):.4f} in Epoch {epoch}', flush=True)
        end_time = time.time()
        print(f'Epoch took {(end_time-start_time):.4f} seconds', flush=True)
        
        val_metric = validate_KF(model, dl_val, epoch, model_name)
        val_negll.append(val_metric)
        
        if val_metric < val_metric_prev:
            print(f'New highest validation metric reached ! : {val_metric}', flush=True)
            print('Saving Model', flush=True)
            file = "/path/" + simulation_name + "_MAX.pt"
            torch.save(model.state_dict(), file)
            val_metric_prev = val_metric

    return_dict = {
        'Name': simulation_name,
        'NegLL_batch': negll_batch,
        'Loss_batch': loss_batch,
        'NegLL_epoch': negll_epoch,
        'Loss_epoch': loss_epoch,
        'NegLL_val': val_negll
    }
    return return_dict


def validate_KF(model, val, step, model_name):
    """
    Evaluates Kalman Filter on validation data

    Keyword arguments:
    model -- model to train (torch model)
    val -- validation data (DataLoader)
    step -- current epoch that will be used as index for tensorboard (bool)
    model_name -- which type of Kalman Filter is evaluated (string)
    """
    with torch.no_grad():
        total_loss = 0
        mask = 0
        for k, batch in enumerate(val):
            x, P = model.initialize_params(len(batch['val_times']))
            pred_mu_1, pred_sigma_1, x, P, _, _, = model.iterate_cont_sequence(batch, x, P)

            concat_times = [np.concatenate((np.expand_dims(x[0][-1], axis=0), x[1])) for x in
                            list(zip(batch['times'], batch['val_times']))]
            pred_mu_2, pred_sigma_2 = model.forecasting(concat_times, batch, x, P)

            z_reord, mask_reord = [], []
            val_numobs = torch.Tensor([len(x) for x in batch['val_times']])
            for ind in range(0, int(torch.max(val_numobs).item())):
                idx = val_numobs > ind
                zero_tens = torch.Tensor([0])
                z_reord.append(batch['val_z'][(torch.cat((zero_tens, torch.cumsum(val_numobs, dim=0)))
                                               [:-1][idx] + ind).long()])
                mask_reord.append(batch['val_mask'][(torch.cat((zero_tens, torch.cumsum(val_numobs, dim=0)))
                                                     [:-1][idx] + ind).long()])
            
            syn_batch = dict()
            syn_batch['z'] = torch.cat(z_reord).to(model.device)
            syn_batch['mask'] = torch.cat(mask_reord).to(model.device)
            loss = masked_likelihood(syn_batch, pred_mu_2, pred_sigma_2, model.flow)
            loss = loss * syn_batch['mask'].sum()
            mask += syn_batch['mask'].sum()
            total_loss += loss
            
        total_loss /= mask
    print(f'Average validation NegLL is {total_loss.item():.4f}', flush=True)
    return total_loss.item()
        
        
def regularize(model, model_name):
    """
    Calculate regularization coefficient for KF
    """
    if model_name == "KF" or model_name == "NKF":
        parameters = torch.cat([torch.flatten(p) for p in list(model.parameters())])
    else:
        if model.use_ode:
            parameter_list = list(model.rnn_map.parameters()) + list(model.rnn_prop.parameters())
        else:
            parameter_list = list(model.rnn_map.parameters()) + list(model.param_func.parameters())
        if model.use_covs:
            parameter_list = parameter_list + list(model.rnn_update.parameters())
        parameters = torch.cat([torch.flatten(p) for p in parameter_list])
    return torch.sum(torch.pow(parameters, 2))


def showpred(z, val, times, timesall, pred_mu, pred_sigma, show_only_val):
    """
    Visualizes train and val predictions
    """
    val_0, val_1 = val
    timesall = timesall.cpu().numpy()
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    if show_only_val:
        mask_val = (torch.sum(val_1, dim=0) != 0)
        z = torch.stack(torch.chunk(z[mask_val.repeat(z.shape[0], 1)], chunks=z.shape[0], dim=0))
        val_1 = torch.stack(torch.chunk(val_1[mask_val.repeat(val_1.shape[0], 1)], chunks=val_1.shape[0], dim=0))
        pred_mu = torch.stack(torch.chunk(pred_mu[mask_val.repeat(pred_mu.shape[0], 1)], chunks=pred_mu.shape[0], dim=0)) 
        pred_sigma = torch.stack(torch.chunk(pred_sigma[mask_val.repeat(pred_sigma.shape[0], 1)],
                                             chunks=pred_sigma.shape[0], dim=0)).transpose(2, 1)
        pred_sigma = torch.stack(torch.chunk(pred_sigma[mask_val.repeat(pred_sigma.shape[0], 1)],
                                             chunks=pred_sigma.shape[0], dim=0)).transpose(2, 1)
    colors = ["green", "blue"]
    for dim in range(len(z[0])):
        obs = z[:, dim]
        obs[obs == 0] = float('nan')
        vals = val_1[:, dim]
        vals[vals == 0] = float('nan')
        ax.scatter(times, obs, color=colors[dim])
        ax.scatter(val_0, vals, color=colors[dim])
        ax.vlines(4, -1.5, 1.5, colors="red")
        ax.plot(timesall, pred_mu[:, dim], color=colors[dim])
        ax.fill_between(timesall, pred_mu[:, dim] - pred_sigma[:, dim, dim], pred_mu[:, dim] + pred_sigma[:, dim, dim],
                        alpha=0.2, color=colors[dim])
    ax.set_ylim([-1.5, 1.5])
    ax.grid(False)
    plt.tight_layout()
    plt.xlabel("time")
    plt.show()
