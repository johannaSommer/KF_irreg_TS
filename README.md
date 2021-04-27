# The Kalman Filter for Irregular Time Series 

## About the Project
Time Series forecasting has become a pivotal part in many fields to extract information
from data collected over time. This task can become challenging when the data exhibits
irregularity in its measurements or dimensions. Rather than preprocessing the data and
losing information in the process, we propose an approach based on the Kalman Filter
to directly model irregularities. We present extensions to the Kalman Filter to establish
an expressive model for irregular time series, which, contrary to existing literature,
does not have to resort to expensive differential equations solvers to model time in a
continuous way. We provide results on both synthetic and medical data, that show
superior performance of the presented approach.

The baseline used for this project is the GRU-ODE-Bayes model, which can be found [here](https://github.com/edebrouwer/gru_ode_bayes)

## Project Structure
_datasets/preprocessing_ <br>
notebooks for preparing the MIMIC4 as well as synthetic 2D Ornstein-Uhlenbeck process data

_datasets/utils_ <br>
dataset class for irregular time series, collate functions for dataloaders, get-data utils

_models_ <br>
implementation for the discrete Kalman Filter, the continuous Kalman Filter with support for varing dimensions, 
the [Deep Kalman Filter](https://arxiv.org/abs/1511.05121) and the [Normalizing Kalman Filter](https://proceedings.neurips.cc/paper/2020/file/1f47cef5e38c952f94c5d61726027439-Paper.pdf)

_seml_  <br>
[seml](https://github.com/TUM-DAML/seml) files to execute HP tuning for the Kalman Filters

_training_ <br>
Kalman Filter training & evaluation utils

_notebooks_ <br>
includes examples for the discrete kalman filter usage, functionality to discretize the Kalman Filter predict ODEs and a check for normalizing the negative 
log-likeihood with varying observation dimensions
