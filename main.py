import torch
import numpy as np

from utils.load_config_file import load_config_file
## set the seed for reproducibility

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


## load the config for different models

rnn_config = load_config_file("./config/RNN_config.yaml")
lstm_config = load_config_file("./config/LSTM_config.yaml")
gru_config = load_config_file("./config/GRU_config.yaml")