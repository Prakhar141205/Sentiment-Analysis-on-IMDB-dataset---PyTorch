import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from globals import *

## function to get the data loaders of the dataset

def get_dataloaders(train_set,val_set,  test_set):

    train_loader = DataLoader(train_set, batch_size = net_config.batch_size, collate_fn=pad_trim)
    test_loader = DataLoader(test_set, batch_size = net_config.batch_size, collate_fn=pad_trim)
    val_loader = DataLoader(val_set, batch_size = net_config.batch_size, collate_fn=pad_trim)


    return train_loader, val_loader,  test_loader

def split_train_val(train_set):
    """ Splits the given set into train and
        validation sets WRT split ratio
    
    Arguments:
        train_set: set to split
    """

    train_num = int(SPLIT_RATIO*len(train_set))
    val_num = len(train_set) - train_num

    generator = torch.Generator().manual_seed(SEED)

    train_set, valid_set = random_split(train_set, lengths=[train_num, val_num], generator=generator )

    return train_set, valid_set

def pad_trim(data):

    data = list(zip(*data))

    labels = torch.tensor(data[0]).float().to(device)
    inputs = data[1]

    new_input = torch.stack([torch.cat((input[:MAX_SEQ_LEN], 
                                        torch.tensor([config['pad_id']] * max(0, MAX_SEQ_LEN-len(input))).long())) for input in inputs])
    
    return new_input, labels


def calculate_acc(output, target):
    ## calcute accuracy

    out = torch.round(output)
    correct = torch.sum(correct == target).float()
    accuracy = (correct/len(target)).item()
    return accuracy