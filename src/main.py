import torch

from init import init
from globals import *
from utils import get_dataloaders
from model import SentimentAnalyzer
from train_eval import evaluate

if __name__ == '__main__':
    # load the dataset and used vocab
    init(config)

    train_loader, valid_loader, test_loader = get_dataloaders(config['train'],
                                                config['valid'],
                                                config['test']
                                                )
    
    if net_config.mode == 'train':
        # create new instance of RNN

        model = SentimentAnalyzer(config['vocab'].
                                  net_config.hidden_dim,
                                  net_config.layers,
                                  net_config.dropout,
                                  net_config.bidirectional
                                  ).to(device)
        
    else :

        model = SentimentAnalyzer(config['vocab'].
                                  net_config.hidden_dim,
                                  net_config.layers,
                                  net_config.dropout,
                                  net_config.bidirectional
                                  ).to(device)
        
        model.load_state_dict(torch.load(net_config.pretrained_loc))

        with torch.no_grad():
            test_loss, test_acc = evaluate(model, test_loader, LOSS_FUN)

        print(f"Test Loss: {test_loss:.3} && Test Accuracy: {test_acc * 100 :.3f}")