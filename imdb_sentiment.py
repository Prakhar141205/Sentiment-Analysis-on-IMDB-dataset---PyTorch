import collections
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
from torch.utils.tensorboard import SummaryWriter
from wordcloud import WordCloud

from data.data import IMDB_data
from data.data_loader import get_dataloader

from models.RNN_model import RNN
from models.LSTM_model import LSTM
from models.GRU_model import GRU


class SentimentAnalyzer:
    def __init__(self, config):
        self.config = config
        self.setup_data()
        self.setup_model()
        self.initialize_weights()
        self.setup_pretrained_embeddings()
        self.setup_training_tools()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to


    def setup_model(self):
        model_name = self.config['model']
        model_config = self.config['model_config']

        ## add common parameters to the model parameters

        model_config.update({'vocab_size': len(self.vocab), 'pad_index': self.pad_index})

        ## initilize the model

        if model_name == 'RNN':
            self.model = RNN(**model_config)
        elif model_name == 'LSTM':
            self.model = LSTM(**model_config)
        elif model_name == 'GRU':
            self.model = GRU(**model_config)
        else:
            raise ValueError("Model can be RNN, LSTM, GRU")
        
    def setup_data(self):
        data_config = self.config['data_config']
        batch_size = self.config['batch_size']
        data = IMDB_data(**data_config)

        self.train_data, self.val_data, self.test_data, self.vocab, self.tokenizer = (data.get_data())
        self.pad_index = data.pad_index

        self.train_loader = get_dataloader(self.train_data, batch_size, self.pad_index, shuffle=True)
        self.valid_loader = get_dataloader(self.val_data, batch_size, self.pad_index)
        self.test_loader = get_dataloader(self.test_data, batch_size, self.pad_index)

    def initialize_weights(self):
        if self.config.get("initialize_weights", False):
            self.model.apply(self.init_weights)

    
    def setup_pretrained_embedding(self):
        vectors = torchtext.vocab.GloVe()
        pretrained_embedding = vectors.get_vecs_by_tokens(self.vocab.get_itos())
        self.model.embedding.weight.data = pretrained_embedding

    def setup_training_tools(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self, model_layer):

        if isinstance(model_layer, nn.Linear()):
            nn.init.xavier_normal_(model_layer.weight)
            nn.init.zeros_(model_layer.bias)
        elif isinstance(model_layer, nn.LSTM):
            for name, param  in model_layer.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    nn.init.orthogonal_param()
        elif isinstance(model_layer, nn.GRU):
            for name, param in m.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    nn.init.orthogonal_(param) 



    def train_epoch(self, dataloader, epoch):
        ## Train the model for one epoch
        self.model.train()
        epoch_losses = []
        epoch_acc = []

        for batch in tqdm.tqdm(dataloader, desc=f'Training loss {epoch+1}'):
            ids = batch['ids'].to(self.device)
            lengths = batch['length']
            label = batch['label'].to(self.device)
            pred = self.model(ids, lengths)
            loss = self.criterion(pred, label)
            accuracy = self.get_accuracy(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.item())
            epoch_acc.append(accuracy)

        return torch.mean(torch.tensor(epoch_losses)), torch.mean(torch.tensor(epoch_acc))
    
    def evaluate_epoch(self, dataloader, epoch):
        ## Train the model for one epoch
        self.model.eval()
        epoch_losses = []
        epoch_acc = []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc=f'Training loss {epoch+1}'):
                ids = batch['ids'].to(self.device)
                lengths = batch['length']
                label = batch['label'].to(self.device)
                pred = self.model(ids, lengths)
                loss = self.criterion(pred, label)
                accuracy = self.get_accuracy(pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
                epoch_acc.append(accuracy)

        return torch.mean(torch.tensor(epoch_losses)), torch.mean(torch.tensor(epoch_acc))
    
    def train_and_evaluate(self):
        """Train and evaluate the model for specified number of epochs"""
        num_epochs = self.config['num_epochs']
        best_valid_loss = float('inf')
        metrics = collections.defaultdict(list)
        model_name = self.config['model']

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(self.train_loader, epoch)
            valid_loss, valid_acc = self.evaluate_epoch(self.test_loader, epoch)
            metrics['train_losses'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            metrics['valid_loss'].append(valid_loss)
            metrics['valid_acc'].append(valid_acc)

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Loss/Validation", valid_loss, epoch)
            self.writer.add_scalar("Accuracy/Validation", valid_acc, epoch)

            print(f"epoch= {epoch+1}")
            print(f"Train Loss = {train_loss} && Train accuracy = {train_acc}")
            print(f"Validation Loss = {valid_loss} && Validation accuracy = {valid_acc}")
            