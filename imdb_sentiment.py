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
import matplotlib.pyplot as plt

from data.data import IMDB_data
from data.data_loader import get_dataloader

from models.RNN_model import RNN
from models.LSTM_model import LSTM
from models.GRU_model import GRU


class SentimentAnalyzer:
    def __init__(self, config):
        print("CONFIG =", config)
        self.config = config
        self.setup_data()
        self.setup_model()
        self.initialize_weights()
        self.setup_pretrained_embeddings()
        self.setup_training_tools()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)


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

    
    def setup_pretrained_embeddings(self):
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

    def test_model(self):
        model_name = self.config['model']
        try:
            self.model.load_state_dict(torch.load(f"{model_name}.pt"))
            self.model.to(self.device)
            self.model.eval()

            test_loss, test_acc = self.evaluate_epoch(self.test_loader)

            print(f"Test loss {test_loss:.3f} && Test accuracy: {test_acc:.3f}")

        except Exception as e:
            print(f"An error occured while testing the model: {e}")


    def predict_sentiment(self, text):
        model_name = self.config['model']
        try:
            self.model.load_state_dict(torch.load(f"{model_name}.pt"))
            self.model.to(self.device)
            self.model.eval()


            ## tokenize and preprocess the input text
            tokens = self.tokenizer(text)
            token_ids = self.vocab.lookup_indices(tokens)
            token_length = torch.Longtensor(len(token_ids))

            ## prepare the input tensor

            input_tensor = torch.Longtensor(token_ids).unsqueeze(dim=0).to(self.device)
            # get the model predictions
            prediction = self.model(input_tensor, token_length).squeeze(dim=0)

            probability = torch.softmax(prediction, dim=-1)
            predicted_class_idx = prediction.argmax(dim=-1).item()
            predicted_probability = probability[predicted_class_idx].item()

            predicted_class = 'Negative' if predicted_probability == 0 else 'Positive'

            print(f"{model_name}\n\t Predicted Class: {predicted_class}\n\tProbability: {predicted_probability:.3f}")


        except Exception as e:
            print(f"An error occured while loading the model: {e}")

    @staticmethod
    def get_accuracy(predictions, labels):
        batch_size = predictions[0]
        predicted_class = predictions.argmax(dim=-1)
        correct_predictions = predicted_class.eq(labels).sum().item()

        accuracy = correct_predictions/batch_size
        return accuracy
    
    def count_parameters(self):
        return sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
    

    def visualize(self):
        data = {'text': self.train_data['text'], 'labels': self.train_data['labels']}
        df = pd.DataFrame(data)
        df['text'] = df['text'].apply(self.clean_text)

        # positive review word cloud

        positive_reviews = df[df['labels'] == 1]['text']

        self.generate_wordclouds(positive_reviews, "WordCloud for positive reviews")

        positive_reviews = df[df['labels'] == 0]['text']

        self.generate_wordclouds(positive_reviews, "WordCloud for positive reviews")


    @staticmethod
    def _clean_text(text):
        """Clean text by removing HTML tags, URLs, special characters, punctuation, and extra whitespace."""
        text = text.lower()
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    @staticmethod
    def generate_wordclouds(text, title):

        text = " ".join(text.astype(str))

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black'
        ).generate(text)

        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()
