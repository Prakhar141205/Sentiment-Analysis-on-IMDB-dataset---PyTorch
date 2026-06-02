import re
import datasets
import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# class for loading or preprocessing the IMDB dataset

class IMDB_data:
    def __init__(self, test_size: float = 0.2, min_freq: int = 5, max_length: int = 300):
        self.test_size = test_size
        self.max_length = max_length
        self.min_freq = min_freq

        ## load the imdb dataset
        self.train_data, self.test_data = datasets.load_dataset("stanfordnlp/imdb", split=['train', 'test'])
        self.tokenizer = get_tokenizer('basic_english')
        
        # Track vocabulary indices
        self.unk_index = 0
        self.pad_index = 1

    def clean_data(self, text):
        # convert to the lowercase
        text = text.lower()
        # remove html tags using re library
        text = re.sub(r"<.*?>", "", text)
        # remove urls 
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        # remove special characters punctuation and numbers
        text = re.sub(r"[^a-z\s]", "", text)
        # remove extra white spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text
    
    def clean_tokenize_input(self, data):
        cleaned_text = self.clean_data(data['text'])
        tokens = self.tokenizer(cleaned_text)[:self.max_length]
        return {'tokens': tokens, 'length': len(tokens)}

    def get_vocab(self, train_data):
        special_tokens = ['<unk>', '<pad>']
        # Fixed the spelling typo from build_vacab_from_iterator
        vocab = build_vocab_from_iterator(
            train_data['tokens'],
            min_freq = self.min_freq,
            specials = special_tokens
        )

        self.unk_index = vocab['<unk>']
        self.pad_index = vocab['<pad>']

        vocab.set_default_index(self.unk_index)

        return vocab

    def get_data(self):

        ## tokenize the data
        # Keep these as self. attributes or create local mappings
        self.train_data = self.train_data.map(self.clean_tokenize_input)
        self.test_data = self.test_data.map(self.clean_tokenize_input)

        # split the train into validating and training sets
        train_valid_data = self.train_data.train_test_split(self.test_size)

        train_data = train_valid_data['train']
        val_data = train_valid_data['test']
        
        # CRITICAL FIX: Assign the instance attribute to a local variable 
        # so Python can resolve 'test_data' locally on line 74
        test_data = self.test_data 

        vocab = self.get_vocab(train_data=train_data)

        # numericalize the data
        train_data = train_data.map(
            lambda x: {"ids": vocab.lookup_indices(x["tokens"])}
        )

        # This will now work perfectly without UnboundLocalError
        test_data = test_data.map(
            lambda x: {"ids": vocab.lookup_indices(x["tokens"])}
        )

        val_data = val_data.map(
            lambda x: {"ids": vocab.lookup_indices(x["tokens"])}
        )

        ## convert the data to torch format
        train_data = train_data.with_format(type='torch', columns=['ids', 'label', 'length']) 
        val_data = val_data.with_format(type='torch', columns=['ids', 'label', 'length']) 
        test_data = test_data.with_format(type='torch', columns=['ids', 'label', 'length']) 

        return train_data, val_data, test_data, vocab, self.tokenizer

