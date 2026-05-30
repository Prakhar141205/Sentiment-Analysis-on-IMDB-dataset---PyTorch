import torch
import torchtext  #LP utilities
import spacy # text tokenizer
import os

from torchtext.data import get_tokenizer

from torch.utils.data import random_split
from torchtext.experimental.datasets import IMDB


from globals import *
from utils import split_train_val

def init(config):
    """
    using GLoVe embedding for the words that occur in the IMDB dataset 
    """

    if not os.path.dir('.data'):
        os.mkdir('.data')
    # extract the initial vocab from the IMDB dataset
    vocab = IMDB(data_select = 'train')[0].get_vocab()

    glove_vocab = torchtext.vocab.Vocab(
        counter=vocab.freqs,
        max_size = MAX_VOCAB_SIZE,
        min_freq = MIN_FREQ,
        vectors = torchtext.vocab.GloVe(name='6B')
    )


    tokenizer = get_tokenizer('spacy', 'en_core_web_sm')

    train_set, test_set = IMDB(tokenizer=tokenizer, vocab=glove_vocab)

    vocab = train_set.get_vocab()

    pad_id = vocab['<pad>']

    train_set, valid_set = split_train_val(train_set)

    config['train']=train_set
    config['test']=test_set
    config['valid']=valid_set
    config['vocab']=vocab
    config['pad_id']=pad_id