import torch
import numpy as np
from imdb_sentiment import SentimentAnalyzer
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

# initilize the model

rnn_model = SentimentAnalyzer(rnn_config)
lstm_model = SentimentAnalyzer(lstm_config)
gru_model = SentimentAnalyzer(gru_config)

rnn_params=rnn_model.count_parameters()
lstm_params=lstm_model.count_parameters()
gru_params=gru_model.count_parameters()

print(f"The number of params in rnn model are {rnn_params}")
print(f"The number of params in rnn model are {lstm_params}")
print(f"The number of params in rnn model are {gru_params}")

print("Training and Evaluating the RNN model")
rnn_model.train_and_evaluate()
rnn_model.test_model()


print("Training and Evaluating the LSTM model")
lstm_model.train_and_evaluate()
lstm_model.test_model()


print("Training and Evaluating the GRU model")
gru_model.train_and_evaluate()
gru_model.test_model()


review1 = (
    "It's a good movie (certainly Garfield's best) that shouldn't be taken seriously. "
    "You just have to have a good time with your family and enjoy it. There are some jokes "
    "that are more childish, but what can you do to it like that is the original children's series. "
    "There are also the odd jokes that the older ones can enjoy. The film leaves you with a beautiful message "
    "that many of us should take into account. It's not a great adaptation, because it takes several elements "
    "that aren't the same as the original series but that's okay, because you replace them with other elements "
    "that make the movie feel refreshing and different. Like any movie, not everything is good and this is no exception "
    "since there are some characters that I didn't like so much, such as the villain, which in my opinion is the weakest "
    "part of the film. It's a good movie to have a good time with the family."
)

# Predict sentiment for the example reviews using all models
print("\nPredicting sentiment for review1...")
print("RNN model prediction:")
rnn_model.predict_sentiment(review1)
print("LSTM model prediction:")
lstm_model.predict_sentiment(review1)
print("GRU model prediction:")
gru_model.predict_sentiment(review1)
