from datasets import load_dataset

# Load the IMDB dataset


# Explore the dataset splits
print(imdb_dataset)

# Access individual splits
train_data = imdb_dataset['train']
test_data = imdb_dataset['test']

# View a single example from the training data
print(train_data[0])
