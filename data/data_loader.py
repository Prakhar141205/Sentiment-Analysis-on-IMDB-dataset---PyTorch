import torch
import torch.nn as nn


def create_collat_fun(pad_index):

    def collate_fun(batch):
        batch_ids = [s['ids'] for s in batch]
        batch_labels = [s['labels'] for s in batch ]
        batch_length = [s['length'] for s in batch]

        ## Pad input sequences to the maximum length in the batch 
        batch_ids_padded = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)

        # combining the existing tensors
        batch_lengths_tensor = torch.stack(batch_length)
        batch_labels_tensor = torch.stack(batch_labels)

        # construct the padded batch
        padded_batch = {
            'ids': batch_ids_padded,
            'length': batch_lengths_tensor,
            'labels': batch_labels_tensor
        }

        return padded_batch
    return collate_fun

def get_dataloader(dataset, batch_size, pad_index, shuffle: bool = True):

    collate_fun = create_collat_fun(pad_index)
    # create dataloader

    data_loader =  torch.utils.data.DataLoader(dataset = dataset, batch_size=batch_size, collate_fn= collate_fun, shuffle=shuffle)
    return data_loader