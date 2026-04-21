import pickle
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_loaders(config, data_dir):
    
    train_data = []
    train_labels = []

    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])

    test_batch = unpickle(os.path.join(data_dir, f'test_batch'))

    X_train = torch.from_numpy(np.vstack(train_data)).float().reshape(-1, 3, 32, 32) / 255.0
    y_train = torch.from_numpy(np.array(train_labels)).long()

    X_test = torch.from_numpy(np.vstack(test_batch[b'data'])).float().reshape(-1, 3, 32, 32) / 255.0
    y_test = torch.from_numpy(np.array(test_batch[b'labels'])).long()


    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
            train_ds,
            batch_size = config.training.batch_size,
            shuffle=True,
            num_workers=2
            )

    test_loader = DataLoader(
            test_ds,
            batch_size=config.training.batch_size,
            shuffle=False,
            )

    return train_loader, test_loader
