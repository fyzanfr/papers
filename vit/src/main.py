import argparse
import os
import numpy as np
from data_loading import unpickle, get_loaders
from config import get_miniVit_config
from model import Embeddings

def main():

    parser = argparse.ArgumentParser(description='Training the miniViT')

    parser.add_argument('--data_dir', type=str, default='data/cifar-10-batches-py', 
                        help ='Path to the extracted CIFAR-10 batches')
    parser.add_argument('--epochs', type=int, default=None, 
                        help ='Override the config epoch if needed')

    args = parser.parse_args()
    
    cfg = get_miniVit_config()

    if args.epochs:
        cfg.training.epochs = args.epochs

    print(f"Loading Data from {args.data_dir}")

    if not os.path.exists(args.data_dir):
        print(f"Error:{args.data_dir} not found..")
        return

    train_loader, test_loader = get_loaders(cfg, args.data_dir)

    in_phase = Embeddings(cfg)
    images, labels = next(iter(train_loader))
    emb_output = in_phase(images)
    print(f'Embeddings Shape: {emb_output.shape}')

if __name__ == '__main__':
    main()
