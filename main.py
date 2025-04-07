import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from optimizer import ScheduledOptim
import random as rd
import numpy as np
from dataset import CustomTextDataset
from trainer import BERTTrainer
from model import BERT
import json
from config import Config

def set_seeds(config):
    """
    Set random seeds for reproducibility.

    Args:
        config (Config): Configuration object containing parameters.
    """
    rd.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if torch.cuda.is_available() and config.with_cuda:
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(config):
    """
    Main function to run BERT training.

    Args:
        config (Config): Configuration object containing parameters.
    """
    # Set random seeds
    set_seeds(config)

    print("Loading Train Dataset...")

    # Load training dataset
    train_dataset = CustomTextDataset(config, 'train')

    # Load test dataset if provided
    test_dataset = CustomTextDataset(config, 'test')

    # Initialize BERT model
    bert = BERT(config)

    # Create data loaders
    train_data_loader = iter(DataLoader(train_dataset, batch_size=None, worker_init_fn=np.random.seed(config.seed)))
    test_data_loader = iter(DataLoader(test_dataset, batch_size=None, worker_init_fn=np.random.seed(config.seed))) if test_dataset is not None else None

    # Initialize optimizer and scheduler
    optim = Adam(bert.parameters(), lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)
    optim_schedule = ScheduledOptim(config, optim)

    # Initialize BERT trainer
    trainer = BERTTrainer(config, bert, optim_schedule, train_data_loader, test_data_loader)

    # Training loop
    for epoch in range(config.epochs):
        # Train the model
        trainer.train(epoch)

        # Save the model
        trainer.save(epoch)

        # Test the model if test data is available
        if test_data_loader is not None:
            trainer.test(epoch)

if __name__ == "__main__":
    # Load configuration

    with open('args.json', 'r') as fp:
        args = json.load(fp)

    config = Config(
        prop=args['prop'],
        seq_len=args['seq_len'],
        delimiters=args['delimiters'],
        lower_case=args['lower_case'],
        buffer_size=args['buffer_size'],
        shuffle=args['shuffle'],
        data_dir=args['data_dir'],
        hidden_size=args['hidden_size'],
        vocab_size=args['vocab_size'],
        hidden_dropout_prob=args['hidden_dropout_prob'],
        num_heads=args['num_heads'],
        num_blocks=args['num_blocks'],
        final_dropout_prob=args['final_dropout_prob'],
        n_warmup_steps=args['n_warmup_steps'],
        weight_decay=args['weight_decay'],
        lr=args['lr'],
        betas=args['betas'],
        cuda_devices=args['cuda_devices'],
        with_cuda=args['with_cuda'],
        log_freq=args['log_freq'],
        batch_size=args['batch_size'],
        save_path=args['save_path'],
        seed=args['seed'],
        epochs=args['epochs']
    )


    # Run BERT training
    main(config)