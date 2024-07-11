import wandb
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from typing import List, Union, Dict
from IPython.display import clear_output
import argparse

from dataset.dataset import create_datasets, create_dataloaders
from models import Swin, SwinJoint
from trainer.train import Trainer


class Logger:
    def __init__(self, project_name: str, run_name: str):
        self.project_name = project_name
        self.run_name = run_name
        self.run = wandb.init(project=project_name, name=run_name)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        wandb.log(metrics, step=step)

    def log_parameters(self, model: nn.Module):
        wandb.watch(model)

    def finish(self):
        self.run.finish()

def main(args):
    logger = Logger(project_name=args.project_name, run_name=args.run_name)
    train_dataset, test_dataset = create_datasets(args.root,
                                                  min_paintings=50,
                                                  transforms = T.Compose([
                                                    T.RandomResizedCrop(384),
                                                    T.RandomHorizontalFlip(p=0.1),
                                                    T.ToTensor(),
                                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                    ]))
    train_loader, test_loader = create_dataloaders(train_dataset=train_dataset,
                                                    test_dataset=test_dataset,
                                                    batch_size=args.batch_size)
    num_classes_styles, num_classes_artists = len(train_dataset.style2idx), len(train_dataset.artist2idx)
    model = SwinJoint(num_classes_styles, num_classes_artists)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    trainer = Trainer(model,
                      criterion, 
                      optimizer, 
                      train_loader, 
                      test_loader, 
                      lr_scheduler, 
                      alpha=args.alpha, 
                      device=args.device, 
                      verbose=args.verbose)

    logger.log_parameters(model)
    trainer.train(num_epochs=args.num_epochs)


    final_train_metrics = trainer.evaluate_epoch(train_loader)
    final_test_metrics = trainer.evaluate_epoch(test_loader)
    logger.log_metrics(final_train_metrics, step=args.num_epochs)
    logger.log_metrics(final_test_metrics, step=args.num_epochs)

    logger.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments with wandb logging')
    parser.add_argument('--root', type=str, default='./', help='Path to data')
    parser.add_argument('--project_name', type=str, default='default_project', help='Name of the wandb project')
    parser.add_argument('--run_name', type=str, default='experiment_run', help='Name of the current run')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--lr_step_size', type=int, default=5, help='Step size for learning rate scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma factor for learning rate scheduler')
    parser.add_argument('--alpha', type=float, default=1, help='Weight for artist loss in the combined loss')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run training on (cuda or cpu)')
    parser.add_argument('--verbose', action='store_true', help='Print verbose outputs during training')

    args = parser.parse_args()

    main(args)
