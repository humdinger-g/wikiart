import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from typing import List, Union, Dict
from IPython.display import clear_output

class Trainer:
    def __init__(self, model: nn.Module, 
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 lr_scheduler: optim.lr_scheduler.LRScheduler,
                 alpha: float = 1,
                 device: str = 'cpu',
                 verbose: bool = False,
                 joint: bool = True):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.alpha = alpha
        self.device = device
        self.verbose = verbose
        self.joint = joint

        self.num_styles = len(self.train_loader.dataset.style2idx)
        self.num_artists = len(self.train_loader.dataset.artist2idx)
        
        self.model.to(self.device)

    def train_epoch(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary containing average losses and accuracies on train data
        """
        self.model.train()
        total_loss_artist, total_loss_style = 0, 0
        correct_styles, correct_artists = 0, 0
        num_batches = len(self.train_loader)
        
        for inputs, style_labels, artist_labels in tqdm(self.train_loader, desc='Training', leave=False):
            inputs, style_labels, artist_labels = inputs.to(self.device), style_labels.to(self.device), artist_labels.to(self.device)
            self.optimizer.zero_grad()
            
            if self.joint:
                style_pred, artist_pred, joint_pred = self.model(inputs)
                loss_style = self.criterion(style_pred, style_labels)
                loss_artist = self.criterion(artist_pred, artist_labels)
                joint_labels = style_labels * self.num_artists + artist_labels
                joint_loss = self.criterion(joint_pred.view(-1, self.num_styles * self.num_artists), joint_labels)
                loss = loss_style + self.alpha * loss_artist + joint_loss
            else:
                style_pred, artist_pred = self.model(inputs)
                loss_style = self.criterion(style_pred, style_labels)
                loss_artist = self.criterion(artist_pred, artist_labels)
                loss = loss_style + self.alpha * loss_artist
            
            loss.backward()
            self.optimizer.step()
            
            total_loss_style += loss_style.item()
            total_loss_artist += loss_artist.item()
            
            predicted_styles = torch.argmax(style_pred, 1)
            predicted_artists = torch.argmax(artist_pred, 1)
            
            correct_styles += (predicted_styles == style_labels).sum().item()
            correct_artists += (predicted_artists == artist_labels).sum().item()
        
        avg_loss_artist = total_loss_artist / len(self.train_loader.dataset)
        avg_loss_style = total_loss_style / len(self.train_loader.dataset)
        avg_acc_artist = correct_artists / len(self.train_loader.dataset)
        avg_acc_style = correct_styles / len(self.train_loader.dataset)
        
        return {
            'avg_loss_artist': avg_loss_artist,
            'avg_loss_style': avg_loss_style,
            'avg_acc_artist': avg_acc_artist,
            'avg_acc_style': avg_acc_style
        }

    def evaluate_epoch(self) -> Dict[str, float]:
        """
        Returns:
            Dictionary containing average losses and accuracies on test data
        """
        self.model.eval()
        total_loss_artist, total_loss_style = 0, 0
        correct_styles, correct_artists = 0, 0
        num_batches = len(self.test_loader)
        
        with torch.no_grad():
            for inputs, style_labels, artist_labels in tqdm(self.test_loader, desc='Evaluating', leave=False):
                inputs, style_labels, artist_labels = inputs.to(self.device), style_labels.to(self.device), artist_labels.to(self.device)
                
                if self.joint:
                    style_pred, artist_pred, _ = self.model(inputs)
                else:
                    style_pred, artist_pred = self.model(inputs)
                
                loss_style = self.criterion(style_pred, style_labels)
                loss_artist = self.criterion(artist_pred, artist_labels)
                
                total_loss_style += loss_style.item()
                total_loss_artist += loss_artist.item()
                
                predicted_styles = torch.argmax(style_pred, 1)
                predicted_artists = torch.argmax(artist_pred, 1)
                
                correct_styles += (predicted_styles == style_labels).sum().item()
                correct_artists += (predicted_artists == artist_labels).sum().item()
        
        avg_loss_artist = total_loss_artist / len(self.test_loader.dataset)
        avg_loss_style = total_loss_style / len(self.test_loader.dataset)
        avg_acc_artist = correct_artists / len(self.test_loader.dataset)
        avg_acc_style = correct_styles / len(self.test_loader.dataset)
        
        return {
            'avg_loss_artist': avg_loss_artist,
            'avg_loss_style': avg_loss_style,
            'avg_acc_artist': avg_acc_artist,
            'avg_acc_style': avg_acc_style
        }

    def train(self, num_epochs: int, plot_interval: int = 1) -> None:
        """
        Args:
            num_epochs (int): Number of train epochs
            plot_interval (int, optional): Interval for plotting losses and accuracies
        """
        train_artist_losses = []
        test_artist_losses = []
        train_style_losses = []
        test_style_losses = []
        train_artist_accuracies = []
        test_artist_accuracies = []
        train_style_accuracies = []
        test_style_accuracies = []
        
        for epoch in range(num_epochs):
            
            train_metrics = self.train_epoch()
            test_metrics = self.evaluate_epoch()
            self.lr_scheduler.step()
            
            train_artist_losses.append(train_metrics['avg_loss_artist'])
            test_artist_losses.append(test_metrics['avg_loss_artist'])
            train_style_losses.append(train_metrics['avg_loss_style'])
            test_style_losses.append(test_metrics['avg_loss_style'])
            
            train_artist_accuracies.append(train_metrics['avg_acc_artist'])
            test_artist_accuracies.append(test_metrics['avg_acc_artist'])
            train_style_accuracies.append(train_metrics['avg_acc_style'])
            test_style_accuracies.append(test_metrics['avg_acc_style'])
            
            if (epoch + 1) % plot_interval == 0:
                self.plot_losses(train_artist_losses, test_artist_losses,
                                 train_style_losses, test_style_losses,
                                 train_artist_accuracies, test_artist_accuracies,
                                 train_style_accuracies, test_style_accuracies)
    
    def plot_losses(self, train_artist_losses: List[float], test_artist_losses: List[float],
                    train_style_losses: List[float], test_style_losses: List[float],
                    train_artist_accuracies: List[float], test_artist_accuracies: List[float],
                    train_style_accuracies: List[float], test_style_accuracies: List[float]):

        clear_output(wait=True)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,8))
        
        axs[0, 0].plot(train_artist_losses, label='Train')
        axs[0, 0].plot(test_artist_losses, label='Test')
        axs[0, 0].set_title('Artist Losses')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        
        axs[0, 1].plot(train_artist_accuracies, label='Train')
        axs[0, 1].plot(test_artist_accuracies, label='Test')
        axs[0, 1].set_title('Artist Accuracies')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].legend()
        
        axs[1, 0].plot(train_style_losses, label='Train')
        axs[1, 0].plot(test_style_losses, label='Test')
        axs[1, 0].set_title('Style Losses')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()
        
        axs[1, 1].plot(train_style_accuracies, label='Train')
        axs[1, 1].plot(test_style_accuracies, label='Test')
        axs[1, 1].set_title('Style Accuracies')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].legend()
        
        for ax in axs.flat:
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
