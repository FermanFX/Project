"""
PART 3.2: TRAINING & OPTIMIZATION (5 points)
=============================================
COMPLETE TRAINING CODE - NO TODO NEEDED
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)


class Trainer:
    """
    Training manager for neural network.
    """
    
    def __init__(self, model, criterion=None, optimizer=None, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else optim.Adam(model.parameters(), lr=0.001)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def create_dataloader(self, X, y, batch_size=32, shuffle=True):
        """Create PyTorch DataLoader."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            self.optimizer.zero_grad()
            
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=True):
        """
        Complete training loop.
        """
        train_loader = self.create_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self.create_dataloader(X_val, y_val, batch_size, shuffle=False)
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.train_losses, self.val_losses
    
    def predict(self, X):
        """Make predictions."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def plot_training_history(self, save_path='figures/training_history.png'):
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2.plot(epochs, self.train_accs, 'b-', label='Train Acc')
        ax2.plot(epochs, self.val_accs, 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Training history saved: {save_path}")


def learning_rate_finder(model, train_loader, start_lr=1e-7, end_lr=1, iterations=100):
    """
    Learning rate finder to optimal learning rate.
    """
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.01)
    
    lrs = []
    losses = []
    
    for i in range(iterations):
        optimizer = optim.Adam(model.parameters(), lr=start_lr * (end_lr/start_lr) ** (i/iterations))
        criterion = nn.CrossEntropyLoss()
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        lrs.append(start_lr * (end_lr/start_lr) ** (i/iterations))
        losses.append(loss.item())
    
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(alpha=0.3)
    plt.savefig('figures/learning_rate_finder.png', dpi=150)
    plt.show()
    
    return lrs, losses
