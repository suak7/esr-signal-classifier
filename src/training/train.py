import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from models.signal_classifier import SignalClassifier
from data.dataloader import ESRDataModule

class Trainer:
    # Model training and evaluation
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        learning_rate: float = 0.001,
        num_epochs: int = 20,
        device: str = None,
        output_dir: str = "outputs/"
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train_loss_history = []
        self.test_accuracy_history = []
        self.test_loss_history = []
        
    def train_epoch(self) -> float:
        # Train for one epoch
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for signals, labels in self.train_loader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self) -> Tuple[float, float]:
        # Evaluate model on test set
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for signals, labels in self.test_loader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate accuracy
                _, predictions = torch.max(outputs, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / num_batches
        accuracy = 100.0 * correct_predictions / total_samples
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, List[float]]:
        # Full training loop
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            self.train_loss_history.append(train_loss)

            test_loss, test_accuracy = self.evaluate()
            self.test_loss_history.append(test_loss)
            self.test_accuracy_history.append(test_accuracy)

            print(f"Epoch {epoch+1}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_accuracy:.2f}%")

        self.save_results()
        
        return {
            "train_loss": self.train_loss_history,
            "test_loss": self.test_loss_history,
            "test_accuracy": self.test_accuracy_history
        }
    
    def save_results(self):
        model_path = self.output_dir / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        np.save(self.output_dir / "train_loss_history.npy", self.train_loss_history)
        np.save(self.output_dir / "test_accuracy_history.npy", self.test_accuracy_history)
        np.save(self.output_dir / "test_loss_history.npy", self.test_loss_history)

        self.plot_training_curves()
    
    def plot_training_curves(self):
        # Generate and save training visualization
        epochs = range(1, self.num_epochs + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(epochs, self.train_loss_history, label='Train Loss', color='blue')
        ax1.plot(epochs, self.test_loss_history, label='Test Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Test Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.test_accuracy_history, label='Test Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Test Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300)
        print(f"Training curves saved to {self.output_dir / 'training_curves.png'}")
        plt.close()

def main():
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20

    data_module = ESRDataModule(batch_size=BATCH_SIZE)
    train_loader, test_loader = data_module.prepare_dataloaders()

    model = SignalClassifier(
        input_size=1000,
        hidden_sizes=(256, 128),
        num_classes=3,
        dropout_rate=0.2
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS
    )
    
    metrics = trainer.train()
    
    print(f"\nFinal Results:")
    print(f"Final Train Loss: {metrics['train_loss'][-1]:.4f}")
    print(f"Final Test Accuracy: {metrics['test_accuracy'][-1]:.2f}%")

if __name__ == "__main__":
    main()