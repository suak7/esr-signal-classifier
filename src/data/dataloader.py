import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import Tuple

class ESRDataModule:
    # Loading and preparing ESR signal dataset
    
    def __init__(
        self, 
        data_path: str = "data/signals.npy",
        labels_path: str = "data/labels.npy",
        batch_size: int = 32,
        train_split: float = 0.8,
        seed: int = 42
    ):
        self.data_path = data_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.train_split = train_split
        self.seed = seed
        
        torch.manual_seed(seed)
        
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load signals and labels from disk
        signals = np.load(self.data_path)
        labels = np.load(self.labels_path)
        
        signals_tensor = torch.tensor(signals, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return signals_tensor, labels_tensor
    
    def prepare_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        # Create train and test dataloaders
        signals, labels = self.load_data()
        dataset = TensorDataset(signals, labels)
        
        total_samples = len(dataset)
        train_size = int(self.train_split * total_samples)
        test_size = total_samples - train_size
        
        train_dataset, test_dataset = random_split(
            dataset, 
            [train_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        print(f"Train samples: {train_size}, Test samples: {test_size}")
        return train_loader, test_loader

# For backwards compatibility with current train.py
def get_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    dm = ESRDataModule(batch_size=batch_size)
    return dm.prepare_dataloaders()

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Number of batches: {len(train_loader)}")