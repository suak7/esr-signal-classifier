from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass
class PathConfig:
    """File paths and directory structure"""
    # Base directories
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
    MODEL_DIR: Path = OUTPUT_DIR / "models"
    PLOTS_DIR: Path = OUTPUT_DIR / "plots"
    
    # Data files
    SIGNALS_FILE: Path = DATA_DIR / "signals.npy"
    LABELS_FILE: Path = DATA_DIR / "labels.npy"
    
    # Output files
    MODEL_CHECKPOINT: Path = MODEL_DIR / "model.pth"
    TRAIN_LOSS_FILE: Path = OUTPUT_DIR / "train_loss_history.npy"
    TEST_LOSS_FILE: Path = OUTPUT_DIR / "test_loss_history.npy"
    TEST_ACC_FILE: Path = OUTPUT_DIR / "test_accuracy_history.npy"
    TRAINING_CURVES_PLOT: Path = PLOTS_DIR / "training_curves.png"
    
    def create_directories(self):
        # Create all necessary directories
        self.DATA_DIR.mkdir(exist_ok=True, parents=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        self.MODEL_DIR.mkdir(exist_ok=True, parents=True)
        self.PLOTS_DIR.mkdir(exist_ok=True, parents=True)


@dataclass
class DataConfig:
    # Signal parameters
    SIGNAL_LENGTH: int = 1000  # Number of points in each signal
    NUM_SAMPLES_PER_CLASS: int = 1000  # Samples to generate per class
    NUM_CLASSES: int = 3
    
    # Noise parameters
    NOISE_STD_LOW: float = 0.1
    NOISE_STD_HIGH: float = 0.15
    
    # Class 0: Single Gaussian peak
    CLASS0_PEAK_POS_RANGE: Tuple[float, float] = (0.35, 0.65)
    CLASS0_WIDTH_RANGE: Tuple[float, float] = (0.07, 0.15)
    CLASS0_AMP_RANGE: Tuple[float, float] = (0.8, 1.2)
    
    # Class 1: Double Gaussian peaks
    CLASS1_PEAK1_POS_RANGE: Tuple[float, float] = (0.25, 0.40)
    CLASS1_PEAK2_POS_RANGE: Tuple[float, float] = (0.60, 0.75)
    CLASS1_WIDTH_RANGE: Tuple[float, float] = (0.07, 0.15)
    CLASS1_AMP_RANGE: Tuple[float, float] = (0.8, 1.2)
    
    # Class 2: Irregular oscillating signal
    CLASS2_LOW_FREQ_AMP_RANGE: Tuple[float, float] = (0.8, 1.2)
    CLASS2_HIGH_FREQ_AMP_RANGE: Tuple[float, float] = (0.4, 0.6)
    CLASS2_LOW_FREQ: int = 5
    CLASS2_HIGH_FREQ: int = 8
    
    # Data split
    TRAIN_SPLIT: float = 0.8
    RANDOM_SEED: int = 42


@dataclass
class ModelConfig:
    # Neural network architecture parameters
    INPUT_SIZE: int = 1000  # Must match DataConfig.SIGNAL_LENGTH
    HIDDEN_SIZES: Tuple[int, ...] = (256, 128)
    NUM_CLASSES: int = 3  # Must match DataConfig.NUM_CLASSES
    DROPOUT_RATE: float = 0.2
    
    def validate(self, data_config: DataConfig):
        # Ensure model config matches data config
        assert self.INPUT_SIZE == data_config.SIGNAL_LENGTH, \
            f"Model input size ({self.INPUT_SIZE}) must match signal length ({data_config.SIGNAL_LENGTH})"
        assert self.NUM_CLASSES == data_config.NUM_CLASSES, \
            f"Model num classes ({self.NUM_CLASSES}) must match data num classes ({data_config.NUM_CLASSES})"


@dataclass
class TrainingConfig:
    # Optimization
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 20
    
    # Device
    DEVICE: str = "auto"  # "auto", "cuda", "cpu", or "mps"
    
    # Regularization 
    WEIGHT_DECAY: float = 0.0

    SAVE_BEST_MODEL: bool = True
    SAVE_LAST_MODEL: bool = True


@dataclass
class VisualizationConfig:
    DPI: int = 300
    FIGURE_SIZE: Tuple[int, int] = (12, 4)

    TRAIN_LOSS_COLOR: str = "blue"
    TEST_LOSS_COLOR: str = "red"
    TEST_ACC_COLOR: str = "green"

    PLOT_STYLE: str = "default"  # "default", "seaborn", "ggplot", etc.
    GRID_ALPHA: float = 0.3

    LOSS_YLABEL: str = "Loss"
    ACCURACY_YLABEL: str = "Accuracy (%)"
    EPOCH_XLABEL: str = "Epoch"


class Config:
    """
    Example:
        from config import Config
        cfg = Config()
        model = SignalClassifier(
            input_size=cfg.model.INPUT_SIZE,
            hidden_sizes=cfg.model.HIDDEN_SIZES
        )
    """
    
    def __init__(self):
        self.paths = PathConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.viz = VisualizationConfig()

        self.model.validate(self.data)

        self.paths.create_directories()
    
    def print_config(self):
        # Print all configuration parameters
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        
        print("\n[Paths]")
        for key, value in vars(self.paths).items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
        
        print("\n[Data]")
        for key, value in vars(self.data).items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
        
        print("\n[Model]")
        for key, value in vars(self.model).items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
        
        print("\n[Training]")
        for key, value in vars(self.training).items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
        
        print("\n[Visualization]")
        for key, value in vars(self.viz).items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
        
        print("=" * 60)

if __name__ == "__main__":
    cfg = Config()
    cfg.print_config()