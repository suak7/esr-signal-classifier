import numpy as np
from dataclasses import dataclass

@dataclass
class SignalConfig:
    # Configuration for signal generation parameters
    NUM_SAMPLES: int = 1000
    NUM_SAMPLES_PER_CLASS: int = 1000
    NOISE_STD_LOW: float = 0.1
    NOISE_STD_HIGH: float = 0.15
    
    # Class 0 (single peak) parameters
    CLASS0_PEAK_POS_RANGE: tuple = (0.35, 0.65)
    CLASS0_WIDTH_RANGE: tuple = (0.07, 0.15)
    CLASS0_AMP_RANGE: tuple = (0.8, 1.2)
    
    # Class 1 (double peak) parameters
    CLASS1_PEAK1_POS_RANGE: tuple = (0.25, 0.40)
    CLASS1_PEAK2_POS_RANGE: tuple = (0.60, 0.75)
    CLASS1_WIDTH_RANGE: tuple = (0.07, 0.15)
    CLASS1_AMP_RANGE: tuple = (0.8, 1.2)

config = SignalConfig()

def add_noise(signal: np.ndarray, std: float) -> np.ndarray:
    # Gaussian noise injection to signal
    noise = np.random.normal(0, std, size=signal.shape)
    return signal + noise

def generate_gaussian_peak(x: np.ndarray, position: float, width: float, amplitude: float) -> np.ndarray:
    # Single Gaussian peak
    return amplitude * np.exp(-((x - position)**2) / (2 * width**2))

def generate_class0_signal(x: np.ndarray) -> np.ndarray:
    # Single Gaussian-like peak signal
    peak_pos = np.random.uniform(*config.CLASS0_PEAK_POS_RANGE)
    width = np.random.uniform(*config.CLASS0_WIDTH_RANGE)
    amplitude = np.random.uniform(*config.CLASS0_AMP_RANGE)
    
    signal = generate_gaussian_peak(x, peak_pos, width, amplitude)
    return add_noise(signal, config.NOISE_STD_LOW)

def generate_class1_signal(x: np.ndarray) -> np.ndarray:
    # Double Gaussian peaks signal
    peak1_pos = np.random.uniform(*config.CLASS1_PEAK1_POS_RANGE)
    peak2_pos = np.random.uniform(*config.CLASS1_PEAK2_POS_RANGE)
    width1 = np.random.uniform(*config.CLASS1_WIDTH_RANGE)
    width2 = np.random.uniform(*config.CLASS1_WIDTH_RANGE)
    amp1 = np.random.uniform(*config.CLASS1_AMP_RANGE)
    amp2 = np.random.uniform(*config.CLASS1_AMP_RANGE)
    
    peak1 = generate_gaussian_peak(x, peak1_pos, width1, amp1)
    peak2 = generate_gaussian_peak(x, peak2_pos, width2, amp2)
    signal = peak1 + peak2
    return add_noise(signal, config.NOISE_STD_LOW)

def generate_class2_signal(x: np.ndarray) -> np.ndarray:
    # Irregular oscillating signal
    low_freq_amp = np.random.uniform(0.8, 1.2)
    high_freq_amp = np.random.uniform(0.4, 0.6)
    signal = low_freq_amp * np.sin(5 * x) + high_freq_amp * np.sin(8 * x)
    return add_noise(signal, config.NOISE_STD_HIGH)

def generate_dataset(output_dir: str = "data/"):
    # Generate and save synthetic ESR dataset
    x = np.linspace(0, 1, config.NUM_SAMPLES)
    
    signal_generators = {
        0: generate_class0_signal,
        1: generate_class1_signal,
        2: generate_class2_signal
    }
    
    signals = []
    labels = []
    
    for class_label, generator in signal_generators.items():
        for _ in range(config.NUM_SAMPLES_PER_CLASS):
            signal = generator(x)
            signals.append(signal.astype(np.float32))
            labels.append(class_label)
    
    # Convert to arrays and shuffle
    signals = np.stack(signals)
    labels = np.array(labels, dtype=np.int64)
    
    indices = np.arange(len(signals))
    np.random.shuffle(indices)
    signals = signals[indices]
    labels = labels[indices]

    np.save(f"{output_dir}signals.npy", signals)
    np.save(f"{output_dir}labels.npy", labels)
    
    print(f"Generated {len(signals)} signals saved to {output_dir}")
    return signals, labels

if __name__ == "__main__":
    generate_dataset()