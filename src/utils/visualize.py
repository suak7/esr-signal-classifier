import numpy as np
import matplotlib.pyplot as plt

# visualize both training loss and test accuracy curves
def plot_training_curves():
    loss_history = np.load("loss_history.npy")
    accuracy_history = np.load("accuracy_history.npy")

    plt.figure()
    plt.plot(loss_history, label='Training Loss', color='blue')
    plt.plot(accuracy_history, label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Performance')
    plt.legend()
    plt.savefig('training_performance.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_training_curves()