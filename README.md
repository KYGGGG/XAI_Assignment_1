# XAI Assignment #1: Adversarial Attack

## Project Structure

- `training.py`: Training script for CIFAR-10 and MNIST using SimpleDLA.
- `test.py`: Evaluation script for adversarial attacks (FGSM, PGD).
- `plot.py`: Visualization script for training/testing loss and accuracy.
- `models/`: Contains the SimpleDLA model architecture.
- `utils.py`: Utility functions (e.g., progress bar).
- `checkpoint/`: Directory for saving model checkpoints.
- `data/`: Directory for dataset storage.
- `Result/`: Directory for logging results, saving adversarial samples, and CSV logs.

### 1. Training the Model

Train the SimpleDLA model on your chosen dataset:

```bash
# Train on CIFAR-10
python training.py --dataset cifar10

# Train on MNIST
python training.py --dataset mnist
```

Training logs (CSV) are saved in the `Result/` directory, and model checkpoints are saved in the `checkpoint/` directory.

### 2. Adversarial Attack Evaluation

Evaluate the model's robustness against adversarial attacks:

**Attack Modes:**
- `untargeted`: Untargeted Fast Gradient Sign Method (FGSM).
- `targeted`: Targeted FGSM.
- `pgd_targeted`: Targeted Projected Gradient Descent (PGD).

### Arguments for `test.py`

- `--mode`: Attack mode. Choices: `untargeted`, `targeted`, `pgd_targeted` (Default: `untargeted`).
- `--dataset`: Dataset to evaluate. Choices: `mnist`, `cifar10`, `all` (Default: `all`).
- `--target`: Target class index for targeted attacks (Default: `4`).
- `--iters`: Number of iterations for PGD attack (Default: `10`).
- `--alpha`: Step size for PGD attack (Default: `0.01`).

Results (CSV) and adversarial samples (Images) are saved in the `Result/` directory.

```bash
# Untargeted FGSM attack on both datasets
python test.py --mode untargeted 

# Targeted FGSM attack on CIFAR-10
python test.py --mode targeted --dataset cifar10 --target 4

# Targeted PGD attack on MNIST
python test.py --mode pgd_targeted --dataset mnist --target 7 --iters 40 --alpha 0.01
```

