import csv
import argparse
import os
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is not installed. Please run 'pip install matplotlib'.")
    exit(1)

def plot_metrics(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    epochs, train_losses, train_accs, test_losses, test_accs = [], [], [], [], []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Some logs might have different naming. Change these keys if needed.
                epochs.append(int(row.get('epoch', len(epochs))))
                if 'train_loss' in row and row['train_loss'].strip(): train_losses.append(float(row['train_loss']))
                if 'train_acc' in row and row['train_acc'].strip(): train_accs.append(float(row['train_acc']))
                if 'test_loss' in row and row['test_loss'].strip(): test_losses.append(float(row['test_loss']))
                if 'test_acc' in row and row['test_acc'].strip(): test_accs.append(float(row['test_acc']))
            except ValueError:
                continue
                
    out_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    if train_losses: plt.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=4)
    if test_losses: plt.plot(epochs, test_losses, label='Test Loss', marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True)
    plt.legend()
    loss_path = os.path.join(out_dir, f'{base_name}_loss.png')
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved: {loss_path}")

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    if train_accs: plt.plot(epochs, train_accs, label='Train Accuracy', marker='o', markersize=4)
    if test_accs: plt.plot(epochs, test_accs, label='Test Accuracy', marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.grid(True)
    plt.legend()
    acc_path = os.path.join(out_dir, f'{base_name}_acc.png')
    plt.savefig(acc_path)
    plt.close()
    print(f"Saved: {acc_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'], help='dataset selection')
    parser.add_argument('--csv_path', type=str, default='', help='Direct path to log csv (overrides --dataset)')
    args = parser.parse_args()
    
    csv_path = args.csv_path if args.csv_path else f'./Result/{args.dataset}_log.csv'
    plot_metrics(csv_path)
