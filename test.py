import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import argparse
from models.dla_simple import SimpleDLA

# Class names for labeling
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
MNIST_CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# 1. Targeted FGSM Attack Function
def fgsm_targeted(model, x, target, eps):
    """
    model : the neural network
    x : input image tensor (requires_grad should be set)
    target : the desired (wrong) class label
    eps : perturbation magnitude (e.g., 0.1, 0.3)
    return : adversarial image x_adv
    """
    model.eval()

    x_adv = x.clone().detach().requires_grad_(True)
    target = target.to(x_adv.device)

    logits = model(x_adv)
    loss = F.cross_entropy(logits, target)
    model.zero_grad()
    loss.backward()

    grad = x_adv.grad.data
    x_adv = x_adv - eps * grad.sign()  
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()

# 2. Untargeted FGSM Attack Function
def fgsm_untargeted(model, x, label, eps):
    """
    model : the neural network
    x : input image tensor
    label : the correct class label
    eps : perturbation magnitude
    return : adversarial image x_adv
    """
    model.eval()

    x_adv = x.clone().detach().requires_grad_(True)
    label = label.to(x_adv.device)

    logits = model(x_adv)
    loss = F.cross_entropy(logits, label)
    model.zero_grad()
    loss.backward()

    grad = x_adv.grad.data
    x_adv = x_adv + eps * grad.sign()  # Increase loss for correct label
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()

# 3. Targeted PGD (Projected Gradient Descent) Attack Function
def pgd_targeted(model, x, target, k, eps, eps_step):
    """
    model : the neural network
    x : input image tensor
    target : desired (wrong) class label
    k : number of iterations (e.g., 10, 40)
    eps : total perturbation budget
    eps_step : step size per iteration
    return : adversarial image x_adv
    """
    model.eval()
    
    # Initialize adversarial image (Same as original x for vanilla PGD)
    x_adv = x.clone().detach()
    target = target.to(x.device)
    
    for _ in range(k):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss = F.cross_entropy(logits, target)
        
        model.zero_grad()
        loss.backward()
        
        # Step: Move TOWARDS the target class (targeted)
        grad = x_adv.grad.data
        x_adv = x_adv.detach() - eps_step * grad.sign()
        
        # Projection: L_inf ball constraint
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, min=0.0, max=1.0)
        
    return x_adv.detach()

# 4. Model Wrapping with Normalization
class WrappedModel(nn.Module):
    def __init__(self, model, dataset_name):
        super(WrappedModel, self).__init__()
        self.model = model
        if dataset_name == 'cifar10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        else: # mnist
            mean = torch.tensor([0.1307, 0.1307, 0.1307]).view(1, 3, 1, 1)
            std = torch.tensor([0.3081, 0.3081, 0.3081]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)

def save_labeled_samples(inputs, adv_inputs, targets, adv_predicted, dataset_name, eps, sample_dir):
    classes = CIFAR10_CLASSES if dataset_name == 'cifar10' else MNIST_CLASSES
    num_samples = len(inputs)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.5, 6))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
        
    for i in range(num_samples):
        # Original
        img = inputs[i].cpu().permute(1, 2, 0).numpy()
        if dataset_name == 'mnist':
            axes[0, i].imshow(img, cmap='gray')
        else:
            axes[0, i].imshow(img)
        axes[0, i].set_title(f"Org: {classes[targets[i]]}")
        axes[0, i].axis('off')
        
        # Adversarial
        adv_img = adv_inputs[i].cpu().permute(1, 2, 0).numpy()
        if dataset_name == 'mnist':
            axes[1, i].imshow(adv_img, cmap='gray')
        else:
            axes[1, i].imshow(adv_img)
        axes[1, i].set_title(f"Adv: {classes[adv_predicted[i]]}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(sample_dir, f'{dataset_name}_eps_{eps}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved labeled samples to {save_path}")

def load_data(dataset_name, batch_size=32):
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

def run_attack_eval(dataset_name, eps_list, attack_mode='untargeted', target_class=8, k=10, alpha=0.01):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Evaluating Dataset: {dataset_name.upper()} | Mode: {attack_mode.upper()}]")
    
    result_dir = 'Result'
    sample_dir = os.path.join(result_dir, f'samples_{attack_mode}')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Load Model
    net = SimpleDLA()
    ckpt_path = f'./checkpoint/SimpleDLA_{dataset_name}.pth'
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['net']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)
    
    wrapped_model = WrappedModel(net, dataset_name).to(device)
    wrapped_model.eval()
    
    testloader = load_data(dataset_name)
    
    log_csv = os.path.join(result_dir, f'{dataset_name}_{attack_mode}_results.csv')
    with open(log_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        if 'untargeted' not in attack_mode:
            writer.writerow(['eps', 'clean_acc', 'adv_acc', 'target_success_rate'])
        else:
            writer.writerow(['eps', 'clean_acc', 'adv_acc'])
        
        for eps in eps_list:
            correct = 0
            adv_correct = 0
            targeted_success = 0
            total = 0
            samples_saved = False
            
            print(f"Running {attack_mode} attack for eps={eps}...")
            
            for i, (inputs, targets) in enumerate(testloader):
                if i >= 4: break # Limit to ~128 images for speed
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                if 'untargeted' not in attack_mode:
                    mask = (targets != target_class)
                    inputs, targets = inputs[mask], targets[mask]
                    if targets.size(0) == 0: continue
                    target_labels = torch.full_like(targets, target_class)
                
                total += targets.size(0)
                
                # Original predictions
                with torch.no_grad():
                    outputs = wrapped_model(inputs)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                
                # Generate adversarial images
                if attack_mode == 'targeted':
                    adv_inputs = fgsm_targeted(wrapped_model, inputs, target_labels, eps)
                elif attack_mode == 'untargeted':
                    adv_inputs = fgsm_untargeted(wrapped_model, inputs, targets, eps)
                elif attack_mode == 'pgd_targeted':
                    adv_inputs = pgd_targeted(wrapped_model, inputs, target_labels, k=k, eps=eps, eps_step=alpha)
                
                # Adversarial predictions
                with torch.no_grad():
                    adv_outputs = wrapped_model(adv_inputs)
                    _, adv_predicted = adv_outputs.max(1)
                    adv_correct += adv_predicted.eq(targets).sum().item()
                    if 'untargeted' not in attack_mode:
                        targeted_success += adv_predicted.eq(target_labels).sum().item()
                
                if not samples_saved and eps > 0:
                    num_samples = min(8, inputs.size(0))
                    save_labeled_samples(inputs[:num_samples], adv_inputs[:num_samples], 
                                         targets[:num_samples], adv_predicted[:num_samples], 
                                         dataset_name, eps, sample_dir)
                    samples_saved = True
            
            clean_acc = 100. * correct / total
            adv_acc = 100. * adv_correct / total
            
            if 'untargeted' not in attack_mode:
                tsr = 100. * targeted_success / total
                print(f"Eps {eps}: Clean={clean_acc:.2f}%, Adv={adv_acc:.2f}%, TSR={tsr:.2f}%")
                writer.writerow([eps, clean_acc, adv_acc, tsr])
            else:
                print(f"Eps {eps}: Clean={clean_acc:.2f}%, Adv={adv_acc:.2f}%")
                writer.writerow([eps, clean_acc, adv_acc])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Attack Evaluation')
    parser.add_argument('--mode', type=str, default='untargeted', choices=['targeted', 'untargeted', 'pgd_targeted'], help='Attack mode')
    parser.add_argument('--target', type=int, default=4, help='Target class index')
    parser.add_argument('--dataset', type=str, default='all', choices=['mnist', 'cifar10', 'all'], help='Dataset to evaluate')
    parser.add_argument('--iters', type=int, default=10, help='Number of iterations for PGD')
    parser.add_argument('--alpha', type=float, default=0.01, help='Step size for PGD')
    args = parser.parse_args()
    
    eps_to_test = [1e-4, 0.05, 0.1, 0.15, 0.2, 0.3] 
    
    datasets = ['mnist', 'cifar10'] if args.dataset == 'all' else [args.dataset]
    
    for ds in datasets:
        run_attack_eval(ds, eps_to_test, attack_mode=args.mode, target_class=args.target, k=args.iters, alpha=args.alpha)
