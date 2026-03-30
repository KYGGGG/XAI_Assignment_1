import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import csv
from models.dla_simple import SimpleDLA

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

def load_data(dataset_name):
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
    
    return torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

def run_attack_eval(dataset_name, eps_list):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Evaluating Dataset: {dataset_name.upper()}]")
    
    result_dir = 'Result'
    sample_dir = os.path.join(result_dir, 'attack_samples')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
    
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
    
    log_csv = os.path.join(result_dir, f'{dataset_name}_attack_results.csv')
    with open(log_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['eps', 'clean_acc', 'adv_acc', 'targeted_success_rate'])
        
        for eps in eps_list:
            correct = 0
            adv_correct = 0
            targeted_success = 0
            total = 0
            
            samples_saved = False
            
            print(f"Running attack for eps={eps}...")
            
            for i, (inputs, targets) in enumerate(testloader):
                if i >= 4: break 
                
                inputs, targets = inputs.to(device), targets.to(device)
                target_class = 8
                mask = (targets != target_class)
                inputs = inputs[mask]
                targets = targets[mask]
                
                if targets.size(0) == 0:
                    continue
                    
                total += targets.size(0)
                target_labels = torch.full_like(targets, target_class)
                
                with torch.no_grad():
                    outputs = wrapped_model(inputs)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                
                adv_inputs = fgsm_targeted(wrapped_model, inputs, target_labels, eps)
                
                with torch.no_grad():
                    adv_outputs = wrapped_model(adv_inputs)
                    _, adv_predicted = adv_outputs.max(1)
                    adv_correct += adv_predicted.eq(targets).sum().item()
                    targeted_success += adv_predicted.eq(target_labels).sum().item()
                
                if not samples_saved and eps > 0:
                    num_samples = 8
                    comparison = torch.cat([inputs[:num_samples], adv_inputs[:num_samples]], dim=0)
                    save_path = os.path.join(sample_dir, f'{dataset_name}_eps_{eps}.png')
                    save_image(comparison, save_path, nrow=num_samples, normalize=False)
                    print(f"Saved samples to {save_path}")
                    samples_saved = True
            
            clean_acc = 100. * correct / total
            adv_acc = 100. * adv_correct / total
            tsr = 100. * targeted_success / total
            
            print(f"Result for eps {eps}: Clean Acc: {clean_acc:.2f}%, Adv Acc: {adv_acc:.2f}%, TSR: {tsr:.2f}%")
            writer.writerow([eps, clean_acc, adv_acc, tsr])

if __name__ == '__main__':
    eps_to_test = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    datasets = ['mnist', 'cifar10']
    
    for ds in datasets:
        run_attack_eval(ds, eps_to_test)
