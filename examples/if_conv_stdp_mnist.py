import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from spikingjelly.activation_based import learning, layer, neuron, functional

def main():
    parser = argparse.ArgumentParser(description='Hybrid STDP/Backprop MNIST Training')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('-data-dir', default='./data', type=str, help='root dir of MNIST dataset')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate for gradient descent')
    parser.add_argument('-lr-stdp', default=1e-2, type=float, help='learning rate for STDP')

    args = parser.parse_args()
    print(args)

    device = torch.device(args.device)
    
    # --- 1. Define the Network ---
    # This is the same network from the docs, but the first Conv2d now takes 1 input channel for grayscale MNIST.
    net = nn.Sequential(
        layer.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Flatten(),
        layer.Linear(16 * 7 * 7, 64, bias=False), # MNIST images are 28x28, so after two 2x2 maxpools, they are 7x7
        neuron.IFNode(),
        layer.Linear(64, 10, bias=False),
        neuron.IFNode(),
    )
    functional.set_step_mode(net, 'm')
    net.to(device)

    # --- 2. Setup Hybrid Learning ---
    # The documentation shows how to separate parameters for STDP and gradient descent [cite: 12, 18]
    instances_stdp = (layer.Conv2d,)
    
    stdp_learners = []
    for i in range(len(net)):
        if isinstance(net[i], instances_stdp):
            stdp_learners.append(
                learning.STDPLearner(step_mode='m', synapse=net[i], sn=net[i+1], tau_pre=2.0, tau_post=2.0,
                                    f_pre=lambda x: torch.clamp(x, -1, 1), f_post=lambda x: torch.clamp(x, -1, 1))
            )

    params_stdp = []
    for m in net.modules():
        if isinstance(m, instances_stdp):
            params_stdp.extend(m.parameters())
            
    params_stdp_set = set(params_stdp)
    params_gradient_descent = [p for p in net.parameters() if p not in params_stdp_set]

    optimizer_gd = Adam(params_gradient_descent, lr=args.lr)
    optimizer_stdp = SGD(params_stdp, lr=args.lr_stdp, momentum=0.) # SGD is used to apply STDP updates [cite: 4]

    # --- 3. Load Data ---
    train_dataset = datasets.MNIST(
        root=args.data_dir, train=True, download=True,
        transform=transforms.ToTensor()
    )
    test_dataset = datasets.MNIST(
        root=args.data_dir, train=False, download=True,
        transform=transforms.ToTensor()
    )

    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False, pin_memory=True)

    # --- 4. Training and Testing Loop ---
    for epoch in range(args.epochs):
        net.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for img, label in tqdm(train_loader, desc=f"Epoch {epoch}"):
            img = img.to(device)
            label = label.to(device)

            # The input needs to be repeated over the time dimension
            x_seq = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)

            # The training step is a precise sequence of operations 
            optimizer_gd.zero_grad()
            optimizer_stdp.zero_grad()
            
            y = net(x_seq).mean(0)
            loss = F.cross_entropy(y, label)
            loss.backward()

            # Zero the gradients for STDP layers to remove backprop-based updates [cite: 13]
            optimizer_stdp.zero_grad()

            for learner in stdp_learners:
                learner.step(on_grad=True)

            optimizer_gd.step()
            optimizer_stdp.step()

            train_loss += loss.item()
            train_correct += (y.argmax(1) == label).float().sum().item()
            train_total += label.numel()

            functional.reset_net(net)
            for learner in stdp_learners:
                learner.reset()

        train_accuracy = 100 * train_correct / train_total
        print(f"Epoch {epoch}: Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # --- Testing Phase ---
        net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(device)
                label = label.to(device)
                x_seq = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)

                y = net(x_seq).mean(0)
                test_correct += (y.argmax(1) == label).float().sum().item()
                test_total += label.numel()

                functional.reset_net(net)
        
        functional.reset_net(net)
        for learner in stdp_learners:
            learner.reset()

        test_accuracy = 100 * test_correct / test_total
        print(f"Epoch {epoch}: Test Accuracy: {test_accuracy:.2f}%\n")


if __name__ == '__main__':
    main()
