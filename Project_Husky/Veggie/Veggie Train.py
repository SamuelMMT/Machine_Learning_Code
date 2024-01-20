import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# Modelldefinition
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(32 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 5)  # 5 Klassen
        )

    def forward(self, X0):
        X1 = self.cnn_layers(X0)
        X2 = X1.view(X1.size(0), -1)
        X3 = self.linear_layers(X2)
        return X3

    def loss(self, Y_true, Y_pred):
        return nn.CrossEntropyLoss()(Y_pred, Y_true)

# Trainingsfunktionen
def train_step(X, Y_true, mdl, opt):
    Y_pred = mdl(X)
    L = mdl.loss(Y_true, Y_pred)  
    L.backward() 
    opt.step()
    opt.zero_grad() 
    return L.detach().numpy()

def calculate_accuracy(y_true, y_pred):
    predicted = torch.max(y_pred, 1)[1]
    correct = (predicted == y_true).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

def train(train_dl, val_dl, mdl, alpha, max_epochs):
    opt = torch.optim.Adam(mdl.parameters(), lr=alpha)
    hist = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(max_epochs):
        mdl.train()
        train_losses = []
        train_accuracies = []
        for X, Y_true in train_dl:
            L = train_step(X, Y_true, mdl, opt)
            acc = calculate_accuracy(Y_true, mdl(X))
            train_losses.append(L)
            train_accuracies.append(acc.item())

        hist['train_loss'].extend(train_losses)
        hist['train_acc'].extend(train_accuracies)

        mdl.eval()
        val_losses = []
        val_accuracies = []
        with torch.no_grad():
            for X_val, Y_val_true in val_dl:
                Y_val_pred = mdl(X_val)
                val_loss = mdl.loss(Y_val_true, Y_val_pred).item()
                val_losses.append(val_loss)
                val_acc = calculate_accuracy(Y_val_true, Y_val_pred)
                val_accuracies.append(val_acc.item())
        
        hist['val_loss'].extend(val_losses)
        hist['val_acc'].extend(val_accuracies)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_train_acc = np.mean(train_accuracies)
        avg_val_acc = np.mean(val_accuracies)

        print(f'Epoch {epoch+1}/{max_epochs}, Train Loss: {avg_train_loss:.6f}, Train Acc: {avg_train_acc:.6f}, Val Loss: {avg_val_loss:.6f}, Val Acc: {avg_val_acc:.6f}')

    return hist

def plot_losses_and_accuracy(hist, num_train_batches, num_val_batches):
    train_x = np.arange(len(hist['train_loss'])) / num_train_batches
    val_x = np.arange(len(hist['val_loss'])) * num_val_batches / num_train_batches
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(train_x, hist['train_loss'], label='Training Loss', color='tab:red')
    ax1.plot(val_x, hist['val_loss'], label='Validation Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:green')  
    ax2.plot(train_x, hist['train_acc'], label='Training Accuracy', color='tab:green')
    ax2.plot(val_x, hist['val_acc'], label='Validation Accuracy', color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()  
    plt.legend()
    plt.show()

# Haupttrainingsteil
if __name__ == "__main__":
    dataset_path = r'C:\Users\taube\OneDrive\Desktop\Uni - LaptopAsus\Git_Uni\WS23_24\Projekt_ML_Datensatz'
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    batch_size = 64
    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=True)

    mdl = Model()
    max_epochs = 10
    hist = train(train_dl, val_dl, mdl, alpha=0.00001, max_epochs=max_epochs)

    plot_losses_and_accuracy(hist, len(train_dl), len(val_dl))

    # Speichern des trainierten Modells
    model_path = os.path.join(os.getcwd(), 'Desktop/Projekt_ML/model.pth')
    torch.save(mdl.state_dict(), model_path)

