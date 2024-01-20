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
        
        self.flatten_size = 32 * 256 * 256  # Für 1024x1024 Bilder
        self.linear_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 4)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.linear_layers(x)
        return x

    def loss(self, y_true, y_pred):
        return nn.CrossEntropyLoss()(y_pred, y_true)


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
        hist['train_acc'].append(np.mean(train_accuracies))  # Hinzufügen der Trainingsgenauigkeit pro Epoche

        mdl.eval()
        val_losses = []
        val_accuracies = []
        with torch.no_grad():
            for X_val, Y_val_true in val_dl:
                Y_val_pred = mdl(X_val)  # Call the model with X_val to get predictions
                val_loss = mdl.loss(Y_val_true, Y_val_pred).item()
                val_losses.append(val_loss)
                val_acc = calculate_accuracy(Y_val_true, Y_val_pred)
                val_accuracies.append(val_acc.item())

        hist['val_loss'].extend(val_losses)
        hist['val_acc'].append(np.mean(val_accuracies))  # Hinzufügen der Validierungs-Genauigkeit pro Epoche

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        print(f'Epoch {epoch+1}/{max_epochs}, Train Loss: {avg_train_loss:.6f}, Train Acc: {hist["train_acc"][-1]:.6f}, Val Loss: {avg_val_loss:.6f}, Val Acc: {hist["val_acc"][-1]:.6f}')

    return hist

def plot_losses(hist, max_epochs):
    train_x = np.arange(len(hist['train_loss']))
    val_x = np.arange(len(hist['val_loss']))

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(train_x, hist['train_loss'], label='Training Loss', color='tab:red')
    ax1.plot(val_x, hist['val_loss'], label='Validation Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.legend()

    # Letzte ermittelte Trainingsgenauigkeit anzeigen
    last_train_acc = hist['train_acc'][-1]
    ax1.text(0.1, 0.1, f'Last Train Acc: {last_train_acc:.4f}', ha='left', va='bottom', transform=ax1.transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7))

    plt.xticks(np.arange(0, len(train_x), len(train_x)//max_epochs))
    plt.show()


# Haupttrainingsteil
if __name__ == "__main__":
    dataset_path = os.path.join(os.getcwd(), 'Desktop/Projekt_ML/try')
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),  # Ändern der Bildgröße auf 256x256
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
    val_dl = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=False)

    mdl = Model()
    max_epochs = 5
    hist = train(train_dl, val_dl, mdl, alpha=0.000001, max_epochs=max_epochs)

    # Erstellen Sie eine Grafik mit den Verlusten für jede Iteration (Batch)
    plot_losses(hist)

    # Speichern des trainierten Modells
    model_path = os.path.join(os.getcwd(), 'Desktop/Projekt_ML/selfmodel.pth')
    # JIT-Kompilierung des Modells vor dem Speichern
    mdl_scripted = torch.jit.script(mdl)
    torch.save(mdl.state_dict(), model_path)
