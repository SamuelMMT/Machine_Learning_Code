#YOLO, Moving Window Approach


#imagenet
#Bereits vorhandenes Netz vorschalten, abschneiden und eigenes Netz ansetzt (kleiner Datensatz mit dem eignen Netz zu bearbeiten)

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import os

# Pfad zu Ihrem Datensatz
dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "traffic_light_data")

# Vordefinierte Transformationen für die Bilder
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Laden der Trainingsdaten mit den angegebenen Transformationen
train_data = torchvision.datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)

# Laden der Validierungsdaten mit den angegebenen Transformationen
val_data = torchvision.datasets.ImageFolder(root=f"{dataset_path}/val", transform=transform)

# Erstellen von DataLoaders für die Trainings- und Validierungsdaten
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True)

# Definition der Klasse für das CNN-Modell
class TrafficLightClassifier(nn.Module):
    def __init__(self):
        super(TrafficLightClassifier, self).__init__()
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
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),  # 4 Ausgabe-Klassen: rot, gelb, grün
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# Erstellen des Modells und Übermittlung an die GPU (falls verfügbar)
model = TrafficLightClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Definition des Loss-Funktion und Optimierungsverfahren
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trainings-Schleife
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    train_loss = 0.0
    model.train()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_data)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_data)
    val_losses.append(val_loss)
    accuracy = correct / total

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Accuracy: {accuracy}")

# Loss-Kurve anzeigen
plt.figure()
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.plot(range(num_epochs), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Überprüfung und Auswertung eines Beispielbildes
model.eval()


#Hier kann man ein Bild einfügen aus den Validierungsdaten
test_image_path = f"{dataset_path}/val/red/red_71.jpg"
test_image = transform(Image.open(test_image_path)).unsqueeze(0).to(device)

class_names = ['Schwarz', 'Grün', 'Rot']
with torch.no_grad():
    output = model(test_image)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities[0]).item()

print(f"Predicted Class: {class_names[predicted_class]}")
print(f"Class Probabilities: {probabilities[0]}")

