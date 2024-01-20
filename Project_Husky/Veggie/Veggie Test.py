import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn

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
            nn.Linear(32 * 32 * 32, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 5)  # 5 Ausgabeklassen (inklusive "Unknown")
        )

    def forward(self, X0):
        X1 = self.cnn_layers(X0)
        X2 = X1.view(X1.size(0), -1)
        X3 = self.linear_layers(X2)
        return X3

# Funktion zum Laden des Modells
def load_model(model_path):
    mdl = Model()
    mdl.load_state_dict(torch.load(model_path))
    mdl.eval()
    return mdl

# Funktion zur Vorhersage eines spezifischen Bildes
def predict_specific_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Klassenliste (angepasst auf 5 Klassen)
class_names = ["Apple_red", "Apple_green", "Orange", "Banana", "None"]

# Hauptfunktionalität
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'Desktop/Projekt_ML/model.pth'  # Pfad zum gespeicherten Modell

    # Laden des Modells
    mdl = load_model(model_path)
    mdl.to(device)

    # Pfad zum neuen Bild
    new_image_path = 'Desktop/vol.jpg'  # Pfad zu Ihrem Testbild

    # Vorhersage für das neue Bild
    predicted_class_index = predict_specific_image(new_image_path, mdl, device)
    predicted_class_name = class_names[predicted_class_index]
    print(f'Predicted class: {predicted_class_name}')
