import torch
import torchvision.transforms as transforms
import cv2
import torch.nn as nn
import numpy as np
from PIL import Image

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

# Laden des Modells
def load_model(model_path):
    mdl = Model()
    mdl.load_state_dict(torch.load(model_path))
    mdl.eval()
    return mdl

# Vorhersage für ein einzelnes Bild
def predict_single_image(image, model, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = transform(image)
    image = image.unsqueeze(0)

    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        predicted_class_probabilities = torch.softmax(outputs, 1)
        max_probability, predicted_class_index = torch.max(predicted_class_probabilities, 1)

        if max_probability > 0.2:
            predicted_class_name = class_names[predicted_class_index]
            return predicted_class_name, max_probability.item()
        else:
            return "None", 0.0

# Klassenliste für Objekte
class_names = ["controller", "taschenrechner", "red", "green"]

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = 'Desktop/Projekt_ML/selfmodel.pth'
    mdl = load_model(model_path)
    mdl.to(device)
    
    #Parallelisierung (Just-In-Time Kompilierung)
    mdl = torch.jit.script(mdl)

    video_path = 'Desktop/Testbilder/apple3.MOV'  # Pfad zum Testvideo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f'Error: Unable to open video file {video_path}')
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_height, frame_width, _ = frame.shape

            # Anpassbare Fenster- und Schrittgrößen
            window_sizes = [256, 512, 1024]
            step_sizes = [128, 256, 512]  # Schrittgrößen kleiner als Fenstergrößen


            # Fenstergrößen in einer Schleife durchlaufen
            max_probability = 0.0
            best_class_name = "None"
            best_x = 0
            best_y = 0
            best_window_size = 0

            for window_size, step_size in zip(window_sizes, step_sizes):
                for y in range(0, frame_height - window_size, step_size):
                    for x in range(0, frame_width - window_size, step_size):
                        window = pil_image.crop((x, y, x + window_size, y + window_size))
                        window = window.resize((1024, 1024))

                        # Vorhersage für jedes Fenster
                        predicted_class_name, current_probability = predict_single_image(window, mdl, device)

                        if current_probability > max_probability:
                            max_probability = current_probability
                            best_class_name = predicted_class_name
                            best_x = x
                            best_y = y
                            best_window_size = window_size

            if best_class_name != "None":
                cv2.rectangle(frame, (best_x, best_y), (best_x + best_window_size, best_y + best_window_size), (0, 255, 0), 5)
                cv2.putText(frame, best_class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

