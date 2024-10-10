from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

app = Flask(__name__)

class myCorruptionModel(nn.Module):
    def __init__(self):
        super(myCorruptionModel, self).__init__()
        # Convolutional layers to capture spatial features
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # New layer
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Fully connected layers for classification
        self.fc_stack = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Adaptive pooling
            nn.Flatten(),
            nn.Linear(256, 256),  # Update input size accordingly
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout for regularization
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x

model = myCorruptionModel().to("cpu")

model.load_state_dict(torch.load('model_0.pth', map_location=torch.device('cpu')))

model.eval()

def preprocess_image(file):
    # Open the image from the file-like object
    image = Image.open(file).convert('RGB')
    
    # Define the necessary transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match model input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize if needed
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    
    input_tensor = preprocess_image(file)

    
    with torch.no_grad():
        output = model(input_tensor)
    
    
    prediction = output.item() > 0.5
    result = "Corrupted" if prediction else "Not Corrupted"

    return render_template('upload.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
