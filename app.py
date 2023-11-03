# Import necessary libraries and modules
from flask import Flask, render_template, request
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import models, transforms
import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from caption import generate_caption
# Create a Flask application instance
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def index():
    # Initialize parameters for rendering the template
    params = {'title': 'Home', 'result': "default"}
    # Render the 'index.html' template with the specified parameters
    return render_template('index.html', params=params)

# Define a route for the image analysis page
@app.route('/image')
def image():
    # Initialize parameters for rendering the template
    params = {'title': 'Image Analysis', 'result': "default"}
    # Render the 'image.html' template with the specified parameters
    return render_template('image.html', params=params)

# Function to predict image labels using a pre-trained ResNet model
def predict(image):
    # Create a ResNet model
    resnet = models.resnet101(pretrained=True)

    # Define a series of transformations for image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load the input image and apply the transformations
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)

    # Set the model to evaluation mode
    resnet.eval()

    # Perform inference on the preprocessed image
    out = resnet(batch_t)

    # Load the class labels from 'imagenet_classes.txt'
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    # Calculate the top 5 predictions and their probabilities
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    top_5_predictions = [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

    return top_5_predictions

# Define a custom CNN model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Define the layers for the custom CNN
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(128 * 7 * 7, 1000)  # Adjust the output size according to your needs

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Function to predict image labels using a pre-trained CNN model (ResNet-50)
def predict_cnn(image):
    # Use a pre-trained ResNet model
    resnet = models.resnet50(pretrained=True)

    # Define a series of transformations for image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the input image and apply the transformations
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)

    # Set the model to evaluation mode
    resnet.eval()

    # Perform inference on the preprocessed image
    out = resnet(batch_t)

    # Load the class labels from 'imagenet_classes.txt'
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    # Calculate the top 5 predictions and their probabilities
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    top_5_predictions = [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

    return top_5_predictions


# Load the data from the Excel file
data2 = pd.read_excel("classifier.xlsx")
# Create a text classification model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(data2["Item"], data2["Category"])
# Take user input and classify it
def get_predicted_type(user_input,captionType="Funny"):
    # user_input = input("Enter a word: ")
    if user_input.lower() == "exit":
        return 
    # Replace underscores with spaces in the user input
    user_input = user_input.replace("_", " ")
    predicted_label = model.predict([user_input])
    # print(f"Type: {predicted_label[0]}")
    sheetname = predicted_label[0]
    output = generate_caption(sheetname,"Funny")
    return output

# Add a new route for CNN-based image analysis
@app.route('/analyzeImage', methods=['GET', 'POST'])
def cnn_image_analysis():
    if request.method == 'POST':
        # Initialize a default result
        result = "default"
        # Get the uploaded image from the POST request
        image = request.files['sample_image']
        # Call the predict_cnn function to analyze the image
        result = predict_cnn(image)
        # Initialize parameters for rendering the template
        params = {'title': 'Image Analysis with CNN', 'result': result}
        # seedtext = result[0][0].split(",")[1]
        # print(get_predicted_type(seedtext).replace("{Replace}",seedtext))
    return params
    
@app.route('/getCaption')
def captionGenerator():
    sheetname = request.args.get('result').split(",")[1]
    captionType = request.args.get('captionType')
    print(sheetname,captionType)
    res_data = get_predicted_type(sheetname,captionType).replace("{Replace}",sheetname)
    return res_data


# Start the Flask application in debug mode
if __name__ == '__main__':
    app.run(debug=True)