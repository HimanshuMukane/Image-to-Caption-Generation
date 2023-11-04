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
app = Flask(__name__)
@app.route('/')
def index():
    params = {'title': 'Home', 'result': "default"}
    return render_template('index.html', params=params)

@app.route('/image')
def image():
    params = {'title': 'Image Analysis', 'result': "default"}
    return render_template('image.html', params=params)

def predict(image):
    resnet = models.resnet101(pretrained=True)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)

    resnet.eval()

    out = resnet(batch_t)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    top_5_predictions = [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

    return top_5_predictions

def predict_cnn(image):
    resnet = models.resnet50(pretrained=True)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    resnet.eval()
    out = resnet(batch_t)
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    top_5_predictions = [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

    return top_5_predictions


data2 = pd.read_excel("classifier.xlsx")
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(data2["Item"], data2["Category"])
def get_predicted_type(user_input,captionType="Funny"):
    if user_input.lower() == "exit":
        return 
    user_input = user_input.replace("_", " ")
    predicted_label = model.predict([user_input])
    sheetname = predicted_label[0]
    output = generate_caption(sheetname,"Funny")
    return output

@app.route('/analyzeImage', methods=['GET', 'POST'])
def cnn_image_analysis():
    if request.method == 'POST':
        result = "default"
        image = request.files['sample_image']
        result = predict_cnn(image)
        params = {'title': 'Image Analysis with CNN', 'result': result}
    return params
    
@app.route('/getCaption')
def captionGenerator():
    sheetname = request.args.get('result').split(",")[1]
    captionType = request.args.get('captionType')
    print(sheetname,captionType)
    res_data = get_predicted_type(sheetname,captionType).replace("{Replace}",sheetname)
    return res_data

if __name__ == '__main__':
    app.run(debug=True)