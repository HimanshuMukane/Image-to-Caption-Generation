import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
# Load the data from the Excel file
data = pd.read_excel("classifier.xlsx")
# Create a text classification model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(data["Item"], data["Category"])
# Take user input and classify it
while True:
    user_input = input("Enter a word: ")
    if user_input.lower() == "exit":
        break
    # Replace underscores with spaces in the user input
    user_input = user_input.replace("_", " ")
    predicted_label = model.predict([user_input])
    print(f"Type: {predicted_label[0]}")