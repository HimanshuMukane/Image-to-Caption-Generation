# Import necessary libraries
import numpy as np                  # Import NumPy for numerical operations
import pandas as pd                 # Import Pandas for data manipulation
import random                       # Import the random module for generating random values
import tensorflow as tf              # Import TensorFlow for deep learning
from tensorflow.keras.models import Sequential   # Import the Sequential model from Keras
from tensorflow.keras.layers import Embedding, LSTM, Dense  # Import layers for the model
from tensorflow.keras.preprocessing.text import Tokenizer  # Import Tokenizer for text preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Import sequence padding
import spacy                         # Import spaCy for natural language processing

# Initialize the spaCy language model
nlp = spacy.load("en_core_web_lg")   # Load the "en_core_web_lg" spaCy language model for English

# Set hyperparameters for training the model
learning_rate = 0.00001              # Learning rate for model training
batch_size = 1224                    # Batch size for training
embedding_size = 50                 # Size of word embeddings
epochs = 100                        # Number of training epochs

# Function to generate captions based on a seed text
def generate_caption(sheetname, caption_type):
    data = pd.read_excel('captions.xlsx', sheet_name=sheetname)  # Read data from an Excel file
    caption_data = data[caption_type]    # Select a specific column from the Excel sheet
    seed_text = random.choice(caption_data)  # Choose a random seed text from the selected column
    
    # Tokenize the captions using Keras Tokenizer
    tokenizer = Tokenizer()           # Initialize a Tokenizer
    tokenizer.fit_on_texts(caption_data)  # Fit the tokenizer on the caption data
    total_words = len(tokenizer.word_index) + 1  # Get the total number of unique words in the captions

    # Create input sequences and labels for training
    input_sequences = []              # Initialize a list to store input sequences
    for line in caption_data:
        token_list = tokenizer.texts_to_sequences([line])[0]  # Tokenize the current caption
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]   # Create n-grams from the tokenized caption
            input_sequences.append(n_gram_sequence)  # Add the n-gram sequence to the input list

    # Find the maximum sequence length and pad sequences
    max_sequence_length = max([len(x) for x in input_sequences])  # Determine the maximum sequence length
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')  # Pad sequences to the maximum length

    # Split the input sequences into X (input) and y (output) data
    X, y = input_sequences[:, :-1], input_sequences[:, -1]  # Split input and output sequences
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)  # Convert output to one-hot encoding

    # Build an RNN model using Keras Sequential API
    model = Sequential()  # Initialize a sequential model
    model.add(Embedding(total_words, embedding_size, input_length=max_sequence_length-1))  # Add an embedding layer
    model.add(LSTM(100))    # Add an LSTM layer with 100 units
    model.add(Dense(total_words, activation='softmax'))  # Add a dense layer with softmax activation

    # Compile the model with categorical cross-entropy loss and the Adam optimizer
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate))

    # Train the model with the input data
    model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1)  # Train the model with the specified parameters

    generated_caption = seed_text  # Initialize the generated caption with the seed text
    for _ in range(max_sequence_length):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]  # Tokenize the current seed text
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')  # Pad the token list
        predicted = np.argmax(model.predict(token_list), axis=-1)[0]  # Predict the next word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word  # Find the word corresponding to the predicted index
                break
        seed_text += " " + output_word  # Add the predicted word to the seed text
        generated_caption += " " + output_word  # Add the predicted word to the generated caption
        if output_word == 'eos':  # Check if the predicted word is an "eos" token (end of sentence)
            break
    words = generated_caption.split()  # Split the generated caption into words
    cleaned_caption = [words[0]]  # Initialize a cleaned caption with the first word
    for word in words[1:]:
        if word != cleaned_caption[-1]:
            cleaned_caption.append(word)  # Remove duplicate consecutive words from the generated caption
    return ' '.join(cleaned_caption)  # Return the cleaned generated caption as a string
