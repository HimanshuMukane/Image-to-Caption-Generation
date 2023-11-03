# Import necessary libraries
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy

# Initialize the spaCy language model
nlp = spacy.load("en_core_web_lg")

# Set hyperparameters for training the model
learning_rate = 0.00001
batch_size = 1224
embedding_size = 50
epochs = 100

# Load caption data from an Excel file (captions.xlsx in this example)
# data = pd.read_excel('captions.xlsx',sheet_name)

# Prompt the user to enter the type of caption they want to generate
# caption_type = input("Enter the type of caption (e.g., Hilarious, Heartwarming, Catchy, etc.): ")
# caption_data = data[caption_type]

# # Tokenize the captions using Keras Tokenizer
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(caption_data)
# total_words = len(tokenizer.word_index) + 1

# # Create input sequences and labels for training
# input_sequences = []
# for line in caption_data:
#     token_list = tokenizer.texts_to_sequences([line])[0]
#     for i in range(1, len(token_list)):
#         n_gram_sequence = token_list[:i+1]
#         input_sequences.append(n_gram_sequence)

# # Find the maximum sequence length and pad sequences
# max_sequence_length = max([len(x) for x in input_sequences])
# input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# # Split the input sequences into X (input) and y (output) data
# X, y = input_sequences[:, :-1], input_sequences[:, -1]
# y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# # Build an RNN model using Keras Sequential API
# model = Sequential()
# model.add(Embedding(total_words, embedding_size, input_length=max_sequence_length-1))
# model.add(LSTM(100))
# model.add(Dense(total_words, activation='softmax'))

# # Compile the model with categorical cross-entropy loss and the Adam optimizer
# model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate))

# # Train the model with the input data
# model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1)

# Function to generate captions based on a seed text
def generate_caption(sheetname,caption_type):
    data = pd.read_excel('captions.xlsx',sheet_name=sheetname)
    caption_data = data[caption_type]
    seed_text = random.choice(caption_data)
    # Tokenize the captions using Keras Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(caption_data)
    total_words = len(tokenizer.word_index) + 1

    # Create input sequences and labels for training
    input_sequences = []
    for line in caption_data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # Find the maximum sequence length and pad sequences
    max_sequence_length = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

    # Split the input sequences into X (input) and y (output) data
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    # Build an RNN model using Keras Sequential API
    model = Sequential()
    model.add(Embedding(total_words, embedding_size, input_length=max_sequence_length-1))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))

    # Compile the model with categorical cross-entropy loss and the Adam optimizer
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate))

    # Train the model with the input data
    model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1)

    generated_caption = seed_text
    for _ in range(max_sequence_length):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)[0]
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
        generated_caption += " " + output_word
        if output_word == 'eos':  # Add this if you have an "eos" token to indicate the end of a sentence.
            break
    words = generated_caption.split()
    cleaned_caption = [words[0]]
    for word in words[1:]:
        if word != cleaned_caption[-1]:
            cleaned_caption.append(word)
    return ' '.join(cleaned_caption)


# Generate captions with a random starting phrase
# random_starting_phrase = random.choice(caption_data)
# cleaned_caption = generate_caption(random_starting_phrase, max_sequence_length, tokenizer)
# print("Cleaned Caption:", cleaned_caption)
