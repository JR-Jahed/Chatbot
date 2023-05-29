import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
from keras import models
import random
import json

model_path = "./Saved Model/model.h5"

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

def bag_of_words(input_sentence, words):
    bag = [0] * len(words)

    sentence_words = nltk.word_tokenize(input_sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    for word in sentence_words:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1

    return np.array(bag)


def chat(words, labels):

    model = models.load_model(model_path)

    print("Start talking with the bot (type quit to stop)")

    while True:
        inp = input("You: ")

        if inp.lower() == "quit":
            break

        bow = bag_of_words(inp, words)
        bow = np.expand_dims(bow, axis=0)

        results = model.predict(bow)[0]

        index = np.argmax(results)

        tag = labels[index]

        responses = None

        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]

        print("Bot: ", random.choice(responses))
