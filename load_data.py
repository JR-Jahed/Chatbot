import numpy as np
import json
import pickle

import nltk
from nltk.stem.lancaster import LancasterStemmer

def load_data():
    try:
        with open("data.pkl", "rb") as f:
            words, labels, training_data, output = pickle.load(f)

        return words, labels, training_data, output
    except:

        stemmer = LancasterStemmer()
    
        with open("intents.json") as file:
            data = json.load(file)
    
        words = []
        labels = []
        docs_x = []
        docs_y = []
    
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])
    
                if intent["tag"] not in labels:
                    labels.append(intent["tag"])
    
        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))
    
        labels = sorted(labels)
    
        training_data = []
        output = []
    
        out_empty = [0] * len(labels)
    
        for i, doc in enumerate(docs_x):
            bag = []
            wrds = [stemmer.stem(w) for w in doc]
    
            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)
    
            output_row = out_empty[:]
            output_row[labels.index(docs_y[i])] = 1
    
            training_data.append(bag)
            output.append(output_row)
    
        training_data = np.array(training_data)
        output = np.array(output)
    
        with open("data.pkl", "wb") as f:
            pickle.dump((words, labels, training_data, output), f)

        return words, labels, training_data, output

