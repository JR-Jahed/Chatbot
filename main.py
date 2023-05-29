from load_data import load_data
from training import train
from chat import chat

model_path = "./Saved Model/model.h5"

if __name__ == "__main__":

    words, labels, training_data, output = load_data()

    chatting = True
    # chatting = False

    if chatting:
        chat(words, labels)
    else:

        train(
            training_data=training_data,
            output=output,
            labels=labels,
        )
