from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import ModelCheckpoint

model_path = "./Saved Model/model.h5"

def train(training_data, output, labels):

    print(len(training_data[0]))

    model = Sequential([
        Input(shape=(len(training_data[0]), )),

        Dense(128, activation='relu'),
        Dropout(.5),

        Dense(128, activation='relu'),
        Dropout(.5),

        Dense(len(labels), activation='softmax')
    ])

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor="accuracy",
        save_best_only=True
    )

    model.fit(
        training_data,
        output,
        epochs=100,
        callbacks=[checkpoint]
    )
