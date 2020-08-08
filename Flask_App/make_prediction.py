import pickle
import pandas as pd
import numpy as np

# read in the model
my_model = pickle.load(open("song_model.pkl","rb"))

# create a function to take in user-entered amounts and apply the model
def top100(song_values, model=my_model):
    # Change duration to msec
    song_values[0] = song_values[0] * 60000

    # Change Key to dummies eliminate key
    key = song_values[11]
    for k in range(1, 12):
        value = 0
        if k == key:
            value = 1
        song_values.append(value)
    song_values.pop(11)

    # Feature engineering append to song_values
    song_values.append(song_values[1] * song_values[2])
    song_values.append(song_values[0] / (song_values[2] + 1))
    song_values.append(song_values[0] * song_values[3])
    song_values.append(song_values[0] / (song_values[1] + 1))
    song_values.append(song_values[8] * song_values[5])
    song_values.append(song_values[6] / (song_values[2] + 1))
    song_values.append(song_values[6] / (song_values[3] + 1))
    song_values.append(song_values[2] / (song_values[3] + 1))

    input_df = pd.DataFrame([song_values])

    # make a prediction
    prediction = my_model.predict(input_df)[0]

    # return a message
    message_array = ["Not a hit",
                     "It's a hit!"]

    return message_array[prediction]
