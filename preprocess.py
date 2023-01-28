
import os
import music21 as m21
import json

import numpy as np
import tensorflow.keras as keras

song_path = "europa/deutschl/erk"
acceptable_durations = [0.25,0.5,0.75,1.0,1.5,2,3 ]
save_path_dir = "dataset"
single_file_path = " combined_file"
num_delimeters = 64
mapping_path = "vocabulary.json"
sequence_length= 64

# load the data

def load_songs(dataset_path):
    # go through all the files
    songs = []
    for path, subdir, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == 'krn':
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)

    return songs

# Replace non acceptable song

def acc_not( song,acceptable_durations ):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

# transposing key and notes into C major and A minor
def transpose(song):

    # get the key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measure_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measure_part0[0][4]
    # print(key.mode)
    # estimate the key using music21
    if not isinstance(key,m21.key.Key):

        # print(type(m21.key.Key))
        key = song.analyze("key")

    # get interval for transposition
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

        # transpose song by calculated interval
    transposed_song = song.transpose(interval)

    return transposed_song

# encoding song
def crypted_song(song, step_size = 0.25):

    encoded_song = []
    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            # print(event)
            symbol = event.pitch.midi
            # print(symbol)
        elif isinstance(event,m21.note.Rest):
            symbol = "r"

         # convert the note and rest in time series
        steps =  int(event.duration.quarterLength/ step_size )
        # print(steps)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # print(encoded_song)
    encoded_song = " ".join(map(str,encoded_song))
    # print(encoded_song)

    return encoded_song

# preprocessing the song and calling individual functions
def preprocess( dataset_path):

    songs = load_songs(dataset_path)
    # print("length of songs", len(songs))


    for i,song in enumerate(songs):
        # filter songs that have non-acceptable durations
        if not acc_not(song,acceptable_durations):
            continue

        # transpose song to Cmajor/Aminor #
        song = transpose(song)

         # encode songs with music time series representation
        encoded_song = crypted_song(song)

        # save songs to text_file
        save_path = os.path.join(save_path_dir,str(i+1))
        with open(save_path,"w") as fp:
            fp.write(encoded_song)


def load( path_to_dataset):
    with open(path_to_dataset,"r") as fp:
        song = fp.read()
    return song

# compile individual song data into one
def single_file_dataset(save_path_dir, combined_datset_path, num_delimeters):
    song_delimeter = " / " * num_delimeters
    songs = ""

    # load encoded data seperated with delimeter
    for path,_,files in os.walk(save_path_dir):
        for file in files:
            file_path = os.path.join(save_path_dir, file)
            song = load(file_path)
            songs = songs + song + "" + song_delimeter

    songs = songs[:-1]

    # save the combined song file in combined data set path

    with open( combined_datset_path, "w") as fp:
        fp.write(songs)

    return songs

# mapping the symbols to intergers
def create_mapping(songs, mapping_path):
    mapping = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))
    # To match the vocabulary
    # print(vocabulary)
    for i,symbol in enumerate(vocabulary):
        mapping[symbol] = i

    # print(mapping)
    # save the final vocabulary to a json file
    with open( mapping_path,"w") as fp:
        json.dump(mapping, fp )

# Converting songs into integers
def song_to_int(songs):

    int_songs = []
    # load the mappings
    with open(mapping_path, "r") as fp:
        mappings = json.load(fp)

    # convert songs to strings
    # print(songs)
    songs = songs.split()
    # print(songs)
    # map songs to strings
    for symbols in songs:
        int_songs.append(mappings[symbols])

    return int_songs

# Arranging song data into time series input for training
def training_sequences(sequence_length):
    # load the songs

    songs = load(single_file_path)
    int_songs = song_to_int(songs)

    # generating training sequences
    input = []
    output = []
    num_sequences = len(int_songs) - sequence_length

    for i in range(num_sequences):
        input.append( int_songs[i:i+ sequence_length])
        output.append(int_songs[i+sequence_length])

    # converting time series to one hot encoding

    vocabulary_size = len(set(int_songs))
    # print(set(int_songs))
    inputs = keras.utils.to_categorical( input,num_classes= vocabulary_size)
    targets = np.array(output)

    return inputs, targets



def main():
    preprocess(song_path)
    songs = single_file_dataset(save_path_dir, single_file_path, num_delimeters)
    create_mapping(songs, mapping_path )
    # print(song_to_int(songs))
    inputs, targets = training_sequences( sequence_length)
    print((inputs.shape))
    print((targets.shape))


if __name__ == "__main__":
    main()

