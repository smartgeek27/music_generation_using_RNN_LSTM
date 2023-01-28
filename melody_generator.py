import json
from preprocess import sequence_length,mapping_path
import numpy as np
import tensorflow.keras as keras
# from itertools import islice
import music21 as m21

class melody_generator:

    def __init__(self, model_path = "model.h5"):

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)


        with open( mapping_path, "r") as fp:
            self._mappings =  json.load(fp)

        self._start_symbols = ["/"] * sequence_length

    def melody_generation(self, seed, num_steps, max_sequence_length, temperature):

        # create seed with start symbols

        # create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self.temperature_sampling(probabilities, temperature)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # update melody
            melody.append(output_symbol)

        return melody


    def temperature_sampling(self, prob, temperature):

        predictions = np.log(prob)/ temperature
        probabilities = np.exp(predictions)/ np.sum(np.exp(predictions))

        choice = range(len(probabilities))
        index = np.random.choice(choice, p =probabilities)
        return index



    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="output/mel1.mid"):

        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)

if __name__ == "__main__":
    mg = melody_generator()
    seed = "55 _ _ _ 60 _ _ _ 60 _ _ _ 64 _ _ _ 67 _ _ _ 72 _ _ _"
    melody = mg.melody_generation(seed, 500,64, 0.7 )
    print(melody)
    mg.save_melody(melody)