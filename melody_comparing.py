
import os
import music21 as m21
save_path_dir = "dataset"

def load( path_to_dataset):
    with open(path_to_dataset,"r") as fp:
        song = fp.read()
    return song


def save_melody( melody, step_duration=0.25, format="midi", file_name="output/mel1_orignal.mid"):

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

def find_song(save_path_dir):
    for path,_,files in os.walk(save_path_dir):
        file_path = os.path.join(save_path_dir, "4")
        song = load(file_path)
        song = song.split()
        print(song)
        return song

if __name__ == "__main__":

    melody = find_song(save_path_dir)
    save_melody(melody)