The project aims at generating melodies using Neural-Networks discussed in the classroom. For the purposes of the project, we have used Keras/TensorFlow for defining the architecture of our neural networks, Python library Music21 for dealing files containing symbolic musical data, and finally an application called MuseScore for listening and analyzing the generated symbolic musical data. For training the network, ESAC Database’s “deutschl” dataset and RNN-LSTM neural-network architecture were utilized.

Libraries to be imported:
1. tensorflow 2.0>{ to train and import LSTM model from keras }
2. music21 7.0> { preprocessing and assesing midi files }
3. json 
4. numpy { perform array operations }
5. matplotlib { to observe training and testing outputs }


Remember to have all files including  **_combined_file, model.h5, dataset, vocabulary.json_** including all the .py files to get the program working without errors. 
Steps:
1. Preprocess.py : Dataset path europa --> deutschl --> erk, running preprocess will output 3 files combined_file, vocabulary and dataset folder.
2. training.py : _From preprocess import training_sequence_ function. Run the training file to save the model named model.h5 which is the trained model on which music is tested upon. 
3. melody_generator.py : You have to run this file to output the music file which will be saved in the output folder naming **_mel1_erk.mid_**. 
4. melody_comparing.py : this files is to convert the music file in dataset to midi file and will be saved in output folder named **_mel1_orignal.mid_**.

The result from  **_melody_generator.py_** and **_melody_comparing.py** will be stored in output folder for melody comparison and can be heared here:





https://user-images.githubusercontent.com/87424679/215295000-4a7a1473-1be2-4be0-976a-9539ad701644.mp4

