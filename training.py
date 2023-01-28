from matplotlib import pyplot as plt
import tensorflow.keras as keras
from preprocess import training_sequences, sequence_length
# from keras.utils.vis_utils import plot_model

OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS= " SparseCategoricalCrossentropy"
LEARNING_RATE = 0.001
EPOCHS = 35
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"

def build_model(output_units, num_units, loss, learning_rate):

    # create the model architecture

    input = keras.layers.Input(shape =(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units,activation= "softmax")(x)

    model = keras.Model( input, output)

    # compile model

    model.compile(loss='sparse_categorical_crossentropy', optimizer = keras.optimizers.Adam(learning_rate = learning_rate), metrics = ['accuracy'])

    model.summary()

    return model

def train( output_units = OUTPUT_UNITS, num_units = NUM_UNITS, loss = LOSS, learning_rate = LEARNING_RATE):

    # generate the training sequence

    input, targets =  training_sequences(sequence_length)

    # build the network

    model = build_model( output_units, num_units, loss, learning_rate)
    # from keras.utils.vis_utils import plot_model
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # train the model

    history = model.fit(input, targets, epochs= EPOCHS ,validation_split = 0.1, batch_size= BATCH_SIZE, verbose = 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # save the model
    model.save(SAVE_MODEL_PATH )

if __name__ == "__main__":
    train()
