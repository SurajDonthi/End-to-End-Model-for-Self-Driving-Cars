import os
import pandas as pd
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, Dropout, Flatten, Dense
from datasets import batch_generator, INPUT_SHAPE
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

"""
This file contains the model definition, training and compilation.
"""

# <editor-fold desc="Set Hyperparameters">
dataset_dir = "Dataset/"
data = ''
batch_size = 100
learning_rate = 0.01
samples_per_epoch = 100             # Assuming 10,000 images
nb_epoch = 25
test_size = 0.3
keep_prob = 0.5
# </editor-fold>

# <editor-fold desc="Load Data">
data_df = pd.read_csv(os.path.join(dataset_dir + data,
                                   'driving_log.csv'))

X = data_df[['center', 'left', 'right']].values
y = data_df['steering'].values


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=0)

# </editor-fold>


class Model():

    def __init__(self, INPUT_SHAPE, keep_prob):
        self.model = self.load(INPUT_SHAPE, keep_prob)

    def load(self, INPUT_SHAPE, keep_prob):

        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
        model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(64, 3, 3, activation='elu'))
        model.add(Conv2D(64, 3, 3, activation='elu'))
        model.add(Dropout(keep_prob))
        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1))
        model.summary()

        return model

    def loss_func(self, learning_rate):
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

        return model

    def train(self, checkpoint):
        self.model.fit_generator(batch_generator(dataset_dir + data,
                                            X_train,
                                            y_train,
                                            batch_size,
                                            True),
                            samples_per_epoch,
                            nb_epoch,
                            max_queue_size=1,
                            validation_data=batch_generator(dataset_dir + data,
                                                            X_valid,
                                                            y_valid,
                                                            batch_size,
                                                            False),
                            validation_steps=len(X_valid),
                            callbacks=[checkpoint],
                            verbose=1)

        return  model

    def save(self):
        checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
        return checkpoint


if __name__ == '__main__':

    model = Model(INPUT_SHAPE, keep_prob)
    model.loss_func(learning_rate)
    checkpoint = model.save()
    model.train(checkpoint)

