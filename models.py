from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Softmax
from keras.models import Sequential

def paper_model(img_width, img_height):
       # create model
       model = Sequential(name="paper_model")
       model.add(Conv2D(32, 5, padding="same", strides=(1, 1), activation="relu", input_shape=(img_width, img_height, 1)))
       model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
       model.add(Conv2D(32, 5, padding="same", strides=(1, 1), activation="relu"))
       model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
       model.add(Flatten())
       model.add(Dense(units=1024, activation="relu"))
       model.add(Dropout(0.5))
       model.add(Dense(units=11, activation="relu"))

       # This is added to make the task a binary classification task
       model.add(Dense(units=1, activation="relu"))

       # Compile model
       model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
       model.summary()
       return model

def our_model(img_width, img_height):
       model = Sequential(name="our_model")
       model.add(Conv2D(64, 7, padding="same", strides=(1, 1), activation="relu", input_shape=(img_width, img_height, 1)))
       model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
       model.add(Conv2D(32, 5, padding="same", strides=(1, 1), activation="relu"))
       model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
       model.add(Conv2D(32, 5, padding="same", strides=(1, 1), activation="relu"))
       model.add(Conv2D(32, 3, padding="same", strides=(1, 1), activation="relu"))
       model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
       model.add(Flatten())
       model.add(Dropout(0.3))
       model.add(Dense(units=64, activation="relu"))
       model.add(Dense(units=32, activation="relu"))
       model.add(Dense(units=1, activation="relu"))

       # Compile model
       model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
       model.summary()
       return model