import tensorflow as tf

from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, GRU, Dense, TimeDistributed
from keras.models import Model, Sequential

import numpy as np
import scipy.io as sio
import os

mat = sio.loadmat('train750_36x36_len60.mat')
vid, hand = mat['data'], mat['hand']
vid = vid[:, :-1, :, :, :]
hand = hand[:, 1:, :]

checkpoint_path = "checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
	save_weights_only=True, 
	verbose=1,
	period=5)
	
side = 36

vision_module = Sequential()
vision_module.add(Conv2D(64, (3, 3), activation='relu', input_shape=(side, side, 3)))
vision_module.add(Conv2D(64, (3, 3), activation='relu'))
vision_module.add(MaxPooling2D((2, 2)))
vision_module.add(Conv2D(64, (3, 3), activation='relu'))
vision_module.add(Conv2D(64, (3, 3), activation='relu'))
vision_module.add(MaxPooling2D((2, 2)))
vision_module.add(Flatten())

video_input = Input(shape=(59, side, side, 3))
encoded_frame_sequence = TimeDistributed(vision_module)(video_input)
encoded_video = GRU(128, return_sequences=True)(encoded_frame_sequence)
pre_output = TimeDistributed(Dense(128, activation='relu'))(encoded_video)
output = TimeDistributed(Dense(2))(pre_output)

dorsal_stream = Model(inputs=video_input, outputs=output)
#dorsal_stream.summary()
dorsal_stream.compile(optimizer='adam', loss='mse')
dorsal_stream.fit(vid, hand, epochs=200, callbacks = [cp_callback])