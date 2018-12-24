import tensorflow as tf

from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, GRU, Dense, TimeDistributed
from keras.models import Model, Sequential

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt

side = 36
vid = np.zeros((1,  1, side, side, 3))
hand_pos = (3, 3)
vid[0, 0, hand_pos[0], hand_pos[1], 1] = 1

checkpoint_path = "checkpoints/cp.ckpt"


vision_module = Sequential()
vision_module.add(Conv2D(64, (3, 3), activation='relu', input_shape=(side, side, 3)))
vision_module.add(Conv2D(64, (3, 3), activation='relu'))
vision_module.add(MaxPooling2D((2, 2)))
vision_module.add(Conv2D(64, (3, 3), activation='relu'))
vision_module.add(Conv2D(64, (3, 3), activation='relu'))
vision_module.add(MaxPooling2D((2, 2)))
vision_module.add(Flatten())

video_input = Input(shape=(1, side, side, 3), batch_shape=(1, 1, side, side, 3))
encoded_frame_sequence = TimeDistributed(vision_module)(video_input)
encoded_video = GRU(128, stateful=True, name='rnn_layer')(encoded_frame_sequence)
pre_output = Dense(128, activation='relu')(encoded_video)
output = Dense(2)(pre_output)

dorsal_stream = Model(inputs=video_input, outputs=output)
dorsal_stream.compile(optimizer='adam', loss='mse')
dorsal_stream.load_weights(checkpoint_path)


target_on = 15 #np.random.randint(16, size=1)
plt.ion()
for i in range(50):
	if i>target_on:
		vid[0, 0, 15, 25, 0] = 1
	predictions = dorsal_stream.predict(vid)
	new_hand_pos = side*np.squeeze(predictions)
	vid[0, 0, hand_pos[0], hand_pos[1], 1] = 0
	hand_pos = np.floor(side*np.squeeze(predictions))
	hand_pos = hand_pos.astype(int)
	vid[0, 0, hand_pos[0], hand_pos[1], 1] = 1
	plt.axis("off")
	plt.imshow(np.squeeze(vid))
	plt.show()
	plt.pause(0.02)
#sio.savemat('predictions128.mat', {'hand_pred':predictions})