# -*- coding: utf-8 -*-
"""scratchpad

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/notebooks/empty.ipynb
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:18:13 2020

@author: avner
"""

import numpy as np

from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot as plt


# generator model
def generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model

# discriminator model
def discriminator(in_shape):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model



# combined generator and discriminator model
def gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	model.add(g_model)
	model.add(d_model)
	
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load  training images
def load_real_images():
	(trainX, _), (_, _) = load_data()
	# add channels dimension
	X = np.expand_dims(trainX, axis=-1)
	# convert from unsigned ints to floats
	X = X.astype('float32')
	# scale to [0,1]
	X = X / 255.0
	return X


# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
    
	# generate 'real' class labels (1)
	y = np.ones((n_samples, 1))
	return X, y

# generate points in latent space for the generator
def generate_latent_points(latent_dim, n_samples):
	x_input = randn(latent_dim * n_samples)
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	x_input = generate_latent_points(latent_dim, n_samples)
	X = g_model.predict(x_input)
	y = np.zeros((n_samples, 1))
	return X, y



# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)

	for i in range(n_epochs):

		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# create training set for the discriminator
			X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = np.ones((n_batch, 1))
			# update the generator 
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))



# load image data
dataset = load_real_images()
# size of the latent space
latent_dim = 100
# create the discriminator
d_model = discriminator(dataset.shape[1:])
# create the generator
g_model = generator(latent_dim)
# create the gan
gan_model = gan(g_model, d_model)

# train model
train(g_model, d_model, gan_model, dataset, latent_dim)



# save weights

filename = 'generator_model.h5'
g_model.save(filename)



# plot 5 examples from generator



# generate images
latent_points = generate_latent_points(100, 25)
# generate images
X = g_model.predict(latent_points)


# plot the result

for i in range(25):
   	# define subplot
   	plt.subplot(5, 5, 1 + i)
  	# turn off axis
   	plt.axis('off')
   	# plot raw pixel data
   	plt.imshow(X[i, :, :, 0], cmap='gray_r')
plt.show()

for i in range(5):
   	# define subplot
   	plt.subplot(5, 5, 1 + i)
  	# turn off axis
   	plt.axis('off')
   	# plot raw pixel data
   	plt.imshow(X[i, :, :, 0], cmap='gray_r')
plt.show()