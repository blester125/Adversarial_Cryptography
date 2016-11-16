from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import range

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

MESSAGE_LENGTH = 16
KEY_LENGTH = 16
BATCH_SIZE = 124
#BATCH_SIZE = 4096
NUMBER_EPOCHS = 60
# This rate was got from the paper
LEARNING_RATE = 0.0008

def generate_message_and_key(
		n=BATCH_SIZE, 
		message_length=MESSAGE_LENGTH,
		key_length=KEY_LENGTH):
	# Create a n by message length array with values of 1 and -1 to use as 
	# the message inputs. 1 and -1 are used to simplify the loss function math
	message = np.random.randint(0, 2, size=(n, message_length)) * 2 - 1
	# Create the Key in the same way as the message
	key = np.random.randint(0, 2, size=(n, key_length)) * 2 - 1
	return message, key

# A function for doing a 1D convolution
def conv1d(ins, filter_shape, stride, name="conv1d"):
	with tf.variable_scope(name):
		weights = tf.get_variable(
					'weights', 
					shape=filter_shape, 
					initializer=tf.contrib.layers.xavier_initializer())
		conv = tf.nn.conv1d(ins, weights, stride, padding="SAME")
		return conv

# A function that builds the convolutional part that each network shares
# The kernel sizes and strides are from the paper
# These kernels are of the form [window size, input size, output size]
def convolution(ins, name):
	conv1 = conv1d(ins, [4, 1, 2], stride=1, name=name+"_conv1")
	layer1 = tf.nn.sigmoid(conv1)
	conv2 = conv1d(layer1, [2, 2, 4], stride=2, name=name+"_conv2")
	layer2 = tf.nn.sigmoid(conv2)
	conv3 = conv1d(layer2, [1, 4, 4], stride=1, name=name+"_conv3")
	layer3 = tf.nn.sigmoid(conv3)
	conv4 = conv1d(layer3, [1, 4, 1], stride=1, name=name+"_conv4")
	# This final layer uses a tanh activation to bring the values back to 
	# the -1, 1 distribution
	layer4 = tf.nn.tanh(conv4)
	return layer4

class Adverserial_Crypto_Networks():
	def __init__(
			self,
			sess,
			message_length=MESSAGE_LENGTH,
			key_length=KEY_LENGTH,
			batch_size=BATCH_SIZE,
			epochs=NUMBER_EPOCHS,
			learning_rate=LEARNING_RATE):
		self.sess = sess
		self.message_length = message_length
		self.key_length = key_length
		self.N = self.message_length
		self.batch_size = batch_size
		self.epochs = epochs
		self.learning_rate = learning_rate

		self.inference()

	def inference(self):
		# Variables to hold the message and key that are generated for training
		self.message = tf.placeholder("float", [None, self.message_length])
		self.key = tf.placeholder("float", [None, self.key_length])

		# Alice
		# Input is the message (plaintext) and key 
		self.alice_input = tf.concat(concat_dim=1, values=[self.message, self.key])
		# Weights for the Fully connceted later
		self.alice_weights = tf.get_variable(
									"alice_weights", 
									[2 * self.N, 2 * self.N], 
									initializer=tf.contrib.layers.xavier_initializer())
		self.alice_bias = tf.get_variable(
									"alice_bias",
									initializer=tf.constant(1.0, shape=[2 * self.N]))
		self.alice_hidden = tf.nn.sigmoid(
									tf.matmul(self.alice_input, self.alice_weights) 
									+ self.alice_bias)
		# Add a dimension for convolutions
		self.alice_hidden = tf.expand_dims(self.alice_hidden, 2)
		self.alice_conv = convolution(self.alice_hidden, "alice")
		# Remove the added dimension
		self.alice_output = tf.squeeze(self.alice_conv)

		# Bob
		# The input is the ciphertext and the key
		self.bob_input = tf.concat(concat_dim=1, values=[self.alice_output, self.key])
		self.bob_weights = tf.get_variable(
									"bob_weights", 
									[2 * self.N, 2 * self.N], 
									initializer=tf.contrib.layers.xavier_initializer())
		self.bob_bias = tf.get_variable(
									"bob_bias",
									initializer=tf.constant(1.0, shape=[2 * self.N]))
		self.bob_hidden = tf.nn.sigmoid(
									tf.matmul(self.bob_input, self.bob_weights) 
									+ self.bob_bias)
		self.bob_hidden = tf.expand_dims(self.bob_hidden, 2)
		self.bob_conv = convolution(self.bob_hidden, "bob")
		self.bob_output = tf.squeeze(self.bob_conv)

		# Eve
		# The input is only the ciphertext
		self.eve_input = self.alice_output
		self.eve_weights = tf.get_variable(
									"eve_weights", 
									[self.N, 2 * self.N], 
									initializer=tf.contrib.layers.xavier_initializer())
		self.eve_bias = tf.get_variable(
									"eve_bias",
									initializer=tf.constant(1.0, shape=[2 * self.N]))
		self.eve_hidden = tf.nn.sigmoid(
									tf.matmul(self.eve_input, self.eve_weights) 
									+ self.eve_bias)
		self.eve_hidden = tf.expand_dims(self.eve_hidden, 2)
		self.eve_conv = convolution(self.eve_hidden, "eve")
		self.eve_output = tf.squeeze(self.eve_conv)

		self.saver = tf.train.Saver()

	def do_training(self):
		# The decrypt errors are the differences between the plaintext and ciphertext
		self.decrypt_err_eve = tf.reduce_mean(tf.abs(self.message - self.eve_output))
		self.decrypt_err_bob = tf.reduce_mean(tf.abs(self.message - self.bob_output))

		# The loss for Bob (and Alice) includes how well Eve did. The function 
		# ((N/2 - Eve Error) ^2) / (N/2) however because the bits are encoded 
		# as -1 and 1 rather than 0 and 1 you can cancel the (N/2) parts 
		self.bob_loss = self.decrypt_err_bob + (1 - self.decrypt_err_eve) ** 2

		# Split the variables into two list so that one can be optimized by the 
		# eve optimizer rather than bob optimizer
		self.t_vars = tf.trainable_variables()
		self.alice_or_bob_vars = (
				[var for var in self.t_vars 
					if "alice_" in var.name 
					or "bob_" in var.name])
		self.eve_vars = [var for var in self.t_vars if "eve_" in var.name]

		# Two optimizers, using AdamOptimizer like in the paper
		self.bob_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
											self.bob_loss, 
											var_list=self.alice_or_bob_vars)
		self.eve_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
											self.decrypt_err_eve, 
											var_list=self.eve_vars)

		# Keeping errors for ploting errors
		self.bob_errors = []
		self.eve_errors = []


		tf.initialize_all_variables().run()
		for i in range(self.epochs):
			iterations = 25000
			
			# Keeping track of errors for ploting
			self.bob_mb_losses = []
			self.eve_mb_losses = []
			
			# Training Alice and bob
			print("Training Alice and Bob, Epoch:",i+1)
			bob_loss, _ = self.train('bob',iterations)
			self.bob_errors.append(bob_loss)

			# Training Eve
			print("Training Eve, Epoch:",i+1)
			_, eve_loss = self.train('eve', iterations)
			self.eve_errors.append(eve_loss)
			
			# Vizualize losses within a minibatch
			if i == 59:
				self.plot_losses()

		# Visualize losses across all epochs
		self.plot_errors()
		save_path = self.saver.save(self.sess, "model.pkl")

	# A single training epoch
	def train(self, network, iterations):
		bob_error = 1.
		eve_error = 1.
		
		batch_size = self.batch_size

		# If this is training eve we use two mini batches as per the paper to 
		# give the network a computational edge.
		if network == "eve":
			batch_size = batch_size * 2

		for i in range(iterations):
			message_val, key_val = generate_message_and_key(n=batch_size)
			if network == "bob":
				_, decrypt_err = self.sess.run(
										[self.bob_optimizer, self.decrypt_err_bob], 
										feed_dict={
											self.message:message_val, 
											self.key:key_val
										}
									)
				bob_error = min(bob_error, decrypt_err)
				
				self.bob_mb_losses.append(decrypt_err)
			
			if network == "eve":
				_, decrypt_err = self.sess.run(
										[self.eve_optimizer, self.decrypt_err_eve], 
										feed_dict={
											self.message:message_val, 
											self.key:key_val
										}
									)
				eve_error = min(eve_error, decrypt_err)
				
				self.eve_mb_losses.append(decrypt_err)
		
		return bob_error, eve_error

	# Plot the minimum errors across all the epochs
	def plot_errors(self):
		plt.plot(self.bob_errors)
		plt.plot(self.eve_errors)
		plt.legend(['Bob', 'Eve'])
		plt.xlabel("Epoch")
		plt.ylabel("Lowest Error")
		plt.show()

	# Plot the errors within a minibatch
	def plot_losses(self):
		plt.plot(self.bob_mb_losses)
		plt.plot(self.eve_mb_losses)
		plt.legend(['Bob', 'Eve'])
		plt.xlabel("Iterations")
		plt.ylabel("Error")
		plt.show()

	def do_example(self):
		message_val, key_val = generate_message_and_key(n=2)
		ciphertext, decoded_text, guess = sess.run([self.alice_output,
													self.bob_output,
													self.eve_output],
													feed_dict={self.message: message_val,
															   self.key:key_val})
		print("The message was:", message_val[0])
		print("The key was:", key_val[0])
		#print("The ciphertext was:", ciphertext[0])
		print("The decoded text was:", decoded_text[0])
		print("The Guess was:", guess[0])
		#print(np.all(message_val == decoded_text))

	def load(self, filename):
		self.saver.restore(sess, filename)

if __name__ == "__main__":
	with tf.Session() as sess:
		nets = Adverserial_Crypto_Networks(sess)
		nets.do_training()
		#nets.load("model.pkl")
		nets.do_example()