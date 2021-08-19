import sys
import os
import numpy as np
import pandas as pd
import random

np.random.seed(42)

NUM_FEATS = 90

class Net(object):
	'''
	'''

	def __init__(self, num_layers, num_units):
		'''
		Initialize the neural network.
		Create weights and biases.

		Here, we have provided an example structure for the weights and biases.
		It is a list of weight and bias matrices, in which, the
		dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
		weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
		biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]

		Please note that this is just an example.
		You are free to modify or entirely ignore this initialization as per your need.
		Also you can add more state-tracking variables that might be useful to compute
		the gradients efficiently.


		Parameters
		----------
			num_layers : Number of HIDDEN layers.
			num_units : Number of units in each Hidden layer.
		'''
		np.random.seed(42)
		self.num_layers = num_layers
		self.num_units = num_units


		self.biases = []
		self.weights = []

		for i in range(num_layers):

			if i==0:
				# Input layer
				self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))

				
			else:
				# Hidden layer
				self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))


			self.biases.append(np.random.uniform(-1, 1, size=(1, self.num_units)))


		# Output layer
		self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))

		self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))


	def __call__(self, X):
		'''
		Forward propagate the input X through the network,
		and return the output.
		
		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
		Returns
		----------
			y : Output of the network, numpy array of shape m x 1
		'''
		weights = self.weights
		biases = self.biases
		self.activations = []
		self.dotz = []
		self.activations.append(X)
		for layer_num in range(self.num_layers):

			dotz = np.dot(X, weights[layer_num]) + biases[layer_num]

			self.dotz.append(dotz)
			activation = np.maximum(dotz, 0)

			self.activations.append(activation)
			X = activation
		
		dotz_out = np.dot(X, weights[-1]) + biases[-1]
		self.dotz.append(dotz_out)
		
		y_hat = np.maximum(dotz_out, 0)
		self.activations.append(y_hat)

		return y_hat
		raise NotImplementedError

	def backward(self, X, y, lamda):
		'''
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
			y : Output of the network, numpy array of shape m x 1
			lamda : Regularization parameter.

		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing bacward pass.
		'''


		del_W = []
		del_b = []

		m = y.shape[0]
		L = self.num_layers

		a_list = self.activations
		z_list = self.dotz
		weights = self.weights
		biases = self.biases

		y = np.reshape(y, (y.shape[0], 1))
		del_aL = (2/m) * (a_list[-1] - y)

		del_WL = np.dot(a_list[-2].T, (del_aL*(z_list[-1] > 0))) + lamda * (weights[-1])

		del_bL = np.sum(del_aL, axis=0) + lamda * (biases[-1])

		del_W.append(del_WL)
		del_b.append(del_bL)
		
		del_al = del_aL
		for l in reversed(range(1, L+1)):
			del_al = np.dot((del_al), weights[l].T)

			del_Wl = np.dot(a_list[l-1].T, (del_al*(z_list[l-1] > 0))) + lamda * (weights[l-1])
			del_bl = np.sum(del_al*(z_list[l-1] > 0), axis=0) + lamda * (biases[l-1])

			del_W.append(del_Wl)
			del_b.append(del_bl)	
			
		del_W = list(reversed(del_W))
		del_b = list(reversed(del_b))

		
		return del_W, del_b
		raise NotImplementedError


class Optimizer(object):
	'''
	'''

	def __init__(self, learning_rate):
		'''
		Create a Gradient Descent based optimizer with given
		learning rate.

		Other parameters can also be passed to create different types of
		optimizers.

		Hint: You can use the class members to track various states of the
		optimizer.
		'''

		self.learning_rate = learning_rate

		self.delta_weights = []
		self.delta_biases = []
		#raise NotImplementedError

	def step(self, weights, biases, delta_weights, delta_biases):
		'''
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.
		'''
		lr = self.learning_rate

		for layer_num in range(len(weights)):			

			weights[layer_num] -= lr * delta_weights[layer_num]
			biases[layer_num] -= lr * delta_biases[layer_num]

		return weights, biases
		raise NotImplementedError


def loss_mse(y, y_hat):
	'''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		MSE loss between y and y_hat.
	'''
	m = y.shape[0]
	y = np.reshape(y, (y.shape[0], 1))
	mse = (1/(m))*np.sum((y - y_hat)**2)
	return mse
	raise NotImplementedError

def loss_regularization(weights, biases):
	'''
	Compute l2 regularization loss.

	Parameters
	----------
		weights and biases of the network.

	Returns
	----------
		l2 regularization loss 
	'''
	running_sum = 0

	for layer_num in range(len(weights)):
		running_sum += np.sum((weights[layer_num])**2) + np.sum((biases[layer_num])**2)

	return running_sum
	raise NotImplementedError

def loss_fn(y, y_hat, weights, biases, lamda):
	'''
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights and biases of the network
		lamda: Regularization parameter

	Returns
	----------
		l2 regularization loss 
	'''
	l2_loss = loss_mse(y, y_hat) + lamda * loss_regularization(weights, biases)
	return l2_loss
	raise NotImplementedError

def rmse(y, y_hat):
	'''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		RMSE between y and y_hat.
	'''
	rsme = (loss_mse(y, y_hat))**0.5
	return rsme
	raise NotImplementedError


def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target
):
	'''
	In this function, you will perform following steps:
		1. Run gradient descent algorithm for `max_epochs` epochs.
		2. For each batch of the training data
			1.1 Compute gradients
			1.2 Update weights and biases using step() of optimizer.
		3. Compute RMSE on dev data after running `max_epochs` epochs.

	Here we have added the code to loop over batches and perform backward pass
	for each batch in the loop.
	For this code also, you are free to heavily modify it.
	'''

	m = train_input.shape[0]

	for e in range(max_epochs):
		epoch_loss = 0.
		epoch_loss_rmse = 0.
		iter_count = 0
		for i in range(0, m, batch_size):
			iter_count += 1

			batch_input = train_input[i:i+batch_size]

			batch_target = train_target[i:i+batch_size]

			pred = net(batch_input)

			# Compute gradients of loss w.r.t. weights and biases
			dW, db = net.backward(batch_input, batch_target, lamda)

			#norm_dW = [np.sum(grad_mat**2) for grad_mat in dW]
			#norm_db = [np.sum(grad_mat**2) for grad_mat in db]

			#if iter_count%200 == 0:
			#	print('norm of gradients')
			#	print(norm_dW, norm_db)
			# Get updated weights based on current weights and gradients
			weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

			#norm_W = [np.sum(weight_mat**2) for weight_mat in weights_updated]
			#norm_b = [np.sum(bias_mat**2) for bias_mat in biases_updated]

			#if iter_count%200 == 0:
			#	print('norm of weights and biases')
			#	print(norm_W, norm_b)
			# Update model's weights and biases
			net.weights = weights_updated
			net.biases = biases_updated


			# Compute loss for the batch
			batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
			epoch_loss += batch_loss
			batch_loss_rmse = rmse(batch_target, pred)
			epoch_loss_rmse += (batch_loss_rmse)**2

		print(f'\n\nEpoch Loss for epoch {e}: {(epoch_loss_rmse*(batch_size/m))**0.5}')
		dev_pred = net(dev_input)
		dev_rmse = rmse(dev_target, dev_pred)
		print(f'RMSE dev for epoch   {e}: {dev_rmse}\n\n')

	dev_pred = net(dev_input)
	dev_rmse = rmse(dev_target, dev_pred)

	print('RMSE on dev data: {:.5f}'.format(dev_rmse))


def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d

	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	'''
	test_preds = np.float32(np.round(net(inputs)))

	Ids = np.arange(1, test_preds.shape[0] + 1, 1, dtype='f')
	Ids = np.reshape(Ids, (Ids.shape[0], 1))	

	
	predictions = np.concatenate((Ids, test_preds), axis=1)

	test_data_predictions = pd.DataFrame(predictions, columns=['Id', 'Predicted'])

	test_data_predictions.to_csv('193109010.csv', index=False)
	return test_preds
	raise NotImplementedError

def read_data():
	'''
	Read the train, dev, and test datasets
	'''
	df1 = pd.read_csv('train.csv')
	train_input = df1.iloc[:, 1:].to_numpy()
	indices = np.arange(0, train_input.shape[0])
	np.random.shuffle(indices)
	train_input = train_input[indices]
	train_target = df1.iloc[:, 0].to_numpy()
	train_target = train_target[indices]


	df2 = pd.read_csv('dev.csv')
	dev_input = df2.iloc[:, 1:].to_numpy()
	dev_target = df2.iloc[:, 0].to_numpy()

	df3 = pd.read_csv('test.csv')
	test_input = df3.iloc[:, 0:].to_numpy()

	return train_input, train_target, dev_input, dev_target, test_input


def main():

	# These parameters should be fixed for Part 1
	max_epochs = 30
	batch_size = 32


	learning_rate = 0.01
	num_layers = 1
	num_units = 32
	lamda = 0.0 # Regularization Parameter

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	get_test_data_predictions(net, test_input)
	

if __name__ == '__main__':
	main()
