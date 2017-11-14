import gym
import numpy as np
# from random import random
from math import exp
from datetime import datetime
from random import seed, shuffle,\
					uniform, randint,\
					randrange, random
from collections import deque
import math
import copy
import sys
import time
from itertools import izip
import csv
import os

EPISODES = 5000
DEQUE = 2000
EPSILON = 0.85
LAMBDA = 0.99
EPSILON_BASE = 0.01
HIDDEN_LAYERS = 3
NEURAL_DENSITY = 5
GAMMA = 0.95
LEARNING_RATE = 0.04
GRADE = 1.0
SCORE_CEILING = 1

# class Neuron:
# 	def __init__(self):
# 		self.output = 0.0
# 		self.delta = 0
# 	def delta(self):
# 		return self._delta
# 	def output(self):
# 		return self._output
#
# class Ada:
# 	def __init__(self, states, actions, layers, density):
# 		self.actions = actions
# 		self.states = states
# 		self.layers = layers
# 		self.network = self._construct_network(states, actions, layers, density)
# 		self.memory = deque(maxlen=20000)
# 		self.epsilon = EPSILON
# 		self.e_decay = LAMBDA
# 		self.e_base = EPSILON_BASE
# 		self.gamma = GAMMA
# 	def layers(self):
# 		return self._layers
# 	def epsilon(self):
# 		return self._epsilon
# 	def e_decay(self):
# 		return self._e_decay
# 	def e_base(self):
# 		return self._e_base
#
# 	def _construct_network(self, states, actions, layers, density):
# 		network = []
# 		network.append({'neurons': [], 'weights': []})
# 		for s in range(states):
# 			network[0]['neurons'].append(Neuron())
# 			network[0]['weights'].append([])
# 			for d in range(density):
# 				network[0]['weights'][s].append(random.uniform(-1, 1))
# 		l = 1
# 		while l <= layers:
# 			network.append({'neurons': [], 'weights': []})
# 			for d in range(density):
# 				network[l]['neurons'].append(Neuron())
# 				network[l]['weights'].append([])
# 				for n in range(density):
# 					network[l]['weights'][d].append(random.uniform(-1, 1))
# 			l += 1
# 		network.append({'neurons': [], 'weights': []})
# 		for a in range(actions):
# 			network[l]['neurons'].append(Neuron())
# 			network[l]['weights'].append([])
# 			for d in range(density):
# 				network[l]['weights'][a].append(random.uniform(-1, 1))
# 		return network
#
# 	def normalise(self, state):
# 		array = []
# 		for s in range(len(state)):
# 			if state[np.argmax(state)] > 0:
# 				array.append(state[s] / state[np.argmax(state)])
# 			else:
# 				array.append(state[s] / 1)
# 		return array
#
# 	def forward_propagate(self, network, state):
# 		d = 0
# 		ret = []
# 		s = 0
#
# 		# normal_state = state
# 		normal_state = self.normalise(state[0])
# 		while s < len(normal_state): # MAPPING STATE TO INPUT NEURONS
# 			network[0]['neurons'][s].output = normal_state[s]
# 			s += 1
# 		max_output = 0
# 		while d <= self.layers:
# 			for n in range(len(network[d + 1]['neurons'])): # OUTSIDE LOOP IS CYCLING THROUGH EACH NEURON IN THE NEXT LAYER
# 				dot_product = 0.0
# 				for w in range(len(network[d]['neurons'])): # INSIDE LOOP IS CYCLING THROUGH EACH NEURON IN THE FIRST LAYER AND COMPUTING A DOT PRODUCT
# 					dot_product += network[d]['weights'][w][n] * network[d]['neurons'][w].output
# 				# if dot_product > max_output:
# 					# max_output = dot_product
# 				# p = math.tanh(dot_product)
# 				p = self.ReLU(dot_product)
# 				# print p
# 				network[d + 1]['neurons'][n].output = p
# 			d += 1
# 		arr = []
# 		for n in range(len(network[d]['neurons'])):
# 			arr.append(network[d]['neurons'][n].output)
# 		ret = self.normalise(arr)
# 		# print ret
# 		return ret
#
# 	def ReLU_derivative(self, x):
# 		y = (x > 0) * 1 + (x <= 0) * 0
# 		return y
#
# 	def calc_err(self, network, target, reward):
# 		for i in reversed(range(len(network))):
# 			layer = network[i]
# 		#error = (target - reward) * self.ReLU_derivative(reward)
# 		#print error
# 		#return error
# 		# print "err : ", error
# 		# return error
# 			errors = []
# 			if i != len(network) - 1:
# 				for l in range(len(layer)):
# 					error = 0.0
# 					for neuron in network[i + 1]['neurons']:
# 						print neuron
# 						error += neuron['weights'][][l] * layer['neurons'][l].delta
# 					errors.append(error)
# 			else:
# 				for n in range(len(layer)):
# 					neuron = layer['neurons'][n]
# 					errors.append((target - reward) * self.ReLU_derivative(layer['neurons'][n].output))
# 			for n in range(len(layer)):
# 				layer['neurons'][n].delta = errors[n] * self.ReLU_derivative(layer['neurons'][n].output)
#
# 	def update_weights(self, network, target):
# 		for i in range(len(network) - 1):
# 			inputs = target
# 			# for neuron in range(len(network[i - 1]))]
# 			for neuron in range(len(network[i + 1])):
# 				if i != len(network) - 1:
# 					inputs = network[i]['neurons'][neuron].output
# 					# print inputs
# 				for w in range(len(network[i]['weights'])):
# 					# print inputs
# 					network[i]['weights'][neuron][w] = inputs * LEARNING_RATE
# 					# print network[i]['weights'][neuron][w]
#
# 	def learn(self, batch_size):
# 		minibatch = random.sample(self.memory, batch_size)
# 		for state, action, reward, next_state, done in minibatch:
# 			target = 4200
# 			update = self.calc_err(self.network, target, reward)
# 			self.update_weights(self.network, update)
#
# 	def remember(self, state, action, reward, observation, done):
# 		self.memory.append((state, action, reward, observation, done))
# 	def sigmoid(self, x):
#   		return 1 / (1 + math.exp(-x))
# 	def ReLU(self, x):
# 		if x > 0:
# 			return x
# 		return 0
# 	def predict(self, state):
# 		return self.forward_propagate(self.network, state)














# def remember(state, action, reward, observation, done):

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	input_layer = [{'weights':[uniform(-1.0, 1.0) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(input_layer)
	hidden_layer = [{'weights':[uniform(-1.0, 1.0) for i in range(n_hidden + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	hidden_layer = [{'weights':[uniform(-1.0, 1.0) for i in range(n_hidden + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	# n_inputs = 32
	# n_hidden = 3
	# hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	# network.append(hidden_layer)
	output_layer = [{'weights':[uniform(-1.0, 1.0) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	# print inputs
	activation = weights[-1]
	# print "weights ", len(weights)
	# print "ins ", len(inputs)
	# if (len(weights) == len(inputs) - 1):
	for i in range(len(weights) - 1):
		# print "inputs tho:", inputs
		# print i
		activation += weights[i] * inputs[i]
	return activation

def transfer_derivative(x):
	return x * (1 - x)

def transfer(x):
	# if opt == 'SOFTMAX':
		# SOFTMAX
		# e_x = np.exp(x - np.max(x))
		# out = e_x / e_x.sum()
		# return out
	# elif opt == 'RELU':
		# RELU
		# return (x > 0) * 1 + (x <= 0) * 0
	# else:
		# SIGMOID
	if x < 0:
		return 1 - 1 / (1 + exp(x))
	else:
		return 1 / (1 + exp(-x))

def forward_propagate(network, row):
	inputs = row
	for layer in range(len(network)):
		new_inputs = []
		l = network[layer]
		for neuron in l:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network) - 1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				err = expected[j] - neuron['output']
				# print err
				# err2 = err**2
				# err = math.sqrt(err2 + 1) - 1
				# print err
				errors.append(err)
				# this = expected - reward
				# errors.append(math.sqrt(1 + (this * this)) - 1)
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += neuron['delta'] * inputs[j] * l_rate
			neuron['weights'][-1] += l_rate * neuron['delta']
			# print neuron['weights']

def train_network(network, train, l_rate, n_epoch, n_outputs, batch_size):
	for epoch in range(n_epoch):
		sum_error = 0
		for example in train:
			if batch_size == 0:
				break
			row = example[0]
			action = example[1]
			reward = example[2]
			obs = example[3]
			outputs = predict(network, row)
			hold = action
			out = predict(network, obs)
			# print action
			# print
			# print obs
			# print "FIRST ", action
			for i in range(n_outputs):
				# thing = obs[i]
				# if action[i] > obs[i]:
					# thing = action[i]
				action[i] = (hold[i] + (((((out[i]*2)+(GRADE*random()))/3)-hold[i]) * (reward / SCORE_CEILING)))
				if action[i] < 0:
					action[i] = 0.00
				elif action[i] > 1:
					action[i] = 0.99
				# tmp = action[i]**2
				# action[i] = math.sqrt(tmp)
				# print action[i]
			# print "SECOND ", action
			backward_propagate_error(network, action)
			update_weights(network, row, l_rate)
			batch_size -= 1
		# print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, expected[0]))

# Make a prediction with a network
def predict(network, row):
	# print row
	outputs = forward_propagate(network, row)
	# print outputs
	return outputs

def save_net(network):
	f = open('BipedalWalker-v2_weights.csv', 'w')
	writer = csv.writer(f, delimiter = ',')
	for layer in range(len(network)):
		for neuron in network[layer]:
			# weights = []
			# for w in range(len(neuron['weights'])):
			writer.writerow(neuron['weights'])
	f.close

def load_net(state_size, actions):
	network = initialize_network(state_size, 64, actions)
	weights = []
	if os.path.isfile('BipedalWalker-v2_weights.csv'):
		f = open('BipedalWalker-v2_weights.csv', 'rb')
		reader = csv.reader(f, delimiter=',')
		# for i in f:
			# weights.append(i)
		for layer in range(len(network)):
			for neuron, row in izip(network[layer], reader):
				# for row in reader:
					# print (np.array(row).astype(float)).tolist()
				neuron['weights'] = np.array(row).astype(float).tolist()
					# print neuron['weights']
	return network

if __name__ == "__main__":
	seed(datetime.now())
	env = gym.make('BipedalWalker-v2')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space
	actions = 4
	# agent = Ada(state_size, actions, HIDDEN_LAYERS, 32)
	network = load_net(state_size, actions)
	#initialize_network(state_size, 64, actions)
	done = False
	batch_size = 256
	e = 0
	bootstrap = 10000
	t_time = bootstrap * 100
	best_score = -1000000
	run_time = 1000
	memory = deque(maxlen=20000)
	grade = 100.0
	value = 0.50
	while (t_time):
		e += 1
		state = env.reset()
		'''action = env.action_space.sample()
		print "ACTION : ", action
		observation, reward, done, info = env.step(action)
		sys.exit(1)'''
		state = np.reshape(state, [1, state_size])
		total_reward = 0
		for time in range(run_time):
			# state_ = np.reshape(state, [1, state_size])
			env.render()
			# print "big ol ham"
			# print state[0]
			# print "\n\n\n"
			# array = []
			# array = predict(network, state[0])
			action = predict(network, state[0])
			# print action
			if bootstrap > 0:
				bootstrap -= 1
				for a in range(len(action)):
					if random() < GRADE:
						action[a] = random()
					# if action[a] < 0.04:
						# action[a] = 0.04
					# elif action[a] > CEILING:
						# action[a] = CEILING
			elif bootstrap == 0:
				print "\n\n\n\n\n\n\n\
				______  _____  _____  _____  _____  _____ ______   ___  ______ ______  _____  _   _  _____     ______  _   _   ___   _____  _____    _____  _____ ___  _________  _      _____  _____  _____  _ \
				| ___ \|  _  ||  _  ||_   _|/  ___||_   _|| ___ \ / _ \ | ___ \| ___ \|_   _|| \ | ||  __ \    | ___ \| | | | / _ \ /  ___||  ___|  /  __ \|  _  ||  \/  || ___ \| |    |  ___||_   _||  ___|| |\
				| |_/ /| | | || | | |  | |  \ `--.   | |  | |_/ // /_\ \| |_/ /| |_/ /  | |  |  \| || |  \/    | |_/ /| |_| |/ /_\ \\\\ `--. | |__    | /  \/| | | || .  . || |_/ /| |    | |__    | |  | |__  | |\
				| ___ \| | | || | | |  | |   `--. \  | |  |    / |  _  ||  __/ |  __/   | |  | . ` || | __     |  __/ |  _  ||  _  | `--. \|  __|   | |    | | | || |\/| ||  __/ | |    |  __|   | |  |  __| | |\
				| |_/ /\ \_/ /\ \_/ /  | |  /\__/ /  | |  | |\ \ | | | || |    | |     _| |_ | |\  || |_\ \    | |    | | | || | | |/\__/ /| |___   | \__/\\\\ \_/ /| |  | || |    | |____| |___   | |  | |___ |_|\
				\____/  \___/  \___/   \_/  \____/   \_/  \_| \_|\_| |_/\_|    \_|     \___/ \_| \_/ \____/    \_|    \_| |_/\_| |_/\____/ \____/    \____/ \___/ \_|  |_/\_|    \_____/\____/   \_/  \____/ (_)\
				\n\n\n\n\n\n\n"
				bootstrap = -1
			print action
			#c = [0.25, 0.25, 0.25, 0.25]
			observation, reward, done, _ = env.step(action)
			# observation[4] = 10
			#print action
			reward = reward if not done else -10
			# if action[4] - 0.5 < value:
				# reward -= grade
			observation = np.reshape(observation, [1, state_size])
			# print state[0]
			memory.append([state[0], action, reward, observation[0], done])
			state = observation
			# print state
			total_reward += reward
			# value = action[4]
			if done or time == 499:
				print("episode :  {}/{}, score :  {}, best score :  {}".format(e, t_time / run_time, total_reward, best_score))
				if best_score < total_reward:
					best_score = total_reward
					if abs(total_reward) > SCORE_CEILING:
						SCORE_CEILING = best_score
				break
		t_time -= 1
		GRADE *= 0.95
		if len(memory) > 1000 and randint(0, 101) < 50:
			print "LEARNING ---"
			train_network(network, memory, LEARNING_RATE, 1, actions, batch_size)
			save_net(network)
	print "Episodes : ", t_time / run_time, " Best Score : ", best_score
