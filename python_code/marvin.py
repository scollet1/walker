import gym
import numpy as np
import random
from collections import deque
import math
import copy
import sys
import time

EPISODES = 5000
DEQUE = 2000
EPSILON = 0.85
LAMBDA = 0.99
EPSILON_BASE = 0.01
HIDDEN_LAYERS = 2
NEURAL_DENSITY = 5
GAMMA = 0.04
LEARNING_RATE = 0.10

class Neuron:
	def __init__(self):
		self.output = 0.0
		self.delta = 0

	def delta(self):
		return self._delta

	def output(self):
		return self._output



class Ada:
	def __init__(self, states, actions, layers, density):
		self.actions = actions
		self.states = states
		self.layers = layers
		self.network = self._construct_network(states, actions, layers, density)
		self.memory = deque(maxlen=20000)
		self.epsilon = EPSILON
		self.e_decay = LAMBDA
		self.e_base = EPSILON_BASE
		self.gamma = GAMMA
	def layers(self):
		return self._layers
	def epsilon(self):
		return self._epsilon
	def e_decay(self):
		return self._e_decay
	def e_base(self):
		return self._e_base

	def _construct_network(self, states, actions, layers, density):
		network = []
		network.append({'neurons': [], 'weights': []})
		for s in range(states):
			network[0]['neurons'].append(Neuron())
			network[0]['weights'].append([])
			for d in range(density):
				network[0]['weights'][s].append(random.uniform(-1, 1))
				#print "THIS MANY WEIGHTS > ", d
		l = 1
		while l <= layers:
			network.append({'neurons': [], 'weights': []})
			for d in range(density):
				network[l]['neurons'].append(Neuron())
				network[l]['weights'].append([])
				for n in range(density):
					network[l]['weights'][d].append(random.uniform(-1, 1))
			l += 1
		network.append({'neurons': [], 'weights': []})
		for a in range(actions):
			#print a
			network[l]['neurons'].append(Neuron())
			network[l]['weights'].append([])
			for d in range(density):
				network[l]['weights'][a].append(random.uniform(-1, 1))
		#sys.exit(0)
		return network

	def normalise(self, state):
		array = []
		for s in range(len(state)):
			if np.argmax(state) > 0:
				array.append(state[s] / np.argmax(state))
			else:
				array.append(state[s] / 1)
		return array

	def forward_propagate(self, network, state):
		d = 0
		ret = []
		s = 0
		#print "length?? ", len(state[0])
		normal_state = self.normalise(state)
		while s < len(normal_state[0]): # MAPPING STATE TO INPUT NEURONS
			network[0]['neurons'][s].output = normal_state[0][s]
			#print "normal_states mapped to input :", network[0]['neurons'][s].output
			s += 1
		#print "STATE : ", state
		max_output = 0
		while d <= self.layers:
			for n in range(len(network[d + 1]['neurons'])): # OUTSIDE LOOP IS CYCLING THROUGH EACH NEURON IN THE NEXT LAYER
				dot_product = 0.0
				for w in range(len(network[d]['neurons'])): # INSIDE LOOP IS CYCLING THROUGH EACH NEURON IN THE FIRST LAYER AND COMPUTING A DOT PRODUCT
					#print "n, w : ", n, w
					#print "OUTPUT :", network[d]['neurons'][w].output
					#print "WEIGHT :", network[d]['weights'][w][n]
					dot_product += network[d]['weights'][w][n] * network[d]['neurons'][w].output
				if dot_product > max_output:
					max_output = dot_product
				#if d == self.layers:
				#	p = self.ReLU(dot_product / max_output)
				#else:
				p = math.tanh(dot_product)
				#print p
				#if random.random() <= p:
				network[d + 1]['neurons'][n].output = p
				#else:
				#	network[d + 1]['neurons'][n].output = 0.001
				#print "dot product @ n : ", dot_product, n
			d += 1
			#print "LAYER : ", d
		for n in range(len(network[d]['neurons'])):
			ret.append(network[d]['neurons'][n].output)
		 	#print network[d]['neurons'][n].output
		#sys.exit(0)
		return ret

	def ReLU_derivative(self, x):
		return np.maximum(0, x)

	def back_propagate(self, network, target):
		for i in reversed(range(len(network))):
			layer = network[i]
			errors = list()
			if i != len(network) - 1:
				for l in range(len(layer)):
					error = 0.0
					for neuron in range(len(network[i + 1])):
						error += (network[i + 1]['weights'][neuron][l] * network[i + 1]['neurons'][l].delta)
					errors.append(error)
			else:
				for n in range(len(layer)):
					neuron = layer['neurons'][n]
					errors.append(target[n] - neuron.output)
			for n in range(len(layer)):
				layer['neurons'][n].delta = errors[n] * self.ReLU_derivative(layer['neurons'][n].output)

	def update_weights(self, network, target):
		for i in range(len(network)):
			inputs = target[:-1]
			#print inputs
			if i != 0:
				inputs = [network[i]['neurons'][neuron].output for neuron in range(len(network[i - 1]))]
			for neuron in range(len(network[i])):
				for j in range(len(inputs)):
					#print network[i]['weights'][neuron][j]
					#print inputs[j]
					network[i]['weights'][neuron][j] += LEARNING_RATE * network[i]['neurons'][neuron].delta * inputs[j]
				network[i]['weights'][neuron][-1] += LEARNING_RATE * network[i]['neurons'][neuron].delta

	def learn(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			#print "Loading ...", i
			target = self.predict(state)
			#print "TARGET ", target
			#if done:
			#	target[np.argmax(action)] = reward
			#else:
				#expected = [0 for i in range(n_outputs)]
				#expected[row[-1]] = 1
				#a = self.predict(next_state)
			t = self.predict(next_state)
				#print t
				#print a
				#print np.argmax(a)
				#print action
			for act in range(len(action)):
				target[act] = reward + self.gamma * t[act]
			self.back_propagate(self.network, target)
			self.update_weights(self.network, target)
		if self.epsilon > self.e_base:
			self.epsilon *= self.e_decay
		#sys.exit(1)

	def remember(self, state, action, reward, observation, done):
		self.memory.append((state, action, reward, observation, done))
		#print self.memory[0][0]
		#sys.exit(1)

	def sigmoid(self, x):
  		return 1 / (1 + math.exp(-x))

	def ReLU(self, x):
		if x > 0:
			return x
		return 0

	def predict(self, state):
		#if np.random.rand() <= self.epsilon:
		#	return random.randrangeself.actions)
		return self.forward_propagate(self.network, state)

if __name__ == "__main__":
	random.seed(42)
	env = gym.make('BipedalWalker-v2')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space
	#print action_size
	#print state_size
	agent = Ada(state_size, 4, HIDDEN_LAYERS, 64)
	done = False
	batch_size = 256
	e = 0
	while (1):
		e += 1
		state = env.reset()
		'''action = env.action_space.sample()
		print "ACTION : ", action
		observation, reward, done, info = env.step(action)
		sys.exit(1)'''
		state = np.reshape(state, [1, state_size])
		total_reward = 0
		for time in range(500):
			env.render()
			#c = []
			action = agent.predict(state)
			#print action
			#print "STATE: ", state
			#c.append(action)
			c = [0.25, 0.25, 0.25, 0.25]
			observation, reward, done, _ = env.step(action)
			#print done
			reward = reward if not done else -10
			observation = np.reshape(observation, [1, state_size])
			#print state
			#print observation
			agent.remember(state, action, reward, observation, done)
			state = observation
			#print "OBSV :", observation
			total_reward += reward
			if done or time == 499:
				print("episode :  {}/{}, score :  {}".format(e, EPISODES, total_reward))
				break
		#sys.exit(0)
		if len(agent.memory) > batch_size:
			print "LEARNING ---"
			agent.learn(batch_size)
