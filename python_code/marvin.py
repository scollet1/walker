import gym
import numpy as np
import random
from collections import deque
import math
import copy
import sys

EPISODES = 5000
DEQUE = 2000
EPSILON = 0.99
LAMBDA = 0.95
EPSILON_BASE = 0.01
HIDDEN_LAYERS = 2
NEURAL_DENSITY = 5

class Neuron:
	def __init__(self):
		self.input = 0.0
		self.output = 0.0

	def input(self):
		return self._input

	def output(self):
		return self._output



class Ada:
	def __init__(self, states, actions, layers, density):
		self.actions = actions
		self.states = states
		self.layers = layers
		self.network = self._construct_network(states, actions, layers, density)
		self.epsilon = EPSILON
		self.e_decay = LAMBDA
		self.e_base = EPSILON_BASE
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
				network[0]['weights'][s].append(random.random())
				#print "THIS MANY WEIGHTS > ", d
		l = 1
		while l <= layers:
			network.append({'neurons': [], 'weights': []})
			for d in range(density):
				network[l]['neurons'].append(Neuron())
				network[l]['weights'].append([])
				for n in range(density):
					network[l]['weights'][d].append(random.random())
			l += 1
		network.append({'neurons': [], 'weights': []})
		for a in range(actions):
			network[l]['neurons'].append(Neuron())
			network[l]['weights'].append([])
			for d in range(density):
				network[l]['weights'][a].append(random.random())
		return network

	def forward_propogate(self, network, state):
		d = 0
		ret = []
		s = 0
		#print "length?? ", len(state[0])
		while s < len(state[0]):
			network[0]['neurons'][s].output = state[0][s]
			#print network[0]['neurons'][s].output
			s += 1
		#print "STATE : ", state
		while d <= self.layers:
			for n in range(len(network[d + 1]['neurons'])): # OUTSIDE LOOP IS CYCLING THROUGH EACH NEURON IN THE NEXT LAYER
				dot_product = 0.0
				for w in range(len(network[d]['neurons'])): # INSIDE LOOP IS CYCLING THROUGH EACH NEURON IN THE FIRST LAYER
					#print "n, w : ", n, w
					dot_product += network[d]['weights'][w][n] * network[d]['neurons'][w].output
				network[d + 1]['neurons'][n].output = self.ReLU(dot_product)
				print "dot product : ", dot_product
				print n
			d += 1
			#print "LAYER : ", d
		for n in range(len(network[d]['neurons'])):
			ret.append(network[d]['neurons'][n].output)
		return ret

	def ReLU(self, x):
		if x > 0:
			return x
		return 0

	def predict(self, state):
		#if np.random.rand() <= self.epsilon:
		#	return random.randrangeself.actions)
		potential = self.forward_propogate(self.network, state)
		#print potential
		return potential

	def act(self, state):
		action = self.predict(state)
		print "ACTION TAKEN : ", action
		return action

if __name__ == "__main__":
	random.seed(42)
	env = gym.make('BipedalWalker-v2')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space
	#print action_size
	#print state_size
	agent = Ada(state_size, 4, HIDDEN_LAYERS, 64)
	done = False
	batch_size = 64
	for e in range(EPISODES):
		state = env.reset()

		'''action = env.action_space.sample()
		print "ACTION : ", action
		observation, reward, done, info = env.step(action)
		sys.exit(1)'''
		state = np.reshape(state, [1, state_size])
		for time in range(500):
			c = []
			action = agent.act(state)
			#c.append(action)
			observation, reward, done, _ = env.step(c)
			reward = reward if not done else -10
			observation = np.reshape(observation, [1, state_size])
			agent.remember(state, action, reward, observation, done)
			state = observation
			if done:
				agent.update_target_model()
				print("episode: {}/{}, score: {}, e: {:2}".format(e, EPISODES, time, agent.epsilon))
				break
		env.render()
		if len(agent.memory) > batch_size:
			agent.learn(batch_size)
