import gym
import numpy as np
import random
from collections import deque
import math
import copy

EPISODES = 5000
DEQUE = 2000
LAMBDA = 0.01
HIDDEN_LAYERS = 2
NEURAL_DENSITY = 5


class Model():
	def __init__(self, layers, density, actions, states):
		self.network = self._build_network(layers, density, actions, states)
		self.density = density
		self.layers = layers
		self.density = density
		self.actions = actions
		self.states = states

	def _build_network(self, layers, density, actions, states):
		network = []
		neurons = density
		i = 1
#		print "ACTIONS: ", actions
		for l in range(layers + 2):
			network.append([])
		for s in range(states):
#			print s
			network[0].append([])
			for d in range(density):
                            dictionary = {'delta': 0.0, 'weights': []}
                            network[0][s].append(copy.deepcopy(dictionary))
#                           print network[0][s][0]['delta']
                            for n in range(density):
                                   network[0][s][0]['weights'].append(random.uniform(-1, 1))
		while i < layers:
			network.append([])
			for n in range(neurons):
				network[i].append([])
			        for d in range(density):
                                    dictionary = {'delta': 0.0, 'weights': []}
                                    network[i][n].append(copy.deepcopy(dictionary))
                                    for s in range(neurons):
                                        network[i][n][0]['weights'].append(random.uniform(-1, 1))
			i += 1
		network.append([])
#		print density
		for d in range(density):
#			print "d", d
			network[i].append([])
			for a in range(actions):
                            dictionary = {'delta': 0.0, 'weights': []}
                            network[i][d].append(copy.deepcopy(dictionary))
                            for n in range(density):
				network[i][d][0]['weights'].append(random.uniform(-1, 1))
		return network

	def layers(self):
		return self._layers

	def ReLU(x):
	    return x * (x > 0)

	def forward_feed(self, state):
		input_vector = state
		for layer in self.network:
			output_vector = np.dot(input_vector, self.network[layer])
			input_vector = np.tanh(output_vector)

	def update_weights(self):
		for layers in range(self.layers):
			self.network[layers] -= LAMBDA * 1

        def derivative(output):
            if output > 0:
                return output
            return 0

        def back_prop(network):
            for l in reversed(range(len(network))):
                layer = network[l]
                errors = []
                if i != len(network):
                    for j in range(len(layer)):
                        error = 0.0
                        for neuron in network[i + 1]:
                            error = (neuron * neuron)

        def learn(network):
            pass            

	def activated(weight, neuron):
		pass

	def predict(state, epsilon):
		x = forward_feed(state)
		return x

class Ada:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=DEQUE)
		self.gamma = 0.95
		self.epsilon = 0.99
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.99
		self.model = self._build_model()
		self.target_model = self._build_model()
		self.update_target_model()

#	def _huber_loss(self, target, prediction):
#		error = prediction - target
#		return np.mean(math.sqrt(1 + math.square(error)) - 1, axis = -1)

	def _build_model(self):
		model = Model(HIDDEN_LAYERS + 2, NEURAL_DENSITY, self.action_size, self.state_size)
		return model

	def update_target_model(self):
		self.target_model.learn(self.target_model.network)

	def remember(self, state, action, erward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = self.model.predict(state)
			if done:
				target[0][action] = reward
			else:
				a = self.model.predict(next_state)[0]
				t = self.target_model.predict(next_state)[0]
				target[0][action] = reward + self.gamma * t[np.argmax(a)]
			self.model.fit(state, target, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay






	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)



if __name__ == "__main__":
	env = gym.make('Marvin-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space
	print action_size
        print state_size
	agent = Ada(state_size, 4)
	done = False
	batch_size = 64
	for e in range(EPISODES):
		env.render()
		state = env.reset()
                print state
		state = np.reshape(state, [1, state_size])
		env.render()
		for time in range(500):
			action = agent.act(state, agent.epsilon)
			next_state, reward, done, _ = env.step(action)
			reward = reward if not done else -10
			next_state = np.reshape(next_state, [1, state_size])
			agent.remember(state, action, reward, next_state, done)
			state = next_state
			if done:
				agent.update_target_model()
				print("episode: {}/{}, score: {}, e: {:2}".format(e, EPISODES, time, agent.epsilon))
				break
		if len(agent.memory) > batch_size:
			agent.replay(batch_size)
