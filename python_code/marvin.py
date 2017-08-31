import gym
import numpy as np
import random
from collections import deque
import math
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

'''	def _build_input_layer(self, states, network):
		for s in range(states):
			#print s	
			input_vector.append([density])
			for d in range(density):
				print density
				print d				
				input_vector[s].append(random.random())
		return input_vector

	def _build_output_layer(self, actions, density):
		output_vector = [density]
		for d in range(denisty):
			output_vector.append([actions])
			for a in range(actions):
				output_vector[d].append(random.random())
		return output_vector

	def _build__hidden_layers(neurons, weights):
		hidden_network = [neurons]
		for n in range(neurons):
			hidden_network.append([weights])
			for w in range(len(weights)):
				hidden_network[n].append(random.random())
		return hidden_network''' # DON'T EVEN NEED IT

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
				network[0][s].append(random.random())
		while i < layers:
			network.append([])
			for n in range(neurons):
				network[i].append([])
			for d in range(density):
				network[i][n].append(random.random())
			i += 1
		network.append([])
#		print density
		for d in range(density):
#			print "d", d
			network[i].append([])
			for a in range(actions):
				network[i][d].append(random.random())
		return network

	def layers(self)
		return self._layers

	def ReLU(x):
	    return x * (x > 0)
	
	def forward_feed(self, state):
		input_vector = state
		for layer in self.network:
			output_vector = np.dot(input_vector, self.network[layer])
			input_vector = np.tanh(output_vector)
		
	def update_weights(self):
		for layers in range(self.layers)
			self.network[layers] -= LAMBDA * 

	def back_prop(self):
		delta = 0
		delta = np.multiply(-())	
	
	def activated(weight, neuron):
		pass
	
	def predict(state, epsilon):
		return x = forward_feed(state):
		#if random.random() <= epsilon:
		#		if activated(weight, neuron):
		#			sum_total += 1
		

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
	
	def _huber_loss(self, target, prediction):
		error = prediction - target
		return K.mean(K.sqrt(1 + K.square(error)) - 1, axis = -1)

	def _build_model(self):
		model = Model(HIDDEN_LAYERS + 2, NEURAL_DENSITY, self.action_size, self.state_size)	
		#model = Sequential()
		#model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		#model.add(Dense(24, activation='relu'))
		#model.add(Dense(self.action_size, activation='linear'))
		#model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))
		return model

	def update_target_model(self):
		self.target_model.back_prop():
	
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
	agent = Ada(state_size, 4)
	done = False
	batch_size = 64
	for e in range(EPISODES):		
		env.render()
		state = env.reset()
		state = numpy.reshape(state, [1, state_size])
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
	
