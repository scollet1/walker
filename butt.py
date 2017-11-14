import time, math, random, bisect, copy, os
import gym
import numpy as np

GYM = 'BipedalWalker-v2'
MAX_GENERATIONS = 500
POPULATION_COUNT = 25
MUTATION_RATE = 0.01

class Network:
    def __init__(self, nodeCount, loadFile):
        self.fitness = 0.0
        self.nodeCount = nodeCount
        self.weights = []
        self.biases = []

        if loadFile:
            self.nodeCount, self.weights, self.biases = loadWeights()
        else:
            for i in range(len(nodeCount) - 1):
                self.weights.append(np.random.uniform(-1, 1, size=(nodeCount[i], nodeCount[i+1])).tolist() )
                self.biases.append(np.random.uniform(-1, 1, size=(nodeCount[i+1])).tolist())

    def getOutput(self, input):
        output = input
        for i in range(len(self.nodeCount)-1):
            output = np.reshape(np.matmul(output, self.weights[i]) + self.biases[i], (self.nodeCount[i+1]))
            output = np.maximum(output, 0)
        return np.subtract(np.multiply(sigmoid(output), 2), 1)

class Population:
    def __init__(self, populationCount, mutationRate, nodeCount, loadFile=False):
        self.nodeCount = nodeCount
        self.popCount = populationCount
        self.m_rate = mutationRate
        if loadFile:
            print("Loading Values from File\n")
        self.population = [Network(nodeCount, False) for i in range(populationCount)]


    def createChild(self, nn1, nn2):
        child = Network(self.nodeCount, False)
        for i in range(len(child.weights)):
            for j in range(len(child.weights[i])):
                for k in range(len(child.weights[i][j])):
                    if random.random() > self.m_rate:
                        if random.random() < nn1.fitness / (nn1.fitness+nn2.fitness):
                            child.weights[i][j][k] = nn1.weights[i][j][k]
                        else :
                            child.weights[i][j][k] = nn2.weights[i][j][k]
        for i in range(len(child.biases)):
            for j in range(len(child.biases[i])):
                if random.random() > self.m_rate:
                    if random.random() < nn1.fitness / (nn1.fitness+nn2.fitness):
                        child.biases[i][j] = nn1.biases[i][j]
                    else:
                        child.biases[i][j] = nn2.biases[i][j]
        return child


    def createNewGeneration(self):
        nextGen = []
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        for i in range(self.popCount):
            if random.random() < float(self.popCount-i)/self.popCount:
                nextGen.append(copy.deepcopy(self.population[i]));

        fitnessSum = [0]
        minFit = min([i.fitness for i in nextGen])
        for i in range(len(nextGen)):
            fitnessSum.append(fitnessSum[i]+(nextGen[i].fitness-minFit)**4)


        while(len(nextGen) < self.popCount):
            r1 = random.uniform(0, fitnessSum[len(fitnessSum)-1] )
            r2 = random.uniform(0, fitnessSum[len(fitnessSum)-1] )
            i1 = bisect.bisect_left(fitnessSum, r1)
            i2 = bisect.bisect_left(fitnessSum, r2)
            if 0 <= i1 < len(nextGen) and 0 <= i2 < len(nextGen) :
                nextGen.append( self.createChild(nextGen[i1], nextGen[i2]) )
            else :
                print("Index Error ");
                print("Sum Array =",fitnessSum)
                print("Randoms = ", r1, r2)
                print("Indices = ", i1, i2)
        del self.population[:]
        self.population = nextGen

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def mapRange(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

def normalizeArray(aVal, aMin, aMax):
    res = []
    for i in range(len(aVal)):
        res.append( mapRange(aVal[i], aMin[i], aMax[i], -1, 1) )
    return res

def scaleArray(aVal, aMin, aMax):
    res = []
    for i in range(len(aVal)):
        res.append( mapRange(aVal[i], -1, 1, aMin[i], aMax[i]) )
    return res

def replayBestBots(bestNeuralNets, steps, sleep):
    choice = input("Do you want to watch the replay ?[Y/N] : ")
    if choice=='Y' or choice=='y':
        for i in range(len(bestNeuralNets)):
            if (i+1)%steps == 0 :
                observation = env.reset()
                totalReward = 0
                for step in range(MAX_STEPS):
                    env.render()
                    time.sleep(sleep)
                    action = bestNeuralNets[i].getOutput(observation)
                    observation, reward, done, info = env.step(action)
                    # reward = reward if not done else -101
                    totalReward += reward
                    if done:
                        break
                print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %4d" % (i+1, bestNeuralNets[i].fitness, totalReward))

def recordBestBots(bestNeuralNets):
    env.monitor.start('OpenAI/'+GYM+"/Data", force=True )
    for i in range(len(bestNeuralNets)):
        totalReward = 0
        observation = env.reset()
        for step in range(MAX_STEPS):
            env.render()
            action = bestNeuralNets[i].getOutput(observation)
            observation, reward, done, info = env.step(action)
            totalReward += reward
            if done:
                break
        print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %4d" % (i+1, bestNeuralNets[i].fitness, totalReward))
    env.monitor.close()

def loadWeights():

    nodeCount = []
    weights = []
    biases = []
    try :
        f = open('OpenAI/'+GYM+"/Data/"+GYM+"_weights.txt", 'r')
    except FileNotFoundError:
        print("File Not Found. Initializing random values")
        nodeCount = node_per_layer
        for i in range(len(nodeCount) - 1):
            weights.append(np.random.uniform(low=-1, high=1, size=(nodeCount[i], nodeCount[i+1])).tolist())
            biases.append(np.random.uniform(low=-1, high=1, size=(nodeCount[i+1])).tolist())
        return nodeCount, weights, biases
    f.readline()
    nodeCount = [int(i) for i in f.readline().split()]
    f.readline()
    for i in range(len(nodeCount) - 1):
        weights.append([])
        for j in range(nodeCount[i]):
            weights[i].append([float(i) for i in f.readline().split()])
    f.readline()
    for i in range(len(nodeCount) - 1):
        biases.append([float(i) for i in f.readline().split()])
    f.close()
    return nodeCount, weights, biases

def saveWeights(best):
    f = open(GYM+"_weights.txt", 'w')
    for i in range(len(best.weights)):
            for j in range(len(best.weights[i])):
                for k in range(len(best.weights[i][j])):
                    print("%+.2f " % best.weights[i][j][k], f, "")
    print("Biases : ", f)
    for i in range(len(best.biases)):
            for j in range(len(best.biases[i])):
                print("%+.2f " % best.biases[i][j], f, "")
    f.close()


if __name__ == "__main__":
    env = gym.make(GYM)

    MAX_STEPS = env.spec.timestep_limit

    in_dimen = env.observation_space.shape[0]
    out_dimen = env.action_space.shape[0]
    obsMin = env.observation_space.low
    obsMax = env.observation_space.high
    actionMin = env.action_space.low
    actionMax = env.action_space.high
    node_per_layer = [in_dimen, 21, 13, 8, out_dimen]

    pop = Population(POPULATION_COUNT, MUTATION_RATE, node_per_layer, True)
    bestNeuralNets = []
    bootstrap = 200000

    try :
        for gen in range(MAX_GENERATIONS):
            genAvgFit = 0.0
            minFit =  1000000
            maxFit = -1000000
            maxNeuralNet = None
            for i, nn in enumerate(pop.population):
                state = env.reset()
                totalReward = 0
                for step in range(MAX_STEPS):
                    #if i == 0 and gen % 10 == 0:
                    env.render()
                    action = nn.getOutput(state)
                    # if bootstrap > 0:
        				# bootstrap -= 1
        				# for a in range(len(action)):
        					# num = action[a] + random.random()
        					# action[a] = num / 2
                    print action
                    state, reward, done, info = env.step(action)
                    # reward = reward if not done else -101
                    totalReward += reward
                    if done:
                        break

                nn.fitness = totalReward
                minFit = min(minFit, nn.fitness)
                genAvgFit += nn.fitness
                if nn.fitness > maxFit :
                    maxFit = nn.fitness
                    maxNeuralNet = copy.deepcopy(nn);

            bestNeuralNets.append(maxNeuralNet)
            genAvgFit /= pop.popCount
            print("Generation : %3d  |  Min : %5.0f  |  Avg : %5.0f  |  Max : %5.0f  " % (gen+1, minFit, genAvgFit, maxFit) )
            pop.createNewGeneration()

        recordBestBots(bestNeuralNets)
        uploadSimulation()
        replayBestBots(bestNeuralNets, max(1, int(math.ceil(MAX_GENERATIONS/10.0))), 0)

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt")
    except Exception as e:
        print("\nUnknown Exception")
        print(str(e))
    finally :
        if len(bestNeuralNets) > 1:
            print("\nSaving Weights to file")
            saveWeights(bestNeuralNets[-1])
        else:
            print("No weights to save")
        print("\nQuitting")
