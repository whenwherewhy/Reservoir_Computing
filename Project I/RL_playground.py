import matplotlib.pyplot as plt
import numpy as np

from LSM import LSM
from spike_encoding import spike_encoding
import gym

class Agent:
    def __init__(self, state_space_size, state_space_bounds, action_space_size, reservoir_size):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        self.lsm = LSM(input_size=state_space_size, output_size=action_space_size, width=reservoir_size[0], height=reservoir_size[1])
        self.lsm.reset()

        self.spike_encoders = []
        for i in range(state_space_size):
            self.spike_encoders.append(spike_encoding(scheme='rate_coding', time_window=100, input_range=state_space_bounds[i], output_freq_range=(10,200)))
        

    def get_action(self, input_state):
        #Convert input states to spike trains
        spike_train = []
        for idx,s in enumerate(input_state):
            spike_train.append(self.spike_encoders[idx].encode(s.reshape(1,1))[0])
        spike_train = np.asarray(spike_train) * 50
        
        #Pass it through LSM
        output_vector = []
        for t in range(spike_train.shape[1]):
            output_vector.append(self.lsm.predict(spike_train[:,t]))
        output_vector = np.asarray(output_vector).T
        
        #Action with max amplitude is output
        output = [np.sum(x) for x in output_vector]
        return np.argmax(output)


env = gym.make('CartPole-v0')
state_space_bounds = [(-2.4,2.4), (-255,255), (-41.8, 41.8), (-255,255)]
agent = Agent(state_space_size=4, state_space_bounds=state_space_bounds, action_space_size=2, reservoir_size=(10,10))


state = env.reset()

for episode in range(100):
    
    env.render()

    action = agent.get_action(state)

    state, reward, dead, info = env.step(action)

env.close()
    #print(state, reward, dead, info)
