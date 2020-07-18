import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from collections import deque

from LSM_v2 import LSM
from spike_encoding import spike_encoding

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from collections import deque

from LSM_v2 import LSM
from spike_encoding import spike_encoding

import gym


class Agent:
    def __init__(self, state_space_size, state_space_bounds, action_space_size, reservoir_size, epsilon_decay_time_constant):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        self.memory = deque(maxlen=100000)
        self.epsilon = 1
        self.epsilon_min = 0.001
        self.epsilon_decay_time_constant = epsilon_decay_time_constant
        self.gamma = 0.95
        self.batch_size = 32
        self.training_threshold = 100

        self.lsm = LSM(input_size=state_space_size, output_size=action_space_size, width=reservoir_size[0], height=reservoir_size[1], depth=reservoir_size[2])
        self.last_lsm_state = 0
        self.lsm.reset_state()

        self.spike_encoders = []
        for i in range(state_space_size):
            self.spike_encoders.append(spike_encoding(scheme='poisson_rate_coding', time_window=100, input_range=state_space_bounds[i], output_freq_range=(10,200)))

        self.spike_amplitude_scaling_constant = 50
        
    def get_spike_data(self, input_state):
        spike_train = []
        for idx,s in enumerate(input_state):            
            spike_train.append(self.spike_encoders[idx].encode(s.reshape(1,1))[0])
        spike_train = np.asarray(spike_train) * self.spike_amplitude_scaling_constant
        return spike_train

    def get_q_values(self, input_state):
        if len(input_state.shape) == 2: #Batch input
            #Convert input states to batched spike trains
            batch_spike_data = []
            for state in input_state:                   
                batch_spike_data.append(self.get_spike_data(state))
            #Obtain Q values in batch
            batch_q_values = self.lsm.predict_on_batch(np.asarray(batch_spike_data))
            return batch_q_values

        else: #Single input
            #Convert input states to spike trains
            spike_train = self.get_spike_data(input_state)            
            #Pass it through LSM
            q_values = self.lsm.predict(spike_train)            
            return q_values

    def get_action(self, input_state, epsilon = None):
        epsilon = self.epsilon if epsilon==None else epsilon

        if random.uniform(0,1) < epsilon:
            return np.random.randint(0, self.action_space_size, 1)[0]
        else:
            return np.argmax(self.get_q_values(input_state)) 

    def remember(self, info):
        self.memory.append(info)

    def replay(self):
        if len(self.memory) > self.training_threshold:
            
            #1. Randomly sample a mini_batch
            minibatch = random.sample(self.memory, self.batch_size)

            #2. Segregate the data 
            state = np.zeros((self.batch_size, self.state_space_size))
            next_state = np.zeros((self.batch_size, self.state_space_size))
            action, reward, dead = [], [], []

            for i in range(self.batch_size):
                state[i] = minibatch[i][0]
                action.append(minibatch[i][1])
                reward.append(minibatch[i][2])
                next_state[i] = minibatch[i][3]
                dead.append(minibatch[i][4])

            #3. Prepare inputs and targets for readout layer
            lsm_states = []
            target = []
            batch_spike_data =[]
            for s in state:
                spike_train = self.get_spike_data(s)
                batch_spike_data.append(spike_train)
            self.lsm.reset_state()
            lsm_states, target = self.lsm.predict_on_batch(np.asarray(batch_spike_data), output='average_firing_rate_and_q_values')
            
            self.lsm.reset_state()
            next_state_q_value = self.get_q_values(next_state)

            for i in range(self.batch_size):
                if dead[i]:
                    target[i][action[i]] = reward[i]
                else:
                    target[i][action[i]] = reward[i] + self.gamma*(np.max(next_state_q_value[i]))

            #4. Update the readout network
            self.lsm.readout_network.train_on_batch(lsm_states, target)

    def update_epsilon(self,episode):
        if len(self.memory) > self.training_threshold:
            if self.epsilon > self.epsilon_min:
                self.epsilon = np.exp(-episode/self.epsilon_decay_time_constant)

    def save_lsm(self, path=''): 
        lsm_parameters = [ self.lsm.input_weight_matrix, self.lsm.liquid_weight_matrix, self.lsm.readout_network.get_weights()]
        
        f = open(path+'best_lsm_model_parameters.pkl', 'wb')
        pickle.dump(lsm_parameters, f)
        f.close()

    def load_lsm(self, path): #[ input weight matrix, liq layer weight matrix, readout weights]

        f = open(path, 'rb')
        parameters_list = pickle.load(f)
        f.close()

        self.lsm.input_weight_matrix = parameters_list[0]
        self.lsm.liquid_weight_matrix = parameters_list[1]
        self.lsm.readout_network.set_weights(parameters_list[2])        

    def save_lsm_state(self):
        N_t, V_m, R_c = self.lsm.liquid_layer_neurons.N_t,self.lsm.liquid_layer_neurons.V_m,self.lsm.liquid_layer_neurons.R_c
        self.last_lsm_state = [N_t, V_m, R_c] 

    def resume_lsm_state(self):
        self.lsm.liquid_layer_neurons.N_t,self.lsm.liquid_layer_neurons.V_m,self.lsm.liquid_layer_neurons.R_c = self.last_lsm_state[0], self.last_lsm_state[1], self.last_lsm_state[2]
#---------------------------------------------------------------------------------------------------------------
EPISODES = 100000

env = gym.make('CartPole-v0')

state_space_bounds = [(-2.4,2.4), (-255,255), (-41.8, 41.8), (-255,255)]
agent = Agent(state_space_size=env.observation_space.shape[0], 
            state_space_bounds=state_space_bounds, 
            action_space_size=env.action_space.n, 
            reservoir_size=(6,6,6),
            epsilon_decay_time_constant=EPISODES/2)

total_scores = []
median_accumulated_reward = []
total_evaluation_scores = []

prev_eval_score = 0
prev_eval_marker = 0

while np.sum(total_scores) < 100000:   #Run episodes only till 1e5 timesteps

    print('Total_timesteps:', np.sum(total_scores), '| Epsilon:',agent.epsilon)

    state = env.reset()
    agent.lsm.reset_state()

    dead = False
    frames = 0

    while not dead:
        #env.render()
        frames += 1

        #1. Get action
        action = agent.get_action(state)

        #2. Perform action
        new_state, reward, dead, info = env.step(action)

        #3. Penalise if agent dies before episode ends
        if dead and frames!=env._max_episode_steps-1:
            reward = -10

        #4. Remember 
        agent.remember((state, action, reward, new_state, dead))

        #6. Update state for next frame
        state = new_state
        
        #7. Replay
        if len(agent.memory) > agent.training_threshold:
            agent.save_lsm_state()
            agent.replay()
            agent.resume_lsm_state()
        
        if dead:
            total_scores.append(frames)         
      
    #Evaluate agent after every 1000 timesteps(/frames)
    if np.sum(total_scores) > prev_eval_score + 1000:


        median_accumulated_reward.append(np.median(total_scores[prev_eval_marker:]))
        print('Median Accumulated Reward in last 1000 timesteps:',median_accumulated_reward[-1])
        prev_eval_marker = len(total_scores)
        
        print('Evaluating...')
        
        agent.save_lsm()
        prev_eval_score = np.sum(total_scores)
        agent.epsilon -= 0.02

        eval_scores, num_of_games = 0, 0
        
        while eval_scores < 1000:   #Evaluate for 1000 timesteps
            print('.',end='')

            state = env.reset()
            agent.lsm.reset_state()
            score = 0
            dead = False

            while not dead:
                #Get action
                action = agent.get_action(state, epsilon = 0.05) #Exploit
                #2. Perform action
                new_state, reward, dead, info = env.step(action)
                #Update state
                state = new_state

                #Accumulate score over this gameplay
                score += 1

                if dead:
                    eval_scores += score
                    num_of_games += 1
        print()
        total_evaluation_scores.append(eval_scores/num_of_games)
        print('Evaluation Score: ', total_evaluation_scores[-1])

        plt.subplot(1,2,1)
        plt.title('Median accumulated Reward')
        plt.plot(median_accumulated_reward)
        plt.subplot(1,2,2)
        plt.title('Evaluation scores')
        plt.plot(total_evaluation_scores)
        plt.show()

    
env.close()