#This implementation follows the startegy proposed in "RL with Low complexity LSM_2019"

import numpy as np
import matplotlib.pyplot as plt
import random
import time

from neuron_models import Dense_LIF

from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model

class LSM(object):
    def __init__(self, input_size, width, height, depth, output_size, num_of_input_duplicates=10, exc_to_inh_ratio=4, C=4, K=4):
                
        self.input_size = input_size
        self.width = width
        self.height = height
        self.depth = depth
        self.output_size = output_size

        #Liquid Layer-------------------------------------------------
        self.num_of_liquid_layer_neurons = width * height * depth
        self.N_t = np.zeros((self.num_of_liquid_layer_neurons,)) #state vector

        self.liquid_weight_matrix = np.zeros((self.num_of_liquid_layer_neurons, self.num_of_liquid_layer_neurons)) #(From X To)
        self.liquid_layer_neurons = Dense_LIF(num_of_neurons = self.num_of_liquid_layer_neurons, Vth=0.5, V_rest=0, tau_m=20, Rm = 20, tau_ref=1)

        #Number of excitatory and inhibitory neurons and their ids
        num_of_excitatory_neurons = int((exc_to_inh_ratio/(1+exc_to_inh_ratio))*self.num_of_liquid_layer_neurons)
        num_of_inhibitory_neurons = self.num_of_liquid_layer_neurons - num_of_excitatory_neurons

        excitatory_neurons_ids = [x for x in np.sort(random.sample([i for i in range(self.num_of_liquid_layer_neurons)], num_of_excitatory_neurons))]
        inhibitory_neurons_ids = []
        for idx in range(self.num_of_liquid_layer_neurons):
            if idx not in excitatory_neurons_ids:
                inhibitory_neurons_ids.append(idx)
        
        self.excitatory_neurons_ids = excitatory_neurons_ids            
        self.inhibitory_neurons_ids = inhibitory_neurons_ids

        #Create I-->E and E-->I connections
        for col in range(self.liquid_weight_matrix.shape[1]):
            #if the post-synaptic neuron is excitatory
            if col in excitatory_neurons_ids:
                rand_ids = random.sample(inhibitory_neurons_ids, C)
                self.liquid_weight_matrix[rand_ids, col] = 3 # I->E
            #if the post-synaptic neuron is inhibitory
            else:
                rand_ids = random.sample(excitatory_neurons_ids, C)
                self.liquid_weight_matrix[rand_ids, col] = 4 # E->I

        #Create E-->E and I--->I connections (1 & 2)
        for i in range(self.liquid_weight_matrix.shape[0]):
            for j in range(self.liquid_weight_matrix.shape[1]):
                if i!=j : #No self recurrence
                    # E --> E
                    if i in excitatory_neurons_ids and j in excitatory_neurons_ids:
                        I_k_1 = [x for x in np.where(self.liquid_weight_matrix[i,:]==4)[0]] #indices of inhibitory neurons that E_i connects to
                        I_k_2 = [x for x in np.where(self.liquid_weight_matrix[:,j]==3)[0]] #indices of inhibitory neurons that connects to E_j
                        intersection = list(set(I_k_1).intersection(I_k_2))
                        if len(intersection) > 0:
                            self.liquid_weight_matrix[i][j] = 1

                    # I --> I
                    elif i in inhibitory_neurons_ids and j in inhibitory_neurons_ids:
                        E_k_1 = [x for x in np.where(self.liquid_weight_matrix[i,:]==3)[0]] #indices of excitatory neurons that I_i connects to
                        E_k_2 = [x for x in np.where(self.liquid_weight_matrix[:,j]==4)[0]] #indices of excitatory neurons that connects to I_j
                        intersection = list(set(E_k_1).intersection(E_k_2))
                        if len(intersection) > 0:
                            self.liquid_weight_matrix[i][j] = 2
        
        #Allocate weights to the particular connection types
        for i in range(self.liquid_weight_matrix.shape[0]):
            for j in range(self.liquid_weight_matrix.shape[1]):
                if self.liquid_weight_matrix[i][j] == 1:    #E-->E
                    self.liquid_weight_matrix[i][j] = random.uniform(0,0.05)
                elif self.liquid_weight_matrix[i][j] == 2:  #I-->I
                    self.liquid_weight_matrix[i][j] = random.uniform(0,0.25)
                elif self.liquid_weight_matrix[i][j] == 3:  #I-->E
                    self.liquid_weight_matrix[i][j] = random.uniform(0,0.3)
                elif self.liquid_weight_matrix[i][j] == 4:  #E-->I
                    self.liquid_weight_matrix[i][j] = random.uniform(0,0.01)

        #Input Layer--------------------------------------------------
        intermediate_input_size = self.input_size * num_of_input_duplicates
        self.pseudo_input_weight_matrix = np.zeros((intermediate_input_size, self.input_size)) #Creates 10 copies of each input dimension

        for i in range(self.input_size):
            self.pseudo_input_weight_matrix[i*num_of_input_duplicates:(i+1)*num_of_input_duplicates,i] = np.ones((num_of_input_duplicates,))

        self.input_weight_matrix = np.zeros((self.num_of_liquid_layer_neurons, intermediate_input_size))

        for neuron_id in range(self.num_of_liquid_layer_neurons):
            if neuron_id in excitatory_neurons_ids: 
                rand_ids = np.random.randint(0, intermediate_input_size, K)
                for r in rand_ids:
                    self.input_weight_matrix[neuron_id, r] = random.uniform(0, 0.6)

        #Read-Out Layer-----------------------------------------------
        self.readout_network = self.get_readout_network()

    def get_readout_network(self):

        x_in = Input(shape=(len(self.excitatory_neurons_ids),))
        x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x_in)
        x = Dense(self.output_size, activation='linear',kernel_initializer='he_uniform')(x)

        model = Model(x_in, x)
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        return model 

    def reset_state(self):
        #Reset Neurons
        self.liquid_layer_neurons.reset()

        #Reset state vectors
        self.N_t = np.zeros((self.num_of_liquid_layer_neurons,)) #state vector
 
    def predict_on_batch(self, input_state, output='q_values'): #Input shape : (Batch, Input_dim, timesteps)
        assert len(input_state.shape) == 3

        batch_average_firing_rate = []
        for b in range(input_state.shape[0]):
            self.reset_state()
            activation = [] #activation of LSM over a single input spike train duration
            for t in range(input_state.shape[1]):                
                pseudo_input_current = np.dot(self.pseudo_input_weight_matrix, input_state[b,:,t])
                input_current = np.dot(self.input_weight_matrix, pseudo_input_current)                
                past_current = np.dot(self.liquid_weight_matrix.T, self.N_t)                 
                total_current = input_current + past_current                
                self.N_t, _, _ = self.liquid_layer_neurons.update(total_current)            
                activation.append(self.N_t)
            #Calculate average firing rate for this single input spike train
            batch_average_firing_rate.append(np.sum(np.asarray(activation).T[self.excitatory_neurons_ids], axis=1) / input_state.shape[-1])
        batch_average_firing_rate = np.asarray(batch_average_firing_rate)
        #Output
        if output == 'q_values':
            #Feed forward the obtained activations into Q_network
            return self.readout_network.predict(batch_average_firing_rate)

        elif output == 'average_firing_rate':
            #Return average firing rate of each liquid layer neuron
            return batch_average_firing_rate

        elif output == 'average_firing_rate_and_q_values':
            #return both average firing rate and q_values
            return [batch_average_firing_rate, self.readout_network.predict(batch_average_firing_rate)]

    def predict(self, input_state, output='q_values'):

        activation = [] #activation of LSM over entire input spike train duration

        for t in range(input_state.shape[1]):
            
            pseudo_input_current = np.dot(self.pseudo_input_weight_matrix, input_state[:,t])
            input_current = np.dot(self.input_weight_matrix, pseudo_input_current)
            
            past_current = np.dot(self.liquid_weight_matrix.T, self.N_t) 
            
            total_current = input_current + past_current
            
            self.N_t, _, _ = self.liquid_layer_neurons.update(total_current)            
            activation.append(self.N_t)
        activation = np.asarray(activation).T   #Shape : N x T

        #Calculate average firing rate of each neuron during the entire input duration
        average_firing_rate = np.sum(activation[self.excitatory_neurons_ids], axis=1) / input_state.shape[1]

        #Output
        if output == 'q_values':
            #Feed forward the obtained activations into Q_network
            q = self.readout_network.predict(np.expand_dims(average_firing_rate, axis=0))
            return q

        elif output == 'average_firing_rate':
            #Return average firing rate of each liquid layer neuron
            return average_firing_rate

        elif output == 'average_firing_rate_and_q_values':
            #return both average firing rate and q_values
            return [average_firing_rate, self.readout_network.predict(np.expand_dims(average_firing_rate, axis=0))]

        elif output == 'ST_lsm_state':
            #Return the spatiotemporal state of the lsm over input duration
            return activation