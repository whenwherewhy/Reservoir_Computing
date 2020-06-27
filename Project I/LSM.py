import numpy as np
import matplotlib.pyplot as plt
from neuron_models import LIF

class LSM:
    def __init__(self, input_size, width, height, output_size, 
                 lamda=5, primary_to_auxiliary_ratio=0.6, input_feature_selection_sparsity=0.6):
        self.input_size = input_size
        self.width = width
        self.height = height
        self.output_size = output_size

        self.num_of_neurons = self.width * self.height
        self.liquid_layer_neurons = [LIF(tau_m=1) for _ in range(self.num_of_neurons)]
        
        #create dictioanries : neurons <=> coordinates
        self.neuron_to_coordinate = {}
        for i in range(self.height):
            for j in range(self.width):
                self.neuron_to_coordinate[self.liquid_layer_neurons[i*self.width+j]] = (i,j)
        self.coordinate_to_neuron = {}
        for i in range(self.height):
            for j in range(self.width):
                self.coordinate_to_neuron[(i,j)] = self.liquid_layer_neurons[i*self.width+j]

        #create dictionary : neuron -> excitory ('E') / Inhibitory ('I')
        self.get_neuron_type = {}
        self.neuron_type_ratio = 0.8 #60% Excitory & 40% Inhibitory
        excitory_neuron_indices = np.random.randint(0, self.num_of_neurons, int(self.neuron_type_ratio*self.num_of_neurons))
        for n in range(self.num_of_neurons):
            if n in excitory_neuron_indices:
                self.get_neuron_type[self.liquid_layer_neurons[n]] = 'E'
            else:
                self.get_neuron_type[self.liquid_layer_neurons[n]] = 'I'

        #create input and output weight matrices (considering only excitory synapses)
        self.primary_to_auxiliary_ratio = primary_to_auxiliary_ratio #default: 60% of the liquid layer neurons receive inputs
        self.input_feature_selection_sparsity = input_feature_selection_sparsity #default:40% of input vector will be alotted to a primary neuron
        self.input_weight_matrix = np.zeros((self.num_of_neurons, self.input_size))        	
        self.primary_neuron_indices = np.random.randint(0, self.num_of_neurons, int(self.primary_to_auxiliary_ratio*self.num_of_neurons))
        for n in self.primary_neuron_indices:
            input_vector_indices = np.random.randint(0, self.input_size, int((1-self.input_feature_selection_sparsity)*self.input_size))
            for i in input_vector_indices:
                self.input_weight_matrix[n][i] = np.random.uniform(0, 1, 1)

        self.output_weight_matrix = np.random.uniform(0, 1, size=(self.output_size, self.num_of_neurons))

        #create liquid layer weight matrix
        self.liquid_weight_matrix = np.zeros((self.num_of_neurons, self.num_of_neurons))
        self.C = {'EE': 0.6,'EI': 1,'II': 0.2,'IE': 1}
        self.lamda = {'EE': lamda,'EI': lamda,'II': lamda,'IE': lamda}
        self.synaptic_strength = {'EE': 3,'EI': 3,'II': -1,'IE': -4}    #Inhibitory synapses are given negetive strengths
        probability_of_connection = []
        for n_1 in self.liquid_layer_neurons:
            temp = []
            for n_2 in self.liquid_layer_neurons:
                connection_type = self.get_neuron_type[n_1]+self.get_neuron_type[n_2]
                d = np.sqrt(((self.neuron_to_coordinate[n_1][0]-self.neuron_to_coordinate[n_2][0])**2)+((self.neuron_to_coordinate[n_1][1]-self.neuron_to_coordinate[n_2][1])**2))
                prob = self.C[connection_type]*np.exp((-d/self.lamda[connection_type])**2)
                temp.append(prob)
            probability_of_connection.append(temp)
        #normalize probability of connections
        probability_of_connection /= np.max(probability_of_connection)
        for i,n_1 in enumerate(self.liquid_layer_neurons):
            for j,n_2 in enumerate(self.liquid_layer_neurons):
                rand = np.random.uniform(0.000, 1.000, 1)
                if rand < probability_of_connection[i][j]:
                    connection_type = self.get_neuron_type[n_1]+self.get_neuron_type[n_2]
                    self.liquid_weight_matrix[i][j] = self.synaptic_strength[connection_type]
        #Normalize sum of excitory and sum of inhibitory synapses for each neuron in liquid layer
        for c in range(self.liquid_weight_matrix.shape[1]):
            pre_synapses = self.liquid_weight_matrix[:,c]
            total_excitory_syn_strength, total_inhibitory_syn_strength = 0, 0
            for s in pre_synapses:
                if s>=0:
                    total_excitory_syn_strength += s
                else:
                    total_inhibitory_syn_strength +=s
            for s_idx in range(pre_synapses.shape[0]):
                if pre_synapses[s_idx]>=0:
                    pre_synapses[s_idx] /= total_excitory_syn_strength
                else:
                    pre_synapses[s_idx] /= -total_inhibitory_syn_strength
            self.liquid_weight_matrix[:,c] = pre_synapses[:]

        print(self.liquid_weight_matrix)
        
        '''
        plt.figure(figsize=(10,10), dpi=200)
        plt.imshow(self.liquid_weight_matrix, cmap='gray')
        plt.show()
        '''
    

    def get_activation(self, ST_input, simulation_time=10000):
        
        N_t = np.zeros((len(self.liquid_layer_neurons),)) #state vector
        activation = [] #activation of LSM over entire simulation time

        for t in range(simulation_time):
            input_current = np.dot(self.input_weight_matrix, ST_input[:,t])
            past_current = np.dot(self.liquid_weight_matrix.T, N_t)
            total_current = input_current + past_current
            temp_activation = []

            for idx, neuron in enumerate(self.liquid_layer_neurons):
                new_Vmem = neuron.update(total_current[idx], t)
                
                if new_Vmem == neuron.V_spike:
                    temp_activation.append(1)
                else:
                    temp_activation.append(0)

            N_t = temp_activation

            activation.append(np.asarray(temp_activation))

        return np.asarray(activation)


