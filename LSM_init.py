import numpy as np
import matplotlib.pyplot as plt
from neuron_models import LIF

class LSM:
    def __init__(self, input_size, width, height, output_size):
        self.input_size = input_size
        self.width = width
        self.height = height
        self.output_size = output_size

        self.num_of_neurons = self.width * self.height
        self.liquid_layer_neurons = [LIF() for _ in range(self.num_of_neurons)]
        
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
        self.primary_to_auxiliary_ratio = 0.6 #60% of the liquid layer neurons receive inputs
        self.input_feature_selection_sparsity = 0.6 #40% of input vector will be alotted to a primary neuron
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
        self.lamda = {'EE': 3,'EI': 3,'II': 3,'IE': 3}
        self.synaptic_strength = {'EE': 3,'EI': 3,'II': 1,'IE': 4}
        for n_1 in self.liquid_layer_neurons:
            temp = []
            for n_2 in self.liquid_layer_neurons:
                connection_type = self.get_neuron_type[n_1]+self.get_neuron_type[n_2]
                d = np.sqrt(((self.neuron_to_coordinate[n_1][0]-self.neuron_to_coordinate[n_2][0])**2)+((self.neuron_to_coordinate[n_1][1]-self.neuron_to_coordinate[n_2][1])**2))
                temp.append(d)
                #p_conn = C*np.exp((-d/lamda)**2)
            print(temp)

