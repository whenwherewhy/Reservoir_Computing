import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
from torch_neuron_models import torch_Dense_LIF

class torch_LSM(object):
    def __init__(self, input_size, width, height, depth, output_size, batch_size = 1, num_of_input_duplicates=10, exc_to_inh_ratio=4, C=4, K=4):
                
        self.input_size = input_size
        self.width = width
        self.height = height
        self.depth = depth
        self.output_size = output_size

        #Liquid Layer-------------------------------------------------
        self.num_of_liquid_layer_neurons = width * height * depth
        self.N_t = torch.zeros(self.num_of_liquid_layer_neurons) #state vector

        self.liquid_weight_matrix = torch.zeros(self.num_of_liquid_layer_neurons, self.num_of_liquid_layer_neurons) #(From X To)
        self.liquid_layer_neurons = torch_Dense_LIF(num_of_neurons = self.num_of_liquid_layer_neurons, Vth=0.5, V_rest=0, tau_m=20, Rm = 20, tau_ref=1)

        #Number of excitatory and inhibitory neurons and their ids
        num_of_excitatory_neurons = int((exc_to_inh_ratio/(1+exc_to_inh_ratio))*self.num_of_liquid_layer_neurons)
        num_of_inhibitory_neurons = self.num_of_liquid_layer_neurons - num_of_excitatory_neurons
        self.num_of_excitatory_neurons = num_of_excitatory_neurons

        excitatory_neurons_ids = random.sample(list(range(self.num_of_liquid_layer_neurons)), num_of_excitatory_neurons)
        inhibitory_neurons_ids = list(set(list(range(self.num_of_liquid_layer_neurons))).difference(excitatory_neurons_ids))
        self.excitatory_neurons_ids = excitatory_neurons_ids            
        self.inhibitory_neurons_ids = inhibitory_neurons_ids

        #Create I-->E and E-->I connections
        for col in range(self.liquid_weight_matrix.size()[1]):
            #if the post-synaptic neuron is excitatory
            if col in excitatory_neurons_ids:
                rand_ids = random.sample(inhibitory_neurons_ids, C)
                self.liquid_weight_matrix[rand_ids, col] = 3 # I->E
            #if the post-synaptic neuron is inhibitory
            else:
                rand_ids = random.sample(excitatory_neurons_ids, C)
                self.liquid_weight_matrix[rand_ids, col] = 4 # E->I

        #Create E-->E and I--->I connections (1 & 2)
        for i in range(self.liquid_weight_matrix.size()[0]):
            for j in range(self.liquid_weight_matrix.size()[1]):
                if i!=j: #No self recurrence
                    # E --> E                                      
                    if i in excitatory_neurons_ids and j in excitatory_neurons_ids:
                        I_k_1 = [x for x in torch.where(self.liquid_weight_matrix[i,:]==4)[0]] #indices of inhibitory neurons that E_i connects to
                        I_k_2 = [x for x in torch.where(self.liquid_weight_matrix[:,j]==3)[0]] #indices of inhibitory neurons that connects to E_j
                        intersection = list(set(I_k_1).intersection(I_k_2))
                        if len(intersection) > 0:
                            self.liquid_weight_matrix[i][j] = 1

                    # I --> I
                    elif i in inhibitory_neurons_ids and j in inhibitory_neurons_ids:
                        E_k_1 = [x for x in torch.where(self.liquid_weight_matrix[i,:]==3)[0]] #indices of excitatory neurons that I_i connects to
                        E_k_2 = [x for x in torch.where(self.liquid_weight_matrix[:,j]==4)[0]] #indices of excitatory neurons that connects to I_j
                        intersection = list(set(E_k_1).intersection(E_k_2))
                        if len(intersection) > 0:
                            self.liquid_weight_matrix[i][j] = 2
        
        #Allocate weights to the particular connection types
        for i in range(self.liquid_weight_matrix.size()[0]):
            for j in range(self.liquid_weight_matrix.size()[1]):
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
        self.pseudo_input_weight_matrix = torch.zeros(intermediate_input_size, self.input_size) #Creates 10 copies of each input dimension

        for i in range(self.input_size):
            self.pseudo_input_weight_matrix[i*num_of_input_duplicates:(i+1)*num_of_input_duplicates,i] = torch.ones(num_of_input_duplicates)

        self.input_weight_matrix = torch.zeros(self.num_of_liquid_layer_neurons, intermediate_input_size)

        for neuron_id in range(self.num_of_liquid_layer_neurons):
            if neuron_id in excitatory_neurons_ids: 
                rand_ids = random.sample(list(range(intermediate_input_size)), K)
                for r in rand_ids:
                    self.input_weight_matrix[neuron_id, r] = random.uniform(0, 0.6)
        
        #Read-Out Layer-----------------------------------------------
        self.readout_network = get_readout_network(input_size = num_of_excitatory_neurons, hidden_size = num_of_excitatory_neurons, output_size = output_size)
        self.readout_optimizer = optim.RMSprop(self.readout_network.parameters(), lr=0.0002)
        self.criterion = nn.MSELoss()

    def reset_state(self):
        self.liquid_layer_neurons.reset()
        self.N_t = torch.zeros(self.num_of_liquid_layer_neurons) #state vector
    
    def predict(self, input_state, output='q_values'): #Input shape : (num_of_neurons, timesteps)
        activation = [] #activation of LSM over entire input spike train duration
        if not torch.is_tensor(input_state):
            input_state = torch.from_numpy(input_state)
        input_state = input_state.float()            
        for t in range(input_state.size()[-1]):          
            pseudo_input_current = torch.matmul(self.pseudo_input_weight_matrix, input_state[:,t])
            input_current = torch.matmul(self.input_weight_matrix, pseudo_input_current)
            past_current = torch.matmul(self.liquid_weight_matrix.T, self.N_t) 
            total_current = (input_current + past_current)
            self.N_t, _, _ = self.liquid_layer_neurons.update(total_current)            
            activation.append(self.N_t.unsqueeze(-1))
        activation = torch.cat(activation, dim=-1)   #Shape : N x T

        #Calculate average firing rate of each neuron during the entire input duration
        average_firing_rate = torch.sum(activation[self.excitatory_neurons_ids], dim=-1) / input_state.shape[-1]

        #Output
        if output == 'q_values':
            #Feed forward the obtained activations into Q_network
            q = self.readout_network(average_firing_rate.unsqueeze(0))
            return q

        elif output == 'average_firing_rate':
            #Return average firing rate of each liquid layer neuron
            return average_firing_rate

        elif output == 'average_firing_rate_and_q_values':
            #return both average firing rate and q_values
            return [average_firing_rate, self.readout_network(average_firing_rate.unsqueeze(0))]

        elif output == 'ST_lsm_state':
            #Return the spatiotemporal state of the lsm over input duration
            return activation        

    def predict_on_batch(self, input_state, output='q_values', network='q_network'):    #Input shape : (batch_size, num_of_neurons, timesteps)
        if not torch.is_tensor(input_state):
            input_state = torch.from_numpy(input_state)
        input_state = input_state.float()
        batch_size = input_state.size()[0]
        N_t = torch.zeros(self.num_of_liquid_layer_neurons, batch_size) #state vector
        LIF_V_m = torch.zeros(self.num_of_liquid_layer_neurons, batch_size)
        LIF_R_c = torch.zeros(self.num_of_liquid_layer_neurons, batch_size)          
        
        activation = [] #activation of LSM over entire input spike train duration

        for t in range(input_state.size()[-1]):          
            pseudo_input_current = torch.matmul(self.pseudo_input_weight_matrix, input_state[:,:,t].T) #input_state[:,:,t].T shape: (N, B, :)
            input_current = torch.matmul(self.input_weight_matrix, pseudo_input_current)
            past_current = torch.matmul(self.liquid_weight_matrix.T, N_t) 
            total_current = (input_current + past_current)                      #Shape : N X B
            N_t, LIF_V_m, LIF_R_c = self.liquid_layer_neurons.update_on_batch(total_current, N_t=N_t, V_m=LIF_V_m, R_c=LIF_R_c)            
            activation.append(N_t.T.unsqueeze(-1))  #Shape: (B, N, 1)
        activation = torch.cat(activation, dim=-1)   #Shape : (B, N, T)
        
        #Calculate average firing rate of each neuron during the entire input duration
        average_firing_rate = torch.sum(activation[:,self.excitatory_neurons_ids,:], dim=-1) / input_state.size()[-1]

        #Output
        if output == 'q_values':
            #Feed forward the obtained activations into Q_network
            q = self.readout_network(average_firing_rate)
            return q

        elif output == 'average_firing_rate':
            #Return average firing rate of each liquid layer neuron
            return average_firing_rate

        elif output == 'average_firing_rate_and_q_values':
            #return both average firing rate and q_values 
            q = self.readout_network(average_firing_rate)
            return [average_firing_rate, q]

    def train_readout_network(self, inputs, targets):
        
        self.readout_optimizer.zero_grad()
        predictions = self.readout_network(inputs)
        loss = self.criterion(predictions, targets)
        loss.backward()
        self.readout_optimizer.step()

    
class get_readout_network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(get_readout_network, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input_x):
        x = F.relu(self.hidden_layer(input_x))
        output = self.output_layer(x)
        return output
