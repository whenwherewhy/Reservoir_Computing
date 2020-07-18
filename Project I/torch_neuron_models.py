import torch

class torch_Dense_LIF(object):
    def __init__(self, num_of_neurons, Vth=0.5, dt=0.001, V_rest=0, tau_ref=4, tau_m = -1, Rm=1, Cm=1):

        #Layer variables - state vectors
        self.num_of_neurons = num_of_neurons
        self.N_t = torch.zeros(num_of_neurons)
        self.V_m = torch.zeros(num_of_neurons)
        self.R_c = torch.zeros(num_of_neurons)  

        #Misc. vectors
        self.zeros = torch.zeros(num_of_neurons)
        
        #simulation parameters
        self.dt = dt                         #(seconds)

        #LIF neuron parameters
        self.V_rest = V_rest                 #resting potential (mV)
        self.tau_ref = tau_ref               #(ms) : refractory period
        self.Vth = Vth                       #(mV)
        self.Rm = Rm                         
        self.Cm = Cm                          
        self.V_spike = Vth+0.5                                  #spike delta (mV)
        self.tau_m = self.Rm * self.Cm if tau_m==-1 else tau_m  #(ms)
    
    def update(self, I):  #This funtion updates the original states of this layer
                          #shape(I) : (num_of_neurons,)
        if not torch.is_tensor(I):
            I = torch.from_numpy(I)       
        assert len(I.size()) == 1

        V_m = self.V_m + ((I*self.Rm + self.V_rest - self.V_m)/self.tau_m)*self.dt
        R_f = (self.R_c.bool()).int()        
        V_m_prime = (1 - R_f)*V_m + R_f*self.V_rest        
        S = (torch.max(self.zeros, V_m_prime - self.Vth)).bool().int()        
        self.N_t = S * self.V_spike
        self.V_m = ((1-S) * V_m_prime) + self.N_t
        R_c_prime = self.R_c - R_f
        self.R_c = (S * self.tau_ref) + R_c_prime

        return self.N_t, self.V_m, self.R_c        

    def update_on_batch(self, I, N_t, V_m, R_c):    #This function updates a set of dummy state vectors and returns them again        
                                                    #shape(I) : (batch_size, num_of_neurons)
        if not torch.is_tensor(I):
            I = torch.from_numpy(I)        
        assert len(I.size()) == 2
        
        zeros = torch.zeros_like(I)

        V_m = V_m + ((I*self.Rm + self.V_rest - V_m)/self.tau_m)*self.dt        
        R_f = (R_c.bool()).int()        
        V_m_prime = (1 - R_f)*V_m + R_f*self.V_rest        
        S = (torch.max(zeros, V_m_prime - self.Vth)).bool().int()        
        N_t = S * self.V_spike
        V_m = ((1-S) * V_m_prime) + N_t
        R_c_prime = R_c - R_f
        R_c = (S * self.tau_ref) + R_c_prime

        return N_t, V_m, R_c        

    def reset(self):
        self.N_t = torch.zeros(self.num_of_neurons)
        self.V_m = torch.zeros(self.num_of_neurons)
        self.R_c = torch.zeros(self.num_of_neurons)  
