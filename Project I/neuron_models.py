import numpy as np

class LIF(object):
    def __init__(self, Vth=0.5, dt=0.001, Vres=0, tau_ref=4, tau_m = -1, Rm=1, Cm=10):

        #simulation parameters
        self.dt = dt                         #(seconds)

        #LIF parameters
        self.Vres = Vres                     #resting potential (mV)
        self.Vm = self.Vres                  #initial potential (mV)
        self.refraction_counter = -1         #refraction counter
        self.tau_ref = tau_ref               #(ms) : refractory period
        self.Vth = Vth                       #(mV)
        self.ref = False                     #Refraction flag

        self.Rm = Rm                         
        self.Cm = Cm                          
        if tau_m!=-1:
            self.tau_m = tau_m               #(ms)
        else:
            self.tau_m = self.Rm * self.Cm   #(ms)

        self.V_spike = Vth+0.5         #spike delta (mV)
            
    def update_(self, I, time_stamp):
        if time_stamp > self.t_rest:
            self.Vm = self.Vm + (((I*self.Rm - self.Vm) / self.tau_m) * self.dt)
            
            if self.Vm >= self.Vth:
                self.Vm = self.V_spike
                self.t_rest = time_stamp + self.tau_ref
        else:
            self.Vm = self.Vres
        return self.Vm

    def update(self, I):
        if not self.ref:
            self.Vm = self.Vm + (((I*self.Rm + self.Vres - self.Vm) / self.tau_m) * self.dt)
            if self.Vm >= self.Vth:
                self.Vm = self.V_spike
                self.ref = True
                self.refraction_counter = self.tau_ref
        else:
            self.Vm = self.Vres
            self.refraction_counter -= 1
            if self.refraction_counter <= 0:
                self.ref = False

        return self.Vm
            
    def reset(self):
        self.Vm = self.Vres
        self.ref = False
        self.refraction_counter = -1         

class Dense_LIF(object):
    def __init__(self, num_of_neurons, Vth=0.5, dt=0.001, V_rest=0, tau_ref=4, tau_m = -1, Rm=1, Cm=1):

        #Layer variables - state vectors
        self.num_of_neurons = num_of_neurons
        self.N_t = np.zeros((num_of_neurons,))
        self.V_m = np.zeros((num_of_neurons,))
        self.R_c = np.zeros((num_of_neurons,))

        #Misc. vectors
        self.zeros = np.zeros((num_of_neurons,))
        self.ref = False
        self.refraction_counter = -1 
        self.Vm = V_rest                  #initial potential (mV)
        
        #simulation parameters
        self.dt = dt                         #(seconds)

        #LIF neuron parameters
        self.V_rest = V_rest                     #resting potential (mV)
        self.tau_ref = tau_ref               #(ms) : refractory period
        self.Vth = Vth                       #(mV)
        self.Rm = Rm                         
        self.Cm = Cm                          
        self.V_spike = Vth+0.5                                  #spike delta (mV)
        self.tau_m = self.Rm * self.Cm if tau_m==-1 else tau_m  #(ms)

    def update(self, I):
        assert I.shape[0] == self.num_of_neurons

        V_m = self.V_m + ((I*self.Rm + self.V_rest - self.V_m)/self.tau_m)*self.dt
        
        R_f = (self.R_c.astype(bool)).astype(int)
        
        V_m_prime = (1 - R_f)*V_m + R_f*self.V_rest
        
        S = ((np.maximum(self.zeros, V_m_prime - self.Vth)).astype(bool)).astype(int)
        
        self.N_t = S * self.V_spike

        self.V_m = np.multiply((1-S),V_m_prime) + self.N_t

        R_c_prime = self.R_c - R_f

        self.R_c = (S * self.tau_ref) + R_c_prime

        return self.N_t, self.V_m, self.R_c

    def reset(self):
        self.N_t = np.zeros((self.num_of_neurons,))
        self.V_m = np.zeros((self.num_of_neurons,))
        self.R_c = np.zeros((self.num_of_neurons,))   

    def update_(self, I):
        if not self.ref:
            self.Vm = self.Vm + (((I*self.Rm + self.V_rest - self.Vm) / self.tau_m) * self.dt)
            if self.Vm >= self.Vth:
                self.Vm = self.V_spike
                self.ref = True
                self.refraction_counter = self.tau_ref
        else:
            self.Vm = self.V_rest
            self.refraction_counter -= 1
            if self.refraction_counter <= 0:
                self.ref = False
        return self.Vm
            
    def reset_(self):
        self.Vm = self.V_rest
        self.ref = False
        self.refraction_counter = -1              