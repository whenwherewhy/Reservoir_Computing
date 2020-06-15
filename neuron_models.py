class LIF:
    def __init__(self, threshold=0.5, dt=0.001, Vres=0, tau_ref=4, tau_m = -1, Rm=1, Cm=10):

        #simulation parameters
        self.dt = dt                         #(seconds)

        #LIF parameters
        self.Vres = Vres                     #resting potential (mV)
        self.Vm = self.Vres                  #initial potential (mV)
        self.t_rest = -1                     #initial resting time point
        self.tau_ref = tau_ref               #(ms) : refractory period
        self.Vth = threshold                 #(mV)

        self.Rm = Rm                         
        self.Cm = Cm                          
        if tau_m!=-1:
            self.tau_m = tau_m               #(ms)
        else:
            self.tau_m = self.Rm * self.Cm   #(ms)

        self.V_spike = threshold+0.5         #spike delta (mV)
            
    def update(self, I, time_stamp):
        if time_stamp > self.t_rest:
            self.Vm = self.Vm + (((I*self.Rm - self.Vm) / self.tau_m) * self.dt)
            
            if self.Vm >= self.Vth:
                self.Vm = self.V_spike
                self.t_rest = time_stamp + self.tau_ref
        else:
            self.Vm = self.Vres
        return self.Vm
    
    def initialize(self):
        self.Vm = self.Vres
        self.t_rest = -1
