class LIF:
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
            self.Vm = self.Vm + (((I*self.Rm - self.Vm) / self.tau_m) * self.dt)
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
            
    def initialize(self):
        self.Vm = self.Vres
        self.ref = False
        self.refraction_counter = -1         
