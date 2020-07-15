from LSM_v2 import LSM
from neuron_models import LIF
import matplotlib.pyplot as plt
import numpy as np
from environment import Environment
from spike_encoding import spike_encoding

#Single neuron simulation---------------------------------------------------------------
'''
neuron = LIF(tau_m=10)

I = 200

neuron.initialize()
activation = []

for t in range(88):
    activation.append(neuron.update(I))

neuron.initialize()

for t in range(100):
    activation.append(neuron.update(I))

print(activation)
plt.plot(activation)
plt.show()
'''

#Environment Simulation------------------------------------------------------------------
'''
import pyautogui as gui

num_objects = 10
W,H = 100, 100

env = Environment(W, H, num_objects)

while gui.position() != (0,0):
    
    x,y = gui.position() #Mouse coordinates
    r,f  = np.interp(x,[0,1000],[-0.5, 0.5]), np.interp(y,[0,700],[5, 0])

    state, reward, done = env.step(forward=f, rotate=r)
'''

#LSM V1 simulation----------------------------------------------------------------------------
'''
sim_time = 600
input_dim = 20
w, h = 10, 10

lsm = LSM(input_size = 4, lamda = 30, output_size=2, width=w, height=h, stp_alpha=0.01, stp_beta=0.3)

x1 = np.zeros((int(sim_time/3),))
x1[np.random.randint(0, int(sim_time/3), 50)] = 30
x2 = np.zeros((int(sim_time/3),))
x2[np.random.randint(0, int(sim_time/3), 150)] = 30
x3 = np.zeros((int(sim_time/3),))
x3[np.random.randint(0, int(sim_time/3), 100)] = 30

ST_input = np.hstack((x1,x2,x3))
ST_input = np.vstack((ST_input, ST_input, ST_input, ST_input))

lsm.reset()
activation = []
N_t = []

act = lsm.predict(ST_input)
print(act.shape)
print(act)
'''
'''
for t in range(sim_time):
    activation.append(lsm.predict(ST_input[:,t]))
    N_t.append(lsm.N_t)
activation = np.asarray(activation)
N_t = np.asarray(N_t)
'''
#print(lsm.summary())
#print(N_t)
#plt.imshow(N_t)
#plt.show()

#Poisson Rate coding simulation----------------------------------------------------------------------------


encoder = spike_encoding(scheme='poisson_rate_coding')
a = np.asarray([200, 100, 50, 150, 255])
s = encoder.encode(np.expand_dims(a, axis=-1))

#LSM V2 simulation----------------------------------------------------------------------------

lsm = LSM(5,3,3,3,2)

lsm.reset_states()
activation = lsm.predict(s*50, output='ST_lsm_state')
print(activation.shape)
plt.imshow(activation)
plt.show()


lsm.reset_states()
activation = lsm.predict(s*50)

print(activation)

