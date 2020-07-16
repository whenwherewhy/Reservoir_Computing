import matplotlib.pyplot as plt
import numpy as np

from LSM_v2 import LSM
from neuron_models import Dense_LIF
from environment import Environment
from spike_encoding import spike_encoding

import time

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
'''
encoder = spike_encoding(scheme='poisson_rate_coding')
a = np.asarray([200, 100, 50, 150, 255])
s = encoder.encode(np.expand_dims(a, axis=-1))
'''
#LSM V2 simulation----------------------------------------------------------------------------
'''
lsm = LSM(5,3,3,3,2)

lsm.reset_states()
activation = lsm.predict(s*50, output='ST_lsm_state')
print(activation.shape)
plt.imshow(activation)
plt.show()


lsm.reset_states()
activation = lsm.predict(s*50)

print(activation)
'''
#LSM v2 time complexity----------------------------------------------------------------------------
'''
signal = np.random.uniform(0,1,200)
signal[np.where(signal>0.2)] = 0

act1 = signal.copy()
act2 = signal.copy()

start_time = time.time()
for i in range(act1.shape[0]):
	if act1[i] > 0:
		act1[i] = 1
print('Using loop: ',time.time()-start_time)

start_time = time.time()
act2[np.where(act2>0)] = 1
print('Using numpy where(): ',time.time()-start_time)
'''
#Dense LIF layer test----------------------------------------------------------------------------
n_neurons = 300
timesteps = 200

liquid_layer_neurons = Dense_LIF(n_neurons)
neurons = [Dense_LIF(1) for _ in range(n_neurons)]

I = np.random.randint(0,2,size=(n_neurons,timesteps))*50

N_t = []
start_time = time.time()
for t in range(timesteps):
    temp = []
    for idx,n in enumerate(neurons):
        temp.append(n.update_(I[idx,t]))
    N_t.append(np.asarray(temp))
print(time.time() - start_time)
N_t = np.asarray(N_t)
plt.plot(N_t.T[0])
plt.show()


N_t = []
start_time = time.time()
for t in range(timesteps):
    N_t.append(liquid_layer_neurons.update(I[:,t])[1])
print(time.time()-start_time)
N_t = np.asarray(N_t)
plt.plot(N_t.T[0])
plt.show()
