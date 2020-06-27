#from LSM import LSM
from neuron_models import LIF
import matplotlib.pyplot as plt
import numpy as np
from environment import Environment

#Single neuron simulation---------------------------------------------------------------
'''
neuron = LIF(tau_m=10)

I = 2

neuron.initialize()
activation = []

for t in range(10000):	#ms
	s = np.random.randint(0,2,1)*10	#{0,10} mV
	print(s)
	Vmem = neuron.update(s, t)
	activation.append(Vmem)

plt.plot(activation)
plt.plot([x/100 for x in spikes])
plt.show()


#LSM simulation----------------------------------------------------------------------------
sim_time = 500
input_dim = 20
w, h = 3,3

lsm = LSM(input_size = input_dim, lamda = 20, output_size=5, width=w, height=h)

#x = np.ones((input_dim, int(sim_time/2)))*20
#y = np.zeros((input_dim, int(sim_time/2)))

x = np.random.randint(0,2,size=(input_dim,int(sim_time/2)))*20
y = np.random.randint(0,2,size=(input_dim,int(sim_time/2)))*40
ST_input = np.hstack((x,y))

act = lsm.get_activation(ST_input=ST_input, simulation_time=sim_time)

for t,s in enumerate(states):
	try:
		s = s.reshape(w,h)
		plt.imshow(s)
		plt.pause(0.001)
		print(t)
	except:
		break
plt.show()
'''
'''
print(act.shape)
plt.imshow(act.T)
plt.show()
'''

#Environment Simulation------------------------------------------------------------------
num_objects = 10
W = 50

env = Environment(W, W, num_objects)

while True:

	env.step(forward=np.random.randint(0,3,1), rotate=np.random.uniform(-3.14,3.14))