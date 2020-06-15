from LSM import LSM
from neuron_models import LIF
import matplotlib.pyplot as plt
import numpy as np

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
'''

sim_time = 200
input_dim = 20
w, h = 20,20 

lsm = LSM(input_size = input_dim, output_size=5, width=w, height=h)

ST_input = np.random.randint(0,2,size=(input_dim,int(sim_time)))*20
#x = np.ones((input_dim, int(sim_time/2)))*20
#y = np.zeros((input_dim, int(sim_time/2)))
#ST_input = np.hstack((x,y))

act, states = lsm.get_activation(ST_input=ST_input, simulation_time=sim_time)

for t,s in enumerate(states):
	try:
		s = s.reshape(w,h)
		plt.imshow(s)
		plt.pause(0.001)
		print(t)
	except:
		break
plt.show()

plt.imshow(act.T)
plt.show()
