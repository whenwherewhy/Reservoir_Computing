from LSM import LSM
from neuron_models import LIF
import matplotlib.pyplot as plt
import numpy as np

#lsm = LSM(input_size = 5, output_size=5, width=4, height=4)
neuron = LIF(dt=0.001, tau_m=10)

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
