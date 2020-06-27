import numpy as np
import matplotlib.pyplot as plt

class Environment:
	def __init__(self, W, H, max_objects=-1):
		#self.env_state = np.zeros((W,H,3))
		
		self.max_objects = max_objects if max_objects!=-1 else W
		self.objects = np.zeros((self.max_objects,5)) #(index of object, [x,y,vx,vy,r])
		self.food = np.zeros((4,)) 	# (x,y,vx,vy) #Considering single food at any instant
		

	def step(self):

		self.objects[:,0:2] += self.objects[:,2:4]
		self.food[0:2] += self.food[2:4]
		self.visualize_simulation()

		print(self.objects)

	def visualize_simulation(self):
		
		plt.scatter(self.objects[:,0], self.objects[:,1], s = self.objects[:,-1], alpha=0.5 , c = 'b')
		plt.scatter(self.food[0], self.food[1], s = np.mean(self.objects[:,-1]), c = 'g', marker='^')
		plt.pause(0.001)
		plt.clf()


	

