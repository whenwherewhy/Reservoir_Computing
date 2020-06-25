from vpython import *
import matplotlib.pyplot as plt
import numpy as np
from environment import Environment

class Visualization:

	def __init__(self, env_objects, env_food, max_objects, scale=0.5):
		#VPython-------------------------------------------------------------------------------
		self.scale = scale
		self.max_objects = max_objects

		self.screen = canvas(x=0, y=0,width=1340,height=750,center=vector(5,0,0), background=vector(0,0,0))
		self.screen.forward = vector(-1,0,0)
		self.screen.up = vector(0,0,1)

		self.axis_x = arrow(pos = vector(0,0,0), axis = vector(self.scale*1000,0,0), shaftwidth = self.scale*5, color = color.red)
		self.axis_y = arrow(pos = vector(0,0,0), axis = vector(0,self.scale*1000,0), shaftwidth = self.scale*5, color = color.green)
		#axis_z = arrow(pos = vector(0,0,0), axis = vector(0,0,self.scale*1000), shaftwidth = 5, color = color.blue)

		self.xy_plane = box(pos = vector(self.scale*500,self.scale*500,0), size = vector(self.scale*1000,self.scale*1000,1), color = vector(0.35,0.35,0.35))
		self.origin = sphere(pos = vector(0,0,0), radius = self.scale*10, color = color.white)

		#Environment--------------------------------------------------------------------------
		self.objects = [sphere(pos = vector(env_objects[_,0],env_objects[_,1],0), radius = env_objects[_,-1], color = color.red) for _ in range(max_objects)]
		self.food = sphere(pos = vector(env_food[0], env_food[1],0), radius = self.scale*10, color = color.green)
		rate(100)

	def step(self, objects, food):
		
		for o in range(self.max_objects):
			self.objects[o].pos = vector(objects[o,0], objects[o,1], 0)

		self.food.pos = vector(food[0], food[1], 0)
