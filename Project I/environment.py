import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

class Environment:
    def __init__(self, W, H, max_objects=-1):

        self.window_w, self.window_h = W, H
        self.max_objects = max_objects if max_objects!=-1 else W
        
        self.objects = np.zeros((self.max_objects,5)) #(index of object, [x,y,vx,vy,r])
        self.food = np.zeros((3,))  # (x,y,r) #Considering single food at any instant
        self.agent = np.zeros((3,)) # (x,y,theta)
        self.agent_health = 50
        self.agent_object_distances = []

        self.objects[:,0:2] = np.random.randint(size=(self.max_objects,2), low=-W, high=W) #X,y
        self.objects[:,2:4] = np.random.randint(size=(self.max_objects,2), low=-2, high=3)  #Vx, Vy
        self.objects[:,-1:] = np.random.randint(size=(self.max_objects,1), low=5, high=25)  #r 
        self.food[:] = [np.random.randint(-W, W, 1), np.random.randint(-H, H, 1), 50]       #x,y,r
        self.agent[:] = np.asarray([np.random.randint(-W, W, 1), np.random.randint(-H, H, 1), np.pi/4]) #x,y,theta

    def vector_distance(self, x,y):
        
        return np.linalg.norm(x-y)

    def update_objects(self):
        #check for wall collisions and update objects' velocity vectors
        self.objects[:,2] = np.where(np.absolute(self.objects[:,0]+self.objects[:,2]) >= self.window_w, self.objects[:,2]*-1, self.objects[:,2])            
        self.objects[:,3] = np.where(np.absolute(self.objects[:,1]+self.objects[:,3]) >= self.window_h, self.objects[:,3]*-1, self.objects[:,3])            
        self.objects[:,0:2] += self.objects[:,2:4]

    def update_agent(self, forward=0, rotate=0):
        #update agent position according to input
        if self.agent[2] + rotate > 2*np.pi:
            self.agent[2] = (self.agent[2]+rotate) - 2*np.pi
        elif self.agent[2] + rotate < 0:
            self.agent[2] = 2*np.pi - (self.agent[2]+rotate)
        else:
            self.agent[2] += rotate 
        self.agent[0] += forward * np.cos(self.agent[2]) #new_x = old_x + Fcos(new_theta)
        self.agent[1] += forward * np.sin(self.agent[2]) #new_y = old_y + Fsin(new_theta)

        #Constrain agent within simulation window
        self.agent[0] = self.agent[0] if self.agent[0] <= self.window_w else self.window_w
        self.agent[0] = self.agent[0] if self.agent[0] >= -self.window_w else -self.window_w
        self.agent[1] = self.agent[1] if self.agent[1] <= self.window_h else self.window_h
        self.agent[1] = self.agent[1] if self.agent[1] >= -self.window_h else -self.window_h

    def update_agent_object_distances(self):

        self.agent_object_distances = [np.sqrt((self.agent[0]-obj[0])**2 + (self.agent[1]-obj[1])**2) for obj in self.objects]

    def agent_dies(self):
        
        #Agent dies if collides with an object
        die = False
        if np.min(self.agent_object_distances) <= self.objects[np.argmin(self.agent_object_distances), -1]/2:
            die = True   

        return die

    def agent_eats_food(self):
        if np.sqrt((self.agent[0]-self.food[0])**2 + (self.agent[1]-self.food[1])**2) < 10:
            return True
        else:
            return False

    def allocate_new_food(self):

        self.food[:] = [np.random.randint(-self.window_w, self.window_w, 1), np.random.randint(-self.window_h, self.window_h, 1), 50]
    
    def get_nearest_object_data(self):

        plt.plot([self.objects[np.argmin(self.agent_object_distances), 0], self.agent[0]], [self.objects[np.argmin(self.agent_object_distances), 1], self.agent[1]], c='r')
        
        #Dist(agent, nearest_object)
        nearest_object_dist = self.vector_distance(self.objects[np.argmin(self.agent_object_distances), :2], self.agent[:2])

        #shift origin to agent's position
        n_obj_x, n_obj_y =  self.objects[np.argmin(self.agent_object_distances), 0] - self.agent[0], self.objects[np.argmin(self.agent_object_distances), 1] - self.agent[1]
        #rotate coordinate axes wrt agent's heading direction
        n_obj_x, n_obj_y = (n_obj_x*np.cos(self.agent[-1])+n_obj_y*np.sin(self.agent[-1])), (-n_obj_x*np.sin(self.agent[-1])+n_obj_y*np.cos(self.agent[-1]))
        
        if n_obj_x >0 and n_obj_y >=0: #1st quad
            relative_object_theta = np.arctan(n_obj_y / n_obj_x)
        elif n_obj_x <0 and n_obj_y >=0: #2nd quad
            relative_object_theta = (np.pi) - np.arctan(n_obj_y / np.absolute(n_obj_x)) 
        elif n_obj_x <0 and n_obj_y <0: #3rd quad
            relative_object_theta = (np.pi) + np.arctan(n_obj_y / n_obj_x)
        elif n_obj_x >0 and n_obj_y <0: #4th quad
            relative_object_theta = (2*np.pi) - np.arctan(np.absolute(n_obj_y) / n_obj_x)
        elif n_obj_x == 0 and n_obj_y != 0:
            relative_object_theta = np.pi/2 if n_obj_y>0 else -np.pi/2
        elif n_obj_x == 0 and n_obj_y == 0:
            relative_object_theta = 0

        if relative_object_theta > np.pi:       #Constrain angle in [-pi, pi]
            relative_object_theta -= (2*np.pi) 

        return nearest_object_dist, np.sin(relative_object_theta), np.cos(relative_object_theta)

    def get_food_data(self):

        plt.plot([self.food[0], self.agent[0]], [self.food[1], self.agent[1]], c='g')
        
        #Dist(agent, food)
        food_dist = self.vector_distance(self.food[:2], self.agent[:2])

        #shift origin to agent's position
        food_x, food_y =  self.food[0] - self.agent[0], self.food[1] - self.agent[1]
        #rotate coordinate axes wrt agent's heading direction
        food_x, food_y = (food_x*np.cos(self.agent[-1])+food_y*np.sin(self.agent[-1])), (-food_x*np.sin(self.agent[-1])+food_y*np.cos(self.agent[-1]))
        
        if food_x >0 and food_y >=0: #1st quad
            food_theta = np.arctan(food_y / food_x)
        elif food_x <0 and food_y >=0: #2nd quad
            food_theta = (np.pi) - np.arctan(food_y / np.absolute(food_x)) 
        elif food_x <0 and food_y <0: #3rd quad
            food_theta = (np.pi) + np.arctan(food_y / food_x)
        elif food_x >0 and food_y <0: #4th quad
            food_theta = (2*np.pi) - np.arctan(np.absolute(food_y) / food_x)
        elif food_x == 0 and food_y != 0:   #divided by zero exception
            food_theta = np.pi/2 if food_y>0 else -np.pi/2
        elif food_x == 0 and food_y == 0:   #divided by zero exception
            food_theta = 0

        if food_theta > np.pi:       #Constrain angle in [-pi, pi]
            food_theta -= (2*np.pi) 

        return food_dist, np.sin(food_theta), np.cos(food_theta)

    def get_agent_direction(self):

        return np.sin(self.agent[-1]), np.cos(self.agent[-1])

    def visualize_simulation(self):
        
        plt.xlim(-self.window_w, self.window_w)
        plt.ylim(-self.window_h, self.window_h)
        plt.scatter(self.objects[:,0], self.objects[:,1], s = self.objects[:,-1]**2, alpha=0.5 , c = 'b')
        plt.scatter(self.food[0], self.food[1], s = self.food[2], c = 'g')

        m = MarkerStyle('^')
        m._transform.rotate_deg(np.degrees(self.agent[2])-90)
        plt.scatter(self.agent[0], self.agent[1], marker=m, c = 'r', s = 100)

        plt.pause(0.0001)
        plt.clf()

    def step(self, forward=0, rotate=0): 
        self.visualize_simulation()

        #Update states of : Objects, Agent, agent_object distances
        self.update_objects()
        self.update_agent(forward=forward, rotate=rotate)
        self.update_agent_object_distances()

        #check conditions:
        #--did agent die?
        #--did agent eat food ?

        if self.agent_dies():
            self.agent_health = 0
            #optional
            self.agent[:] = np.asarray([np.random.randint(-self.window_w, self.window_w, 1), np.random.randint(-self.window_h, self.window_h, 1), np.pi/4])

        if self.agent_eats_food():
            self.agent_health = 50
            self.allocate_new_food()
        else:
            self.agent_health -= 1

        #Feedbacks
        #Nearest Object info
        R_o, sin_o, cos_o = self.get_nearest_object_data()
        #Food info
        R_f, sin_f, cos_f = self.get_food_data()
        #Agent's heading direction info
        sin_h, cos_h = self.get_agent_direction()
        #Agent's health 
        health = self.agent_health

        return R_o, sin_o, cos_o, R_f, sin_f, cos_f, sin_h, cos_h, health
    

