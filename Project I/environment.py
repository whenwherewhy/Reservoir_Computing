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
        self.food[:] = [np.random.randint(-W, W, 1), np.random.randint(-H, H, 1), 50]
        self.agent[:] = np.asarray([np.random.randint(-W, W, 1), np.random.randint(-H, H, 1),0])

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

    def update_agent_object_distances(self):

        self.agent_object_distances = [np.sqrt((self.agent[0]-obj[0])**2 + (self.agent[1]-obj[1])**2) for obj in self.objects]

    def agent_dies(self):
        
        #Agent dies if --
        #--if agent goes outside simulation window
        #--if agent collides with an object

        die = False
        if np.absolute(self.agent[0]) >= self.window_w or np.absolute(self.agent[1]) >= self.window_h:
            die = True
            #print('out')
        elif np.min(self.agent_object_distances) <= self.objects[np.argmin(self.agent_object_distances), -1]/4:
            die = True   
            #print('collide', np.min(self.agent_object_distances), self.objects[np.argmin(self.agent_object_distances), -1]/4)

        return die

    def agent_eats_food(self):
        if np.sqrt((self.agent[0]-self.food[0])**2 + (self.agent[1]-self.food[1])**2) < 10:
            return True
        else:
            return False

    def allocate_new_food(self):

        self.food[:] = [np.random.randint(-self.window_w, self.window_w, 1), np.random.randint(-self.window_h, self.window_h, 1), 50]
    
    def visualize_simulation(self):
        
        plt.xlim(-self.window_w, self.window_w)
        plt.ylim(-self.window_h, self.window_h)
        plt.scatter(self.objects[:,0], self.objects[:,1], s = self.objects[:,-1]**2, alpha=0.5 , c = 'b')
        plt.scatter(self.food[0], self.food[1], s = self.food[2], c = 'g')

        m = MarkerStyle('^')
        m._transform.rotate_deg(np.degrees(self.agent[2]))
        plt.scatter(self.agent[0], self.agent[1], marker=m, c = 'r', s = 200)

        plt.pause(0.0001)
        plt.clf()

    def step(self, forward=0, rotate=0): 
        self.visualize_simulation()

        self.update_objects()
        self.update_agent(forward=forward, rotate=rotate)
        self.update_agent_object_distances()

        if self.agent_dies():
            self.agent_health = 0
            #optional
            self.agent[:] = np.asarray([np.random.randint(-self.window_w, self.window_w, 1), np.random.randint(-self.window_h, self.window_h, 1),0])

        if self.agent_eats_food():
            self.agent_health = 50
            self.allocate_new_food()
        else:
            self.agent_health -= 1

        #Prepare return info

        print(self.agent_health)


    

