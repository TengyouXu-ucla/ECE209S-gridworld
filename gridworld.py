from dataclasses import dataclass
import sys
import numpy as np
import matplotlib.pyplot as plt

label_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Direction vectors for actions
STAY = np.array((0,0))
RIGHT = np.array((1,0))
LEFT = np.array((-1,0))
UP = np.array((0,1))
DOWN = np.array((0,-1))
actions = [STAY,RIGHT,LEFT,UP,DOWN]

@dataclass
class state:
    label:str
    coord:np.ndarray

@dataclass
class destination(state):
    destination_type:str = 'ice cream shop'

@dataclass
class obstacle(state):
    pass
        
@dataclass
class hazard(state):
    pass

class agent:
    def __init__(self,s:state,history:list[state]=[]) -> None:
        self.s = s
        self.history = history

    def update_state(self,s_new):
        self.history.append(self.s)
        self.s = s_new


class gridworld:
    def __init__(self,size:tuple[int],agent_start:np.ndarray,destinations:list[tuple[int]],obstacles:list[tuple[int]],hazards:list[tuple[int]],p_e = 0.2,destination_type='ice cream shop') -> None:
        '''
        `size` = size of the grid-world in X,Y\\
        `agent_start` = (x,y) coordinate for the starting position of the agent\\
        `destinations` = list of (x,y) coordinates for destinations of type `destination_type`\\
        `obstacles` = list of (x,y) coordinates for obstacles\\
        `hazards` = list of (x,y) coordinates for hazards\\
        `destination_type` = str or int label for type of destination, ex: "ice cream shop"
        '''
        if len(size) == 2:
            self.x_max = size[0]
            self.y_max = size[1]
            self.num_states = size[0]*size[1]
        else:
            raise Exception("size should be integer valued tuple lenth 2")
        # Initialize/calc grid states
        self.make_grid(destinations,obstacles,hazards,destination_type)
        self.p_e = p_e # prob of error
        # transition probabilities
        self.calc_p_matrix()

        if len(agent_start) == 2:
            # Check start coord to be in bounds
            s_start = self.coord2state(agent_start)
            if s_start and s_start is not obstacle:
                self.agent = agent(s_start)
            else:
                raise Exception("could not find valid state from agent_start")
        else:
            raise Exception("agent_start should be integer valued tuple of length 2")

    def make_grid(self,destinations,obstacles,hazards,destination_type):
        self.states = []
        self.state_labels = []
        self.state_coords = []
        for i,label in enumerate(label_str[0:self.num_states]):
            x = i%self.x_max
            y = i//self.x_max
            self.state_labels.append(label)
            self.state_coords.append(np.array((x,y)))
            if (x,y) in destinations:
                self.states.append(destination(label,np.array((x,y)),destination_type))
            elif (x,y) in obstacles:
                self.states.append(obstacle(label,np.array((x,y))))
            elif (x,y) in hazards:
                self.states.append(hazard(label,np.array((x,y))))
            else:
                self.states.append(state(label,np.array((x,y))))

    def coord2state(self,coord):
        for state in self.states:
            if np.all(coord==state.coord):
                return state
        return
        
    def label2state(self,label):
        for state in self.states:
            if label == state.label:
                return state
        return
    
    # def update_action_set(self):
    #     self.action_set = []
    #     for action in actions:
    #         s_new = self.coord2state((self.state.x+action[0],self.state.y+action[1]))
    #         if s_new is obstacle:
    #             continue
    #         if s_new in self.states:
    #             self.action.append(action)
    
    def calc_p_matrix(self):
        self.p_matrix = np.zeros((len(actions),self.num_states,self.num_states))
        for i,a_t in enumerate(actions):
            for j,s_t in enumerate(self.states):
                s_calc = self.coord2state(s_t.coord + a_t)
                if not s_calc or (s_calc is obstacle):
                    self.p_matrix[i,j,j] += 1 - self.p_e - self.p_e/(len(actions)-1)
                    self.p_matrix[:,j,j] += self.p_e/(len(actions)-1)
                for k,s_tp1 in enumerate(self.states):
                    # limit motion to one action away
                    if np.any(np.all((s_tp1.coord - s_t.coord)==actions,axis=1)):
                        # intended direction
                        if s_calc == s_tp1:
                            self.p_matrix[i,j,k] += 1 - self.p_e
                        else:
                            self.p_matrix[i,j,k] += self.p_e/(len(actions)-1)

                    # states that are more than 1 action away
                    else:
                        self.p_matrix[i,j,k] += 0 # redundant
                        
def plot_pmatrix(p_matrix, title): 
    grid_matrix = np.zeros((5,5))
    
    for i in range(len(p_matrix)):
        grid_matrix[i//5,i%5] = p_matrix[i]

    # Flip vertically to adjust for the origin being in the bottom-left corner
    grid_matrix = np.flipud(grid_matrix)

    label_str = "ABCDEFGHIJKLMNOPQRSTUVWXY"
    state_labels = np.array(list(label_str)).reshape((5, 5))
    state_labels = np.flipud(state_labels)  # Flip to match the grid orientation

    # Plot the heatmap
    plt.imshow(grid_matrix, cmap='viridis', interpolation='nearest')
    plt.yticks(np.arange(4,-1,-1))

    # Add annotations for each cell with both the state label and probability value
    for i in range(grid_matrix.shape[0]):
        for j in range(grid_matrix.shape[1]):
            # Display state label and probability in each cell
            plt.text(j, i, f"{state_labels[i, j]}\n{grid_matrix[i, j]:.2f}",
                    ha='center', va='center', color='white' if grid_matrix[i, j] < 0.5 else 'black')

    # Add a colorbar and title

    plt.xticks(range(5))  # Show x-axis ticks
    #plt.yticks([4, 3, 2, 1, 0])  # Show y-axis ticks
    plt.title(title)
    plt.yticks(ticks=np.arange(grid_matrix.shape[0]), labels=np.flipud(np.arange(grid_matrix.shape[0])))

    plt.show()         

