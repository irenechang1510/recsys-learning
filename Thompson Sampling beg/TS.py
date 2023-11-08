import numpy as np
import pandas as pd

class ThompsonSampling:
    # sample theta: theta follows beta
    # np.random.beta(10, 7)
    def __init__(self, a, b, T, N_arm):
        self.alpha = a
        self.beta = b
        self.T = T
        self.N_arm = N_arm

    def action_to_outcome(self, x_t):
        '''
        The outcome 
        '''
        o = [0, 0, 0]
        if x_t[0] == 1:
            o[0] = 1 # clicked
        elif x_t[1] == 1:
            u = np.random.uniform()
            if u <= 0.05:
                o[1] = 1 # clicked 70% time
        elif x_t[2] == 1:
            u = np.random.uniform()
            if u <= 0.7:
                o[2] = 1 # clicked 5% time
        
        return o
    
    def sample_theta(self):
        theta_list = []
        for i in range(self.N_arm):
            # sample a value from the beta distributions for each of the actions
            theta_list.append(np.random.beta(self.alpha[i], self.beta[i]))
        return theta_list
    
    def update_param(self, rt, action):
        self.alpha, self.beta = self.alpha + np.array(rt) * np.array(action), self.beta + (1 - np.array(rt)) * np.array(action)
    
    def run_simulation(self):
        for _ in range(self.T):
            # sample model
            theta = self.sample_theta()
        
            # select and apply action
            chosen_action = np.argmax(theta)

            actions = [1 if i == chosen_action else 0 for i in range(self.N_arm)]
            # print(actions)

            # in this case rt = yt
            rt = self.action_to_outcome(actions)

            # update distribution
            self.update_param(rt, actions)


            
