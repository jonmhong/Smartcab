import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.q_table = {}
        
        self.possible_actions = {}
        for move in self.env.valid_actions:
            self.possible_actions[move] = 12
            
        print "initialize: self.possible_actions: {}".format(self.possible_actions)
        
        self.previous_action = None
        self.previous_state = None
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.destination = self.planner.destination
        # objective: choose what variables are necessary in the state
        
        
    def update(self, t):   
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
                
        # TODO: Update state
        self.location = self.env.agent_states[self]['location']
        self.heading = self.env.agent_states[self]['heading']
        self.light = self.env.sense(self)['light']
        state = (self.location, self.next_waypoint, self.light) # add cars later

        # adjust these parameters
        gamma = 0.9 # discounting rate
        epsilon = 0.1 # chance of choosing a random action
        
        if t == 0:
            alpha = 1
        else:
            alpha = 1.0 / t        
        
        #BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG
        # there's a bug with updating nested dictionary values
        
        if (state is not None) and (state not in self.q_table):
            # set the q-values to zero
            self.q_table[state] = {'forward': 0, 'left': 0, 'right': 0, None: 0}

        # TODO: Select action according to your policy
        action = None
        
        if random.random() > epsilon and (0 not in self.q_table[state].values()):
            action = max(self.q_table[state])
        else:
            action = random.choice(self.possible_actions.keys())
        
        reward = self.env.act(self, action)
        
        # TODO: Learn policy based on state, action, reward
        q_val_f = self.q_table[state]['forward']
        q_val_l = self.q_table[state]['left']
        q_val_r = self.q_table[state]['right']
        q_val_n = self.q_table[state][None]
        q_hat = None
        
        max_q = max([q_val_f, q_val_l, q_val_r, q_val_n])
        
        if self.q_table[state][action] == 0:
            # if the state, action has never been taken, reward == q-value
            self.q_table[state][action] = reward
        else:
            q_hat = ((1 - alpha) * self.q_table[self.previous_state][action]) + (alpha * (reward + gamma * max_q))
            self.q_table[self.previous_state][action] = q_hat
                
        self.previous_state = state
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        print self.q_table
        print self.possible_actions
        print action
        print state


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.1)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
