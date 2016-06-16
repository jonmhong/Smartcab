import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

# these are used to analyze the agent's results 
import matplotlib.pyplot as pl
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        # setting up the q_table
        self.q_table = {}
        self.initial_q = 12        
        self.possible_actions = {}
        for move in self.env.valid_actions:
            self.possible_actions[move] = self.initial_q

        # these are variables to remember the previous state and action (for q-learning)
        self.previous_action = None
        self.previous_state = None

        # these are debugging variables
        self.successful_runs = []
        self.success_tally = 0
        self.run_trial = 1


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.destination = self.planner.destination
        self.previous_action = None
        self.previous_state = None
        self.run_trial += 1
        
        
    def update(self, t):   
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        # objective: choose what variables are necessary in the state
        self.location = self.env.agent_states[self]['location'] # hypothesis: this doesn't matter
        self.heading = self.env.agent_states[self]['heading']
        self.light = self.env.sense(self)['light']
        
        """
        1 self.next_waypoint is important for the smartcar to know what direction to head towards
        2 The smartcar should know the state of the light when it is rewarded
        """
        state = (self.next_waypoint, self.light)

        gamma = 0.7 ** self.run_trial # discount rate

        # decay the learning rate and epsilon
        if self.run_trial <= 1:
            alpha = 1
            epsilon = 1
        else:
            alpha = 1.0 / self.run_trial
            epsilon = 1.0 / self.run_trial
        
        # create new state in q_table if explored
        if (state is not None) and (state not in self.q_table):
            # set the q-values to zero
            self.q_table[state] = {'forward': self.initial_q, 'left': self.initial_q, 
                                   'right': self.initial_q, None: self.initial_q}

        # TODO: Select action according to your policy
        action = None
        
        # choose a random action epsilon percent of the time, else choose highest q-value action
        if random.random() > epsilon:
            for a, q in self.q_table[state].iteritems():
                if q == max(self.q_table[state].values()):
                    action = a
        else:
            action = random.choice(self.possible_actions.keys())

        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        q_hat = None
        max_q = max(self.q_table[state].values())
        
        left = self.env.sense(self)['left']
        right = self.env.sense(self)['right']
        oncoming = self.env.sense(self)['oncoming']
        # True if there's another car in the same intersection
        car_in_int = not ((left or right or oncoming) is None)

        ### IMPLEMENT GAME THEORY
        # optimize the policy by implementing game theory
        # find the expected result in the situation
        #if left or right or oncoming in ['left', 'right', 'oncoming']:
            #agent needs to find a strategy based on 
            # 1 if there's a car in the intersection
            # 2 determine the maximin strategy in the intersection        
        ### IMPLEMENT GAME THEORY
        
        # calculate the q-value, as long as it is not in the initial state
        if self.previous_state is not None:
            v = self.q_table[self.previous_state][self.previous_action]
            x = reward + gamma * max_q
            q_hat = ((1 - alpha) * v) + (alpha * x)
            
            if q_hat is not None:
                self.q_table[self.previous_state][self.previous_action] = q_hat
        
        # save previous state and action to use for q-learning equation
        self.previous_state = state
        self.previous_action = action

        # observing results for debugging
        if self.env.done == True:
            self.success_tally += 1     
            self.successful_runs.append(self.run_trial)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit    

    def plot_success(successful_runs):
        x = range(0, 101)
        y = np.zeros(len(x))
        for x_coord in x:
            if x_coord in successful_runs:
                y[x_coord] = 1
        pl.title("Successful Runs")
        pl.hist(successful_runs, bins=len(x))
        pl.xlabel("Trial Run Number")
        pl.ylabel("Success (binary)")
        pl.show()

    print a.q_table
    print "List of trial numbers, where the agent successfully reached the destination:\n{}".format(a.successful_runs)
    print "Number of successful runs:\n{}".format(len(a.successful_runs))
    #print plot_success(a.successful_runs)


if __name__ == '__main__':
    run()