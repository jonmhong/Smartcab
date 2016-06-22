import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

# these are used to analyze the agent's results 
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        # setting up the q_table
        self.q_table = {}
        self.initial_q = 4
        self.possible_actions = {}
        for move in self.env.valid_actions:
            self.possible_actions[move] = self.initial_q

        # these variable are initialized to remember the previous state and action (for q-learning)
        self.previous_action = None
        self.previous_state = None

        # these are debugging variables and for analysis
        self.successful_runs = []
        self.success_tally = 0
        self.run_trial = 1
        self.last_ten_trials = []
        self.time_steps = []
        
        # created to analyze the rewards
        self.net_reward = 0
        self.all_rewards = []


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.previous_action = None
        self.destination = self.planner.destination
        self.previous_state = None
        self.run_trial += 1
        
        self.all_rewards.append(self.net_reward)
        self.net_reward = 0
        
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.light = self.env.sense(self)['light']

        # these aren't used toward the state but will be used for experimentation
        left = self.env.sense(self)['left']
        right = self.env.sense(self)['right']
        oncoming = self.env.sense(self)['oncoming']
        car_in_int = not ((left or right or oncoming) is None) # True if there's another car in the current intersection

        self.state = (self.next_waypoint, self.light)
        
        # creating and decaying parameters
        self.gamma = 0.82 ** self.run_trial # discount rate
        self.alpha = 1.0 / self.run_trial # learning rate
        self.epsilon = 1.0 / self.run_trial # randomness
        
        # create new state in q_table if explored
        if (self.state is not None) and (self.state not in self.q_table):
            self.q_table[self.state] = {'forward': self.initial_q, 'left': self.initial_q, 
                                        'right': self.initial_q, None: self.initial_q}

        # TODO: Select action according to your policy
        # choose a random action epsilon percent of the time, else choose highest q-value action
        if random.random() > self.epsilon:
            for a, q in self.q_table[self.state].iteritems():
                if q == max(self.q_table[self.state].values()):
                    action = a
        else:
            action = random.choice(self.possible_actions.keys())

        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        q_hat = None
        max_q = max(self.q_table[self.state].values())

        # calculate the q-value, as long as it is not in the initial state
        if self.previous_state is not None:
            # bellman equation
            v = self.q_table[self.previous_state][self.previous_action]
            x = reward + self.gamma * max_q
            q_hat = ((1 - self.alpha) * v) + (self.alpha * x)
            
            self.q_table[self.previous_state][self.previous_action] = q_hat
        
        # save previous state and action to use in q-learning equation
        self.previous_state = self.state
        self.previous_action = action

        # analysis for the last 10 trials
        if self.run_trial > 90:
            location = self.env.agent_states[self]['location']
            destination = self.planner.destination
            # this keeps track of the next waypoint, action, traffic light, cars in the intersection, agent's current location, and destination
            # at each step, during the last 10 trials
            self.last_ten_trials.append((self.next_waypoint, action, self.light, left, right, oncoming, location, destination))
            self.net_reward += reward # this is used to check the net rewards at each trial
            if self.env.done == True:
                self.time_steps.append((self.run_trial, t, self.net_reward))

        # observing results for analysis and debugging
        if self.env.done == True:
            self.success_tally += 1
            self.successful_runs.append(self.run_trial)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        
def plot_success(successful_runs):
        x = range(0, 101)
        y = np.zeros(len(x))
        for x_coord in x:
            if x_coord in successful_runs:
                y[x_coord] = 1
        pl.title("Successful Runs")
        pl.hist(successful_runs, 
                bins=len(x))
        pl.xlabel("Trial Run Number")
        pl.ylabel("Success (binary)")
        pl.show()


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit    

    print a.q_table
    print a.last_ten_trials
    print a.time_steps
    print a.all_rewards
    print "List of trial numbers, where the agent successfully reached the destination:\n{}".format(a.successful_runs)
    print "Number of successful runs:\n{}".format(len(a.successful_runs))
    print plot_success(a.successful_runs)


if __name__ == '__main__':
    run()