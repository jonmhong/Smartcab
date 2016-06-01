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
        
        initialize_q = 12
        self.possible_actions = {}
        for move in self.env.valid_actions:
            self.possible_actions[move] = initialize_q
        
        self.breaking_law_tally = 0
        self.total_moves = 0

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
        
        previous_state = self.state
        
        # TODO: Update state
        # create state entry in q_table as the agent explores a new state
        self.location = self.env.agent_states[self]['location']
        self.heading = self.env.agent_states[self]['heading']
        self.light = self.env.sense(self)['light']
        self.state = (self.location, self.heading, self.light)
        
        
        # tweak these variables
        gamma = 0.5 # discounting rate 
        alpha = 0.1 # learning rate to update q-value
        epsilon = 0.05 # chance of choosing a random action for simulated annealing
        
        if self.state is not None and self.state not in self.q_table:
            self.q_table[self.state] = self.possible_actions
        #else:
            # TODO*: update q-table rewards, according to state and action
        #    pass
        
        # TODO: Select action according to your policy
        action = None
        
        for move, q_value in self.q_table[self.state].iteritems():
            max_q_value = max(self.q_table[self.state].values())
            if q_value == max_q_value and max_q_value > 0:
                policy = move
                if random.random() > epsilon:
                    action = move
                else:
                    action = random.choice(self.env.valid_actions)
            else:
                action = random.choice(self.env.valid_actions)
        

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.q_table[self.state][action] = reward
        
        # set alpha, then decay it over time
        

        # TODO: Learn policy based on state, action, reward
        # utility function
        if previous_state is not None:
            if (previous_state[2] == 'red') and (self.state[0] != previous_state[0]):
                self.breaking_law_tally += 1
        self.total_moves += 1
        
        
        #q_hat = ((1 - alpha) * self.q_table[state][action]) + (alpha * (gamma * sum(policy)))
        
        print "previous state: {}".format(previous_state)
        print "current state: {}".format(self.state)
        print "illegal moves tally: {}".format(self.breaking_law_tally)
        print "total moves: {}".format(self.total_moves)
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
