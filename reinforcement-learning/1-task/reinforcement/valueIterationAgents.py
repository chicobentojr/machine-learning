# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp
import util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for iteration in range(iterations):
            print '\nVALUE ITERATION ROUND', iteration, '\n'
            print 'self values', self.values
            states = self.mdp.getStates()
            print 'states', states
            for state in states:
                best_action = self.computeActionFromValues(state)
                print best_action

                print 'for state {} best action is {}' % (state, best_action)

                q_value = self.computeQValueFromValues(state, best_action)
                print 'for state {} best action is {} with {} qvalue' % state, best_action, q_value

                self.values[state] += q_value

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # util.raiseNotDefined()

        print 'comput QValue from Values\n'

        q = 0

        print 'getting transition from', state, 'with', action
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        print 'transitions', transitions
        for next_state, prob in transitions:
            reward = self.mdp.getReward(state, action, next_state)

            q += prob * reward

            print 'transition from', state, 'to', next_state, 'with prob', prob, 'and reward', reward
            # self.values[next_state] += reward
        print ''

        print 'returning qvalue from state', state, 'with action', action, 'as', q, '\n'
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        print 'compute Action From Values'
        print 'state', state
        print 'self values', self.values

        actions = self.mdp.getPossibleActions(state)

        print 'mpd actions', actions, len(actions)

        r = None

        if state == 'TERMINAL_STATE':
            r = None
        else:
            r = 'exit'
        print ''

        # if len(actions) == 1:
        #     r = actions[0]

        print 'returning action from value', r, '\n'
        return r

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        print 'getQValue action', action
        return self.computeQValueFromValues(state, action)
