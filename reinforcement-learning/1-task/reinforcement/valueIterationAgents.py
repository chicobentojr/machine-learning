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

        print '\nINITING ITERATION AGENT WITH {} ITERATIONS\n'.format(
            iterations)

        for iteration in range(iterations):
            print '\nVALUE ITERATION ROUND', iteration, '\n'
            print 'self values', self.values
            states = self.mdp.getStates()
            print 'states', states
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    transitions = self.mdp.getTransitionStatesAndProbs(
                        state, action)
                    for next_state, prob in transitions:
                        reward = self.mdp.getReward(state, action, next_state)

                        self.values[state] += prob * \
                            (reward + self.values[state])

                # print 'best action is {}'.format(best_action)

                # print 'for state {}, best action is {}'.format(
                #     state, best_action)

                # q_value = self.getQValue(state, best_action)
                # print 'for state {} best action is {} with {} qvalue' % state, best_action, q_value
                # q_value = 0
                # self.values[state] += q_value

        print 'self values after {} iteration {}'.format(
            iterations, self.values)
        print '\nEND ITERATION\n'

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

        print 'COMPUT QVALUE FROM VALUES s {} a {}\n'.format(state, action)
        print 'self values', self.values

        q = 0
        best_state = self.values.argMax()
        print 'best state {}'.format(best_state)

        print 'getting transition from', state, 'with', action
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        print 'transitions', transitions
        for next_state, prob in transitions:
            reward = self.mdp.getReward(state, action, next_state)

            q += prob * (reward + self.values[next_state])

            print 'transition from', state, 'to', next_state, 'with prob', prob, 'and reward', reward
            # self.values[next_state] += reward
        print ''

        # self.values[state] = q

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
        import math
        print 'COMPUTE ACTION FROM VALUES s {}\n'.format(state)
        print 'self values', self.values

        actions = self.mdp.getPossibleActions(state)
        best_state = self.values.argMax()

        print 'possible actions', actions
        print 'best state {}'.format(best_state)

        best_action = None

        if state == 'TERMINAL_STATE':
            best_action = None
        else:
            max_reward = None
            max_prob = None
            for action in actions:
                transitions = self.mdp.getTransitionStatesAndProbs(
                    state, action)
                print 'for action {} are this transitions {}'.format(
                    action, transitions)
                for next_state, prob in transitions:
                    reward = self.mdp.getReward(state, action, next_state)
                    print 'transition {} from {} to {} with prob {} and reward {}'.format(
                        action, state, next_state, prob, reward)

                    if next_state == best_state and (prob > max_prob or not max_prob):
                        max_prob = prob
                        best_action = action

                    if reward > max_reward or not max_reward:
                        max_reward = reward
                        best_action = action
                        # self.values[state] = max_reward

        print 'returning action {} from state {}\n'.format(best_action, state)
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        print 'getQValue action', action
        return self.computeQValueFromValues(state, action)
