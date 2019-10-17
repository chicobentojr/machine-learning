# sarsaAgents.py
# ------------------
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
# SARSA Agent extension by Anderson Tavares (anderson@dcc.ufmg.br)


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import collections

import random
import util
import math


class SarsaAgent(ReinforcementAgent):
    """
      Sarsa Agent
      run with: python gridworld.py -a s -k 100
      (any gridworld run with '-a s' will work, except for the manual agent)
      Useful options:
      --epsilon value
      --edecay value
      --lambda value


      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - computeAction
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, epsilon_decay=1, lamda=0, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.Q = util.Counter()
        self.eligibility_traces = util.Counter()
        self.lamda = lamda

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.Q[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)

        if not actions:
            return 0.0

        max_value = None

        for action in actions:
            q_value = self.getQValue(state, action)
            if max_value is None or q_value > max_value:
                max_value = q_value

        return max_value

    def computeActionFromQValues(self, state):
        """
          Compute the greedy action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)

        if not actions:
            return None

        best_actions = []
        best_action_value = self.getValue(state)

        for action in actions:
            q_value = self.getQValue(state, action)
            if q_value == best_action_value:
                best_actions.append(action)

        return random.choice(best_actions)

    def computeAction(self, state):
        """
          Compute the action to take in the given state.  With
          probability self.epsilon, take a random action and
          take the greedy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, it
          chooses None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        if legalActions:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)

        return action

    def getAction(self, state):
        """
          Returns the action computed in computeAction
        """
        "*** YOUR CODE HERE ***"
        return self.computeAction(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        q_value = self.getQValue(state, action)

        next_action = self.getAction(nextState)
        next_q_value = self.getQValue(nextState, next_action)

        td_error = reward + self.discount * next_q_value - q_value

        self.eligibility_traces[(state, action)] += 1

        for e_state, e_action in self.eligibility_traces.keys():
            self.Q[(e_state, e_action)] += self.alpha * td_error * \
                self.eligibility_traces[(e_state, e_action)]
            self.eligibility_traces[(e_state, e_action)
                                    ] *= self.discount * self.lamda

        if not self.getLegalActions(nextState):
            self.eligibility_traces = util.Counter()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanSarsaAgent(SarsaAgent):
    "Exactly the same as SarsaAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanSarsaAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        SarsaAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of SarsaAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = SarsaAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateSarsaAgent(PacmanSarsaAgent):
    """
       ApproximateSarsaAgent

       You should only have to overwrite getQValue
       and update.  All other SarsaAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanSarsaAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        if action is None:
            return 0

        features = self.featExtractor.getFeatures(state, action)

        q_value = 0
        for key, feature_value in features.items():
            w = self.weights[key]
            q_value += w * feature_value

        return q_value

    def update(self, state, action, nextState, reward):
        """
                   Should update your weights based on transition
                """
        "*** YOUR CODE HERE ***"
        q_value = self.getQValue(state, action)

        next_action = self.getAction(nextState)
        next_q_value = self.getQValue(nextState, next_action)

        td_error = reward + self.discount * next_q_value - q_value

        features = self.featExtractor.getFeatures(state, action)

        for feature, feature_value in features.items():
            self.weights[feature] += self.alpha * td_error * feature_value

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanSarsaAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
