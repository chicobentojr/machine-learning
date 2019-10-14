# dynaAgents.py
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

# Dyna Agent support by Anderson Tavares (artavares@inf.ufrgs.br)


from game import *
from learningAgents import ReinforcementAgent

import random
import util
import math


class DynaQAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
        - self.plan_steps (number of planning iterations)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, plan_steps=5, kappa=0, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.Q = util.Counter()
        self.model = util.Counter()
        self.plan_steps = plan_steps

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
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
          Compute the best action to take in a state.  Note that if there
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

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if legalActions:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here.

          NOTE: You should never call this function,
          it will be called on your behalf

          NOTE2: insert your planning code here as well
        """
        "*** YOUR CODE HERE ***"
        q_value = self.getQValue(state, action)

        next_action = self.computeActionFromQValues(nextState)
        next_q_value = self.getQValue(nextState, next_action)

        new_q_value = q_value + self.alpha * \
            (reward + self.discount * next_q_value - q_value)

        self.Q[(state, action)] = new_q_value  # Direct RL
        self.model[(state, action)] = (reward, nextState)  # Model learning

        # Planning
        for i in range(self.plan_steps):
            state, action = random.choice(self.model.keys())
            reward, nextState = self.model[(state, action)]

            q_value = self.getQValue(state, action)
            next_action = self.computeActionFromQValues(nextState)
            next_q_value = self.getQValue(nextState, next_action)

            new_q_value = q_value + self.alpha * \
                (reward + self.discount * next_q_value - q_value)

            self.Q[(state, action)] = new_q_value

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanDynaQAgent(DynaQAgent):
    "Exactly the same as DynaAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanDynaAgent -a epsilon=0.1

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
        DynaQAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of DynaAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = DynaQAgent.getAction(self, state)
        self.doAction(state, action)
        return action
