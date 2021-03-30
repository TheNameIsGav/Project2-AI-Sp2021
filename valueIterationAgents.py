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

import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        states = self.mdp.getStates()

        for _ in range(self.iterations): #go through every iteration 
            values = util.Counter()
            for state in states:  #go through every state 
                maxQ = float('-inf') #be as low as we can be
                if self.mdp.isTerminal(state): #if we're terminal , who gives a hoot
                    continue
                actions = self.mdp.getPossibleActions(state) #get all possible moves 
                for action in actions: #for every action
                    q = self.getQValue(state, action) #compute q value of state and action(equivalent to computeQValuesFromValues(state, action))
                    if q > maxQ: #if q is greater than max
                        maxQ = q #update
                values[state] = maxQ
            self.values = values

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
        value = 0
        statesAndProbabilities = self.mdp.getTransitionStatesAndProbs(state, action) #Gets next states and their probabilities
        for newState, prob in statesAndProbabilities:
            value += prob * (self.mdp.getReward(state, action, newState) + self.discount*self.values[newState]) #adds the values of a state's reward, and the discount for moving there
        return value
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        #Had to rewrite for use with terminal states
        if self.mdp.isTerminal(state):
          return None

        #gets all possible actions and initalizes a new counter
        actions = self.mdp.getPossibleActions(state)
        qPerAction = util.Counter()

        #keeps track of the action and the q value
        for action in actions:
          qPerAction[action] = self.getQValue(state, action)

        #returns the action with the highest q value
        return qPerAction.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*
        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates() #get all the states
        for i in range(self.iterations): #go through every iteration
          state = states[i % len(states)] #pick a cycliccally changing state
          maxQ = float('-inf')
          if not self.mdp.isTerminal(state): #if we're not terminal
            actions = self.mdp.getPossibleActions(state) #get actions
            for action in actions: #for every action
                q = self.getQValue(state, action) #compute q value of state and action(equivalent to computeQValuesFromValues(state, action))
                if q > maxQ: #if q is greater than max
                    maxQ = q #update
            self.values[state] = maxQ

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def getPredecessors(self):
        #finds all the predecessors of all states

        stateMappings = {} #empty dictionary of state : the set of {predecessors}
        for state in self.mdp.getStates():

            if self.mdp.isTerminal(state): #we do not want terminal states in our predecessors
                continue

            actions = self.mdp.getPossibleActions(state)
            for action in actions:

                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob == 0:
                        continue
                    if nextState in stateMappings: #we have seen this state before and need to add our current state to next states predecessors
                        stateMappings[nextState].add(state)
                    else:
                        stateMappings[nextState] = {state} #creates a new set of states
        return stateMappings


    def maxQSolve(self, state):
        maxQ = float('-inf')
        actions = self.mdp.getPossibleActions(state) #get actions
        for action in actions: #for every action
            q = self.getQValue(state, action) #compute q value of state and action(equivalent to computeQValuesFromValues(state, action))
            if q > maxQ: #if q is greater than max
                maxQ = q #update
        return maxQ

    def runValueIteration(self):
        
        preds = self.getPredecessors()
        pQ = util.PriorityQueue()

        for state in self.mdp.getStates():
            
            if self.mdp.isTerminal(state):
                continue

            diff = abs(self.values[state] - self.maxQSolve(state))
            pQ.push(state, -diff) #min heap, prio to updating higher error

        for i in range(0, self.iterations):
            if pQ.isEmpty():
                return
            state = pQ.pop()

            if not self.mdp.isTerminal(state):
                self.values[state] = self.maxQSolve(state)

            for pred in preds[state]:
                if(self.mdp.isTerminal(pred)):
                    continue
                diff = abs(self.values[pred] - self.maxQSolve(pred))
                if diff > self.theta:
                    pQ.update(pred, -diff)
                    