# multiAgents.py
# --------------
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

  
from util import manhattanDistance
from game import Directions
import random, util
import datetime
import math

from game import Agent

class MultiAgentSearchAgent(Agent):
	"""
	  This class provides some common elements to all of your
	  multi-agent searchers.  Any methods defined here will be available
	  to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	  You *do not* need to make any changes here, but you can if you want to
	  add functionality to all your adversarial search agents.  Please do not
	  remove anything, however.

	  Note: this is an abstract class: one that should not be instantiated.  It's
	  only partially specified, and designed to be extended.  Agent (game.py)
	  is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent (question 2)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action from the current gameState using self.depth
		  and self.evaluationFunction.

		  Here are some method calls that might be useful when implementing minimax.

		  gameState.getLegalActions(agentIndex):
			Returns a list of legal actions for an agent
			agentIndex=0 means Pacman, ghosts are >= 1

		  gameState.generateSuccessor(agentIndex, action):
			Returns the successor game state after an agent takes an action

		  gameState.getNumAgents():
			Returns the total number of agents in the game
		"""
		"*** YOUR CODE HERE ***"
		curDepthVal = []
		legalActions = gameState.getLegalActions(self.index)
		# for every successor state under pacman, get state value for each state
		# and return the action corresponding to that state value
		for action in legalActions:
			SuccessorState = gameState.generateSuccessor(self.index, action)
			SuccessorStateVal = self.DFMiniMax(SuccessorState, self.depth, self.index+1)
			curDepthVal.append (SuccessorStateVal)
		return legalActions[curDepthVal.index(max(curDepthVal))]

	def DFMiniMax(self, gameState, depth, agentIndex):
		# no legal actions available or reach the last depth specified by user, then 
		# directly evaluate the state
		legalActions = gameState.getLegalActions(agentIndex)
		if(depth == 0 or not legalActions):
			return scoreEvaluationFunction(gameState)
		else:
			curDepthVal = []
			# find the max value of all successor states under pacman
			if(agentIndex == self.index):
				for action in legalActions:
					SuccessorState = gameState.generateSuccessor(agentIndex, action)
					SuccessorStateVal = self.DFMiniMax(SuccessorState, depth, agentIndex+1)
					curDepthVal.append (SuccessorStateVal)
				return max(curDepthVal)
			# for each ghost, find the min value of all successor states under that ghost
			elif(agentIndex != gameState.getNumAgents() - 1):
				for action in legalActions:
					SuccessorState = gameState.generateSuccessor(agentIndex, action)
					SuccessorStateVal = self.DFMiniMax(SuccessorState, depth, agentIndex+1)
					curDepthVal.append (SuccessorStateVal)
				return min(curDepthVal)
			# for each ghost, find the min value of all successor states under that ghost, 
			# decrease the depth by 1
			else:
				for action in legalActions:
					SuccessorState = gameState.generateSuccessor(agentIndex, action)
					SuccessorStateVal = self.DFMiniMax(SuccessorState, depth - 1, self.index)
					curDepthVal.append (SuccessorStateVal)
				return min(curDepthVal)

				
class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action using self.depth and self.evaluationFunction
		"""
		"*** YOUR CODE HERE ***"
		legalActions = gameState.getLegalActions(self.index)
		curDepthVal = []
		# initialize alpha and beta
		alpha = float("inf")
		beta = float("-inf")
		# for every successor state under pacman, get state value for each state and return the
		# action corresponding to that state value, update beta at the same time
		for action in legalActions:
			SuccessorState = gameState.generateSuccessor(self.index, action)
			SuccessorStateVal = self.AlphaBeta(SuccessorState, self.depth, self.index+1, alpha, beta)
			curDepthVal.append (SuccessorStateVal)
			if(beta < SuccessorStateVal):
				beta = SuccessorStateVal
		return legalActions[curDepthVal.index(max(curDepthVal))]

	def AlphaBeta(self, gameState, depth, agentIndex, alpha, beta):
		# no legal actions available or reach the last depth specified by user, then 
		# directly evaluate the state
		legalActions = gameState.getLegalActions(agentIndex)
		if(depth == 0 or not legalActions):
			return scoreEvaluationFunction(gameState)
		else:
			curDepthVal = []
			# find the max value of all successor states under pacman, do pruning and update
			# beta at the same time
			if(agentIndex == self.index):
				for action in legalActions:
					SuccessorState = gameState.generateSuccessor(agentIndex, action)
					SuccessorStateVal = self.AlphaBeta(SuccessorState, depth, agentIndex+1, alpha, beta)
					curDepthVal.append (SuccessorStateVal)
					if(alpha < SuccessorStateVal):
						return SuccessorStateVal
					if(beta < SuccessorStateVal):
						beta = SuccessorStateVal
				return max(curDepthVal)
			# for each ghost, find the min value of all successor states under that ghost, do pruning and 
			# update alpha at the same time
			elif(agentIndex != gameState.getNumAgents() - 1):
				for action in legalActions:
					SuccessorState = gameState.generateSuccessor(agentIndex, action)
					SuccessorStateVal = self.AlphaBeta(SuccessorState, depth, agentIndex+1, alpha, beta)
					curDepthVal.append (SuccessorStateVal)
					if(beta > SuccessorStateVal):
						return SuccessorStateVal
					if(alpha > SuccessorStateVal):
						alpha = SuccessorStateVal
				return min(curDepthVal)
			# for each ghost, find the min value of all successor states under that ghost, do pruning and 
			# update alpha at the same time, decrease the depth by 1
			else:
				for action in legalActions:
					SuccessorState = gameState.generateSuccessor(agentIndex, action)
					SuccessorStateVal = self.AlphaBeta(SuccessorState, depth - 1, self.index, alpha, beta)
					curDepthVal.append (SuccessorStateVal)
					if(beta > SuccessorStateVal):
						return SuccessorStateVal
					if(alpha > SuccessorStateVal):
						alpha = SuccessorStateVal
				return min(curDepthVal)


class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""

	def getAction(self, gameState):
		"""
		  Returns the expectimax action using self.depth and self.evaluationFunction

		  All ghosts should be modeled as choosing uniformly at random from their
		  legal moves.
		"""
		"*** YOUR CODE HERE ***"
		curDepthVal = []
		legalActions = gameState.getLegalActions(self.index)
		# for every successor state under pacman, get state value for each state
		# and return the action corresponding to that state value
		for action in legalActions:
			SuccessorState = gameState.generateSuccessor(self.index, action)
			SuccessorStateVal = self.Expectimax(SuccessorState, self.depth, self.index+1)
			curDepthVal.append (SuccessorStateVal)
		return legalActions[curDepthVal.index(max(curDepthVal))]

	def Expectimax(self, gameState, depth, agentIndex):
		# no legal actions available or reach the last depth specified by user, then 
		# directly evaluate the state
		legalActions = gameState.getLegalActions(agentIndex)
		if(depth == 0 or not legalActions):
			return scoreEvaluationFunction(gameState)
		else:
			curDepthVal = []
			# find the max value of all successor states under pacman
			if(agentIndex == self.index):
				for action in legalActions:
					SuccessorState = gameState.generateSuccessor(agentIndex, action)
					SuccessorStateVal = self.Expectimax(SuccessorState, depth, agentIndex+1)
					curDepthVal.append (SuccessorStateVal)
				return max(curDepthVal)
			# for each ghost, find the average value of all successor states under that ghost
			elif(agentIndex != gameState.getNumAgents() - 1):
				for action in legalActions:
					SuccessorState = gameState.generateSuccessor(agentIndex, action)
					SuccessorStateVal = self.Expectimax(SuccessorState, depth, agentIndex+1)
					curDepthVal.append (SuccessorStateVal)
				return sum(curDepthVal)/len(curDepthVal)
			# for each ghost, find the average value of all successor states under that ghost, 
			# decrease the depth by 1
			else:
				for action in legalActions:
					SuccessorState = gameState.generateSuccessor(agentIndex, action)
					SuccessorStateVal = self.Expectimax(SuccessorState, depth - 1, self.index)
					curDepthVal.append (SuccessorStateVal)
				return sum(curDepthVal)/len(curDepthVal)

class MonteCarloAgent(MultiAgentSearchAgent):
	"""
		Your monte-carlo agent (question 5)
		***UCT = MCTS + UBC1***
		TODO:
		1) Complete getAction to return the best action based on UCT.
		2) Complete runSimulation to simulate moves using UCT.
		3) Complete final, which updates the value of each of the states visited during a play of the game.

		* If you want to add more functions to further modularize your implementation, feel free to.
		* Make sure that your dictionaries are implemented in the following way:
			-> Keys are game states.
			-> Value are integers. When performing division (i.e. wins/plays) don't forget to convert to float.
	  """

	def __init__(self, evalFn='mctsEvalFunction', depth='-1', timeout='50', numTraining=100, C='2', Q=None):
		# This is where you set C, the depth, and the evaluation function for the section "Enhancements for MCTS agent".
		if Q:
			if Q == 'minimaxClassic':
				self.C = 1
			elif Q == 'testClassic':
				self.C = 1
				pass
			elif Q == 'smallClassic':
				depth = 3
				pass
			else: # Q == 'contestClassic'
				assert( Q == 'contestClassic' )
				depth = 3
			evalFn = 'betterEvaluationFunction'
		# Otherwise, your agent will default to these values.
		else:
			self.C = int(C)
			# If using depth-limited UCT, need to set a heuristic evaluation function.
			if int(depth) > 0:
				evalFn = 'scoreEvaluationFunction'
		self.states = []
		self.plays = dict()
		self.wins = dict()
		self.calculation_time = datetime.timedelta(milliseconds=int(timeout))

		self.numTraining = numTraining

		"*** YOUR CODE HERE ***"

		MultiAgentSearchAgent.__init__(self, evalFn, depth)

	def update(self, state):
		"""
		You do not need to modify this function. This function is called every time an agent makes a move.
		"""
		self.states.append(state)

	def getAction(self, gameState):
		"""
		Returns the best action using UCT. Calls runSimulation to update nodes
		in its wins and plays dictionary, and returns best successor of gameState.
		"""
		"*** YOUR CODE HERE ***"
		games = 0
		begin = datetime.datetime.utcnow()
		while datetime.datetime.utcnow() - begin < self.calculation_time:
			self.run_simulation(gameState)
			games += 1
		legalActions = gameState.getLegalActions(self.index)
		SuccessorStateVal = []
		for action in legalActions:
			successorState = gameState.generateSuccessor(self.index, action)
			try:
				SuccessorStateVal.append(float(self.wins[successorState])/float(self.plays[successorState]))
			except:
				SuccessorStateVal.append(float("inf"))
		return legalActions[SuccessorStateVal.index(max(SuccessorStateVal))]

	def run_simulation(self, state):
		"""
		Simulates moves based on MCTS.
		1) (Selection) While not at a leaf node, traverse tree using UCB1.
		2) (Expansion) When reach a leaf node, expand.
		4) (Simulation) Select random moves until terminal state is reached.
		3) (Backpropapgation) Update all nodes visited in search tree with appropriate values.
		* Remember to limit the depth of the search only in the expansion phase!
		Updates values of appropriate states in search with with evaluation function.
		"""
		"*** YOUR CODE HERE ***"
		if(len(self.plays) == 0 or state not in self.plays.keys()):
			# directly roll out
			self.plays[state] = 1
			self.wins[state] = self.simulation(state, self.depth, self.index)
			# states = [state.generateSuccessor(self.index, a) for a in state.getLegalActions(self.index)]
			# for s in states:
			# 	self.plays[s] = 0
			# 	self.wins[s] = 0
		else:
			# initial state traversal is automatically added
			nodes_traversed = [state]
			cur_state = state
			cur_states = []
			agentIndex = self.index
			while(1):
				if(self.plays[cur_state] == 0):
					# roll out
					score = self.simulation(cur_state, self.depth, agentIndex)
					for s in nodes_traversed:
						self.plays[s] += 1
						self.wins[s] += score
					break
				cur_states = [cur_state.generateSuccessor(agentIndex, a) for a in cur_state.getLegalActions(agentIndex)]
				# if no available actions can be taken
				if(not cur_states):
					break
				# any missed extended nodes should be extended manually
				if any(s in cur_states for s in self.plays.keys()):
					for s in cur_states:
						if(s not in self.plays.keys()):
							self.plays[s] = 0
							self.wins[s] = 0
				# if not any(s in cur_states for s in self.plays.keys()):
				if(cur_states[0] not in self.plays.keys()):
					# expansion
					for s in cur_states:
						self.plays[s] = 0
						self.wins[s] = 0
					cur_state = cur_states[0]
				 	if agentIndex == self.index:
						nodes_traversed.append(cur_state)
					# roll out
					score = self.simulation(cur_state, self.depth, agentIndex)
					for s in nodes_traversed:
						self.plays[s] += 1
						self.wins[s] += score
					break
				else:
					# UCB search for node
					ucb1Dict = {}
					for s in cur_states:
						ucb1Dict[s] = self.UCB1(cur_state, s)
					cur_state = max(ucb1Dict.iterkeys(), key=(lambda key: ucb1Dict[key]))
					if agentIndex == self.index:
						nodes_traversed.append(cur_state)
				agentIndex += 1
				if(agentIndex == state.getNumAgents()):
					agentIndex = self.index

	def final(self, state):
		"""
		Called by Pacman game at the terminal state.
		Updates search tree values of states that were visited during an actual game of pacman.
		"""
		"*** YOUR CODE HERE ***"
		pass

	def UCB1(self, parentState, state):
		try:
			v = float(self.wins[state])/float(self.plays[state])
			ucb1Val = v + self.C * (math.log(float(self.plays[parentState])/float(self.plays[state])))**(.5)
		except:
			ucb1Val = float("inf")
		return ucb1Val

	def simulation(self, state, depth, agentIndex):
		legalActions = state.getLegalActions(agentIndex)
		if(depth == -1):
			if(not legalActions):
				return mctsEvalFunction(state)
		else:
			if(depth == 0 or not legalActions):
				return self.evaluationFunction(state)
		random_number = random.randint(0, len(legalActions)-1)
		action = legalActions[random_number]
		if(agentIndex < state.getNumAgents()-1):
			score = self.simulation(state.generateSuccessor(agentIndex, action), depth, agentIndex+1)
		else:
			if(depth == -1):
				score = self.simulation(state.generateSuccessor(agentIndex, action), depth, self.index)
			else:
				score = self.simulation(state.generateSuccessor(agentIndex, action), depth-1, self.index)
		return score

def mctsEvalFunction(state):
	"""
	Evaluates state reached at the end of the expansion phase.
	"""
	return 1 if state.isWin() else 0

def scoreEvaluationFunction(state):
	"""
	 This default evaluation function just returns the score of the state.
	 The score is the same one displayed in the Pacman GUI.

	 This evaluation function is meant for use with adversarial search agents
	"""
	return state.getScore()

def betterEvaluationFunction(state):
	"""
	  Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	  evaluation function (question 5).

	  DESCRIPTION: <write something here so we know what you did>
	"""
	"*** YOUR CODE HERE ***"
	# Pacman position
	pacman_pos = state.getPacmanPosition()

	# total distance between Pacman and all the ghosts
	ghost_pos_list = state.getGhostPositions()
	pacman_to_ghost = [manhattanDistance(pacman_pos, ghost_pos) for ghost_pos in ghost_pos_list]
	pacman_to_ghost = sum(pacman_to_ghost)
	# total food that is around Pacman
	pacman_to_food = 0
	surrounding = [[i,j] for i in range(-2,3) for j in range(-2,3)]
	food_pos_list = state.getFood()
	for food in surrounding:
		x = pacman_pos[0] + food[0]
		y = pacman_pos[1] + food[1]
		try:
			if(food_pos_list[x][y]):
				pacman_to_food += 1
		except:
			pass
	num_ghost = state.getNumAgents() - 1
	if(pacman_to_ghost <= num_ghost):
		score = -50
	else:
		score = 10 * pacman_to_ghost + 50 * pacman_to_food	
	return score

# Abbreviation
better = betterEvaluationFunction

