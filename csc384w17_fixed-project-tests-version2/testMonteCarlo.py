import unittest
from pacman import *
from multiAgents import *
import util, layout
import textDisplay
import datetime

class TestGetAction(unittest.TestCase):

    def setUp(self):
        self.rules = ClassicGameRules(30)
        self.ghostType = loadAgent('RandomGhost', nographics=True)
        self.rules.quiet = False

    def testSingleMove(self):
        map = layout.tryToLoad('test_cases/stuckClassic.lay')
        pacman = MonteCarloAgent()
        game = self.rules.newGame(map, pacman, [], textDisplay.NullGraphics(), True, True)
        state = game.state
        pacman.getAction(state)
        self.assertEqual(len(pacman.plays), 0)

    def testTimeOut(self):
        map = layout.getLayout('smallClassic')
        pacman = MonteCarloAgent()
        ghosts = [self.ghostType(i + 1) for i in range(2)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        begin = datetime.datetime.utcnow()
        pacman.getAction(state)
        total = datetime.datetime.utcnow() - begin
        self.assertAlmostEqual(total, datetime.timedelta(milliseconds=50), delta=datetime.timedelta(milliseconds=400))

    def testSelection(self):
        map = layout.getLayout('testClassic')
        pacman = MonteCarloAgent()
        ghosts = [self.ghostType(i + 1) for i in range(2)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        for i in range(5):# pacman cannot die in 5 moves on this layout
            action = pacman.getAction(state)
            state = state.generateSuccessor(0, action)
        stud_act = pacman.getAction(state)
        actions = state.getLegalActions(0)
        max_val = max(float(pacman.wins[state.generateSuccessor(0, a)]) / pacman.plays[state.generateSuccessor(0, a)]
                      for a in actions if state.generateSuccessor(0, a) in pacman.plays)
        max_acts = [a for a in actions if state.generateSuccessor(0, a) in pacman.plays and
                    float(pacman.wins[state.generateSuccessor(0, a)]) /
                    pacman.plays[state.generateSuccessor(0, a)] == max_val]
        self.assertIn(stud_act, max_acts)

class TestRunSimulation(unittest.TestCase):

    def setUp(self):
        self.rules = ClassicGameRules(30)
        self.ghostType = loadAgent('RandomGhost', nographics=True)
        self.rules.quiet = False

    def testDepth(self):
        map = layout.getLayout('smallClassic')
        pacman = MonteCarloAgent(depth=1)
        ghosts = [self.ghostType(i + 1) for i in range(1)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        states = [state]
        for i in range(2):
            states = [s.generateSuccessor(i, a) for s in states for a in s.getLegalActions(i)]
            for s in states:
                pacman.plays[s] = 1
                pacman.wins[s] = 0
        # If pacman searches to correct depth after expansion, will result in one state getting added to plays.
        prev = len(pacman.plays)
        pacman.run_simulation(state)
        self.assertEqual(prev + 1, len(pacman.plays))

    def testExpansion(self):
        map = layout.getLayout('smallClassic')
        pacman = MonteCarloAgent()
        ghosts = [self.ghostType(i + 1) for i in range(1)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        for i in range(2):
            pacman.run_simulation(state)
        self.assertEqual(len(pacman.plays), 2)

    def earlyWin(self):
        map = layout.tryToLoad('test_cases/shortClassic.lay')
        pacman = MonteCarloAgent()
        ghosts = [self.ghostType(i + 1) for i in range(1)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        actions = state.getLegalActions(0)
        # Run it once for every successor.
        for a in actions:
            succ = state.generateSuccessor(0, a)
            if a == Directions.EAST:
                pacman.plays[succ] = 1
                pacman.wins[succ] = 1
            else:
                pacman.plays[succ] = 1
                pacman.wins[succ] = 0
        # Based on UCB1 should pick state where will win, if don't exit early, code will crash.
        pacman.run_simulation(state)

    def testNoInfo(self):
        map = layout.getLayout('smallClassic')
        pacman = MonteCarloAgent()
        ghosts = [self.ghostType(i + 1) for i in range(1)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        pacman.plays[state] = 1
        pacman.wins[state] = 0
        succ = [state.generateSuccessor(0, a) for a in state.getLegalActions(0)]
        for s in succ[1:len(succ)]:
            pacman.plays[s] = 1
            pacman.wins[s] = 0
        pacman.run_simulation(state)
        # Should play the single state without information.
        self.assertEqual(pacman.plays[succ[0]], 1)

    def testUCB1(self):
        map = layout.getLayout('smallClassic')
        pacman = MonteCarloAgent()
        ghosts = [self.ghostType(i + 1) for i in range(1)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        succ = [state.generateSuccessor(0, a) for a in state.getLegalActions(0)]
        pacman.plays[succ[0]] = 1
        pacman.wins[succ[0]] = 1
        for i in range(1, len(succ)):
            pacman.plays[succ[i]] = 1
            pacman.wins[succ[i]] = 0
        pacman.run_simulation(state)
        self.assertEqual(pacman.plays[succ[0]], 2)

    def testUCB2(self):
        map = layout.getLayout('smallClassic')
        pacman = MonteCarloAgent()
        ghosts = [self.ghostType(i + 1) for i in range(1)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        succ = [state.generateSuccessor(0, a) for a in state.getLegalActions(0)]
        pacman.plays[succ[0]] = 1
        pacman.wins[succ[0]] = 0
        for i in range(1, len(succ)):
            pacman.plays[succ[i]] = 2
            pacman.wins[succ[i]] = 0
        pacman.run_simulation(state)
        self.assertEqual(pacman.plays[succ[0]], 2)


if __name__ == '__main__':
    unittest.main()
