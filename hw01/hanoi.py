# Basic Hanoi-state Class
class HanoiState:

    def __init__(self, pegList):
        self.pegs = pegList
        self.noPegs = len(self.pegs)
        self.heights = self.getHeights()
        self.valid = self.isValid()

    # compute each peg's heights
    def getHeights(self):
        return [len(x) for x in self.pegs]

    # isValid (Checking all discs)
    def isValid(self):
        # fill out! (1 point)
        ## 전체 기둥들에 대해
        for i in range(self.noPegs):
            ## 각 기둥들의 원판에 대해
            for j in range(self.heights[i] - 1):
                ## 작은 원판이 큰 원판 아래에 있으면 False
                if self.pegs[i][j] < self.pegs[i][j + 1]:
                    return False
        ## 모든 원판이 잘 쌓여있으면 True
        return True

    # check two HanoiStates are identical
    # not by their address, but by their values!
    def isSame(self, S1):
        if self.noPegs != S1.noPegs:
            return False
        for i in range(self.noPegs):
            if self.heights[i] != S1.heights[i]:
                return False
            for j in range(self.heights[i]):
                if self.pegs[i][j] != S1.pegs[i][j]:
                    return False
        return True

    # the overloading method for equality
    # With this method, you may use A == B or (item in open)
    def __eq__(self, other):
        if not isinstance(other, HanoiState):
            return False
        return self.isSame(other)

# When generate a new HanoiState, follow the guideline below.
# 1) get possible actions from the current state using getPossibleActions
# 2) deep-copy the current state. (It's very important!)
# 3) Try to move with moveDisc, and generated action "over copied state".
# 4) Finally, check whether the copied state is valid.

# generate possible actions

    def getPossibleActions(self):
        nz_pegs = []
        for i in range(self.noPegs):
            if self.heights[i] != 0:
                nz_pegs.append(i)
        nnz_pegs = len(nz_pegs)
        actions = []
        for i in range(nnz_pegs):
            for j in range(self.noPegs):
                if nz_pegs[i] != j:
                    actions.append((nz_pegs[i], j))
        return actions

    # moveDisc from A to B (action=(fromPeg, toPeg))
    def moveDisc(self, action):
        # fill out! (0.5 point)
        fp = action[0]  ## from peg
        tp = action[1]  ## to peg

        ## tp 에 해당하는 list 에 fp 에 해당하는 list 의 top 추가
        self.pegs[tp].append(self.pegs[fp].pop())
        self.heights = self.getHeights()
        self.valid = self.isValid()


# HanoiState initialization
import copy

peg1 = [3, 2, 1]
peg2 = []
peg3 = []
S0 = HanoiState([peg1, peg2, peg3])
Goal = HanoiState([peg2.copy(), peg3.copy(), peg1.copy()])
# # testing the equality
# print(S0.isSame(Goal))
# print(S0.isSame(S0))
# # testing the __eq__ method
# print(S0 == Goal)
# print(S0 == S0)
# print(S0.heights)

# action = S0.getPossibleActions()
# print("action", action)
# S1 = copy.deepcopy(S0)  # very important!
# S1.moveDisc(action[0])
# print(S1.pegs)
# S1.moveDisc(action[0])
# print(S1.pegs)
# if not S1.valid:
#     print('S1 is not valid, Please ignore it')

# DFS/BFS
open = [S0]
close = []
goal = Goal

# queue: add => open.append. remove => open[0], open.pop(0)
# stack: add => open.append. remove => open[-1], open.pop()
while len(open):
    # step 1
    item = open[-1]
    open.pop()
    close.append(item)
    print('------------------')
    print('visiting ', item.pegs)

    # step 2
    if item == goal:
        print('the goal reached, exiting')
        break

    # step 3
    for action in item.getPossibleActions():
        ## 기존의 state 를 건들이지 않기 위함
        S1 = copy.deepcopy(item)  # fill out (0.5 points)

        S1.moveDisc(action)
        ## 허용되지 않는 state 일 때, 다시 for 문으로 돌아감
        if not S1.valid:
            print("not valid!")
            continue

        # if s1 is not in the open and not in the close.
        if (not S1 in close) & (not S1 in open):
            open.append(S1)
