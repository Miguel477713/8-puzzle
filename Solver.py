import numpy, random, os, heapq
from time import time

if os.name == 'nt':
    os.system('cls')  # Clear the console for Windows
elif os.name == 'posix':
    os.system('clear')  # Clear the console for Linux/Mac

start_time = time()

sizeP = 0
sizeQ = 0
HEURISTIC_WEIGHT = 2

def RandomizeRanks():
    """
    Shuffles the list of numbers from 1 to 8 and -1.
    Returns a shuffled list.
    """
    ranks = [1, 2, 3, 4, 5, 6, 7, 8, -1]
    # shuffles the list in-place
    random.shuffle(ranks)
    return ranks


class Puzzle:
    # Class attributes for search statistics
    def __init__(self, matrix=None, parent=None, goalPuzzle=None):

        """
        Initializes a Puzzle object with the given matrix.
        If the given matrix is invalid, it will raise a ValueError.
        If no matrix is given, it will create a default matrix and randomize it.
        """
        if (matrix is not None):
            self.matrix = matrix

            if (len(self.matrix.shape) != 2):
                raise ValueError("Invalid layout")
            if (self.matrix.shape[0] != 3):
                raise ValueError("Invalid number of rows")
            if (self.matrix.shape[1] != 3):
                raise ValueError("Invalid number of columns")
        else:
            self.matrix = numpy.array([[1, 2, 3],
                                       [4, 5, 6],
                                       [7, 8, 9]])

            self.PuzzleRandomize()

        self.positions = {}
        self.parent = parent
        self.goalPuzzle = goalPuzzle
        self.last_move = None  # last_move  # 'up', 'down', 'left', 'right', or None for initial state

        for i, j in numpy.ndindex(self.matrix.shape):
            self.positions[int(self.matrix[i, j])] = (i, j)  # int to remove the numpy wrapper

        if (parent is not None):
            self.g = 1 + parent.g
        else:
            self.g = 0

        if (goalPuzzle is not None):
            self.h = self.AccumulatedManhattanDistance()
        else:
            self.h = 0

    def __eq__(self, otherPuzzle):
        if not hasattr(otherPuzzle, "matrix"):
            return NotImplemented
        return numpy.array_equal(self.matrix, otherPuzzle.matrix)

    def __lt__(self, otherPuzzle):
        return (self.g + self.h) < (
                    otherPuzzle.g + otherPuzzle.h)

    def __hash__(self):  # for set lookup use plain array of layout as key
        return hash(tuple(int(x) for x in self.matrix.ravel()))

    def Display(self):
        print(self.matrix)

    def PuzzleRandomize(self):
        """
        Randomizes the puzzle by shuffling the ranks and then assigning them to the puzzle in row-major order.
        """
        randomizedRanks = RandomizeRanks()

        index = 0

        for row in range(self.matrix.shape[0]):
            for col in range(self.matrix.shape[1]):
                self.matrix[row, col] = randomizedRanks[index]
                index += 1

    def Exchange(self, newPosition):
        """
        Exchanges the element at newPosition with the element at the current space position.
        If newPosition is out of bounds, it does nothing.
        """
        row, col = newPosition
        spaceRow, spaceCol = self.positions[-1]

        if (row > 2 or row < 0 or col > 2 or col < 0):
            return

        self.positions[-1], self.positions[int(self.matrix[row, col])] = newPosition, self.positions[-1]
        self.matrix[row, col], self.matrix[spaceRow, spaceCol] = self.matrix[spaceRow, spaceCol], self.matrix[row, col]

        if self.goalPuzzle is not None:
            self.h = self.AccumulatedManhattanDistance()

    def IsExpandableToRight(self):
        '''
        Tries to see if the space is expandable to the right
        '''
        _, col = self.positions[-1]
        if col == 2:
            return False
        return True

    def IsExpandableToLeft(self):
        _, col = self.positions[-1]
        if col == 0:
            return False
        return True

    def IsExpandableToUp(self):
        row, _ = self.positions[-1]
        if row == 0:
            return False
        return True

    def IsExpandableToDown(self):
        row, _ = self.positions[-1]
        if row == 2:
            return False
        return True

    def Up(self):
        '''
        This function checks if the position is expandable up and if it is, it exchanges the space with the element above it
        it returns True since it will be used later.
        :return:
        '''
        row, col = self.positions[-1]
        newPosition = ((row - 1, col))
        self.Exchange(newPosition=newPosition)
        self.last_move = 'up'

    def Down(self):
        row, col = self.positions[-1]
        newPosition = ((row + 1, col))
        self.Exchange(newPosition=newPosition)
        self.last_move = 'down'

    def Left(self):
        row, col = self.positions[-1]
        newPosition = ((row, col - 1))
        self.Exchange(newPosition=newPosition)
        self.last_move = 'left'

    def Right(self):
        row, col = self.positions[-1]
        newPosition = ((row, col + 1))
        self.Exchange(newPosition=newPosition)
        self.last_move = 'right'

    def AccumulatedManhattanDistance(self):
        accumulatedManhattanDistance = 0

        for number, coordinates in self.goalPuzzle.positions.items():
            if -1 == number:
               continue

            goalPuzzleRow, goalPuzzleCol = coordinates
            puzzleRow, puzzleColumn = self.positions[number]

            accumulatedManhattanDistance += abs(goalPuzzleRow - puzzleRow) + abs(goalPuzzleCol - puzzleColumn)

        return accumulatedManhattanDistance * HEURISTIC_WEIGHT

    def getSolutionPath(self):
        path = []
        current = self
        while current is not None:
            path.append((current, current.last_move))
            current = current.parent
        return path[::-1]  # Reverse to get start->goal order, backtracking

    def getMoveSequence(self):
        """Returns a list of moves from start to this state"""
        moves = []
        current = self
        while current is not None and current.last_move is not None:
            moves.append(current.last_move)
            current = current.parent
        return moves[::-1]  # Return moves in correct order (start to goal)


def FindSolution(originPuzzle, goalPuzzle):
    global sizeP, sizeQ
    # Reset search statistics
    priorityQueue = []
    exploredPuzzles = set()

    heapq.heappush(priorityQueue, originPuzzle)

    while priorityQueue:
        # Update search statistics
        currentBestPuzzle = heapq.heappop(priorityQueue)
        if currentBestPuzzle == goalPuzzle:
            sizeP = len(priorityQueue)  # P: Current number of open states
            sizeQ = len(exploredPuzzles)
            return currentBestPuzzle.getSolutionPath()

        # Right move
        if currentBestPuzzle.IsExpandableToRight():  # If the move is valid
            rightChild = Puzzle(matrix=currentBestPuzzle.matrix.copy(),
                                parent=currentBestPuzzle,
                                goalPuzzle=goalPuzzle,
                                )
            rightChild.Right()
            if rightChild not in exploredPuzzles:  # If the move is valid and the state has not been explored
                heapq.heappush(priorityQueue, rightChild)

        # Left move
        if currentBestPuzzle.IsExpandableToLeft():
            leftChild = Puzzle(matrix=currentBestPuzzle.matrix.copy(),
                               parent=currentBestPuzzle,
                               goalPuzzle=goalPuzzle,
                               )

            leftChild.Left()
            if leftChild not in exploredPuzzles:
                heapq.heappush(priorityQueue, leftChild)

        # Up move
        if currentBestPuzzle.IsExpandableToUp():
            upChild = Puzzle(matrix=currentBestPuzzle.matrix.copy(),
                             parent=currentBestPuzzle,
                             goalPuzzle=goalPuzzle,
                             )
            upChild.Up()
            if upChild not in exploredPuzzles:
                heapq.heappush(priorityQueue, upChild)

        # Down move
        if currentBestPuzzle.IsExpandableToDown():
            downChild = Puzzle(matrix=currentBestPuzzle.matrix.copy(),
                               parent=currentBestPuzzle,
                               goalPuzzle=goalPuzzle,
                               )
            downChild.Down()
            if downChild not in exploredPuzzles:
                heapq.heappush(priorityQueue, downChild)

        exploredPuzzles.add(currentBestPuzzle)

    sizeP = len(priorityQueue)  # P: Current number of open states
    sizeQ = len(exploredPuzzles)
    return None  # No solution

#numpy.array([[2, 8, 3],
                          #[1, 6, 4],
                          #[7, -1, 5]])

goalLayout = numpy.array([[1, 2, 3],
                          [8, -1, 4],
                          [7, 6, 5]])

# originLayout = numpy.array([[2, 8, 3],
#                             [1, 6, 4],
#                             [7, -1, 5]])  # pag 36

originLayout = numpy.array([[2, 1, 6],
                          [4, -1, 8],
                          [7, 5, 3]])  # pag 93, mismo patrón con el libro cuando se usa un peso de heuristica de 1.5

goalPuzzle = Puzzle(matrix=goalLayout)
originPuzzle = Puzzle(matrix=originLayout, goalPuzzle=goalPuzzle)

# goalPuzzle.Display()
print("-" * 20)
print("Origin Puzzle")
originPuzzle.Display()
print("-" * 20)
print("Goal Puzzle")
goalPuzzle.Display()
print("-" * 20)
print("\n")
print("\n")

solution_path = FindSolution(originPuzzle, goalPuzzle)

if solution_path:
    final_state = solution_path[-1][0]

    # Print search statistics from class attributes
    print(f"\nSearch Statistics:")
    print("-" * 20)
    move_sequence = final_state.getMoveSequence()
    print("Move sequence:", ' -> '.join(move_sequence))
    print("Total moves:", len(move_sequence))
    print("\nDetailed solution:")

    for i, (state, move) in enumerate(solution_path):
        print(f"Step {i}: {'Initial state' if move is None else f'Move: {move}'}")
        state.Display()
        print()
else:
    print("No solution found")

print(f"P (Open states): {sizeP}")
print(f"Q (Expanded states): {sizeQ}")
print('Heuristic Weight: ', HEURISTIC_WEIGHT)

end_time = time()
print("Execution time:", end_time - start_time)
print("-"*20)