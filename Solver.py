
import numpy, random, os, copy, heapq
from time import time

if os.name == 'nt':
	os.system('cls')
elif os.name == 'posix':
	os.system('clear')

start_time = time()
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
	open_states = 0
	expanded_states = 0

	def __init__(self, matrix=None, parent=None, goalPuzzle=None, last_move=None, heuristic_weight=1.0):
		"""
		Initializes a Puzzle object with the given matrix.
		If the given matrix is invalid, it will raise a ValueError.
		If no matrix is given, it will create a default matrix and randomize it.
		"""
		if(matrix is not None):
			self.matrix = matrix

			if(len(self.matrix.shape) != 2):
				raise ValueError("Invalid layout")
			if(self.matrix.shape[0] != 3):
				raise ValueError("Invalid number of rows")
			if(self.matrix.shape[1] != 3):
				raise ValueError("Invalid number of columns")		
		else:
			self.matrix = numpy.array([[1, 2, 3],
                   					   [4, 5, 6],
                   					   [7, 8, 9]])
			
			self.PuzzleRandomize()

		self.positions = {}
		self.parent = parent
		self.goalPuzzle = goalPuzzle
		self.heuristic_weight = heuristic_weight

		self.last_move = last_move  # 'up', 'down', 'left', 'right', or None for initial state

		for i, j in numpy.ndindex(self.matrix.shape):
			self.positions[int(self.matrix[i, j])] = (i, j)  #int to remove the numpy wrapper

		if(parent is not None):
			self.g = 1 + parent.g
		else:
			self.g = 0

		if(goalPuzzle is not None):
			self.h = self.AccumulatedManhattanDistance()
		else:
			self.h = 0
	
	def __eq__(self, otherPuzzle):
		if not hasattr(otherPuzzle, "matrix"):
			return NotImplemented
		return numpy.array_equal(self.matrix, otherPuzzle.matrix)

	def __lt__(self, otherPuzzle):
		return (self.g + self.heuristic_weight * self.h) < (otherPuzzle.g + otherPuzzle.heuristic_weight * otherPuzzle.h)

	def __hash__(self): #for set lookup use plain array of layout as key
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
				index +=1

	def Exchange(self, newPosition):
		"""
		Exchanges the element at newPosition with the element at the current space position.
		If newPosition is out of bounds, it does nothing.
		"""
		row, col = newPosition
		spaceRow, spaceCol = self.positions[-1]

		if(row > 2 or row < 0 or col > 2 or col < 0):
			return

		self.positions[-1], self.positions[int(self.matrix[row, col])] = newPosition, self.positions[-1]
		self.matrix[row, col], self.matrix[spaceRow, spaceCol] = self.matrix[spaceRow, spaceCol], self.matrix[row, col]

		if self.goalPuzzle is not None:
			self.h = self.AccumulatedManhattanDistance()
	
	def IsExpandableToRight(self):
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
		if self.IsExpandableToUp():
			row, col = self.positions[-1]
			newPosition = ((row - 1, col))
			self.Exchange(newPosition=newPosition)
			self.last_move = 'up'
			return True
		return False

	def Down(self):
		if self.IsExpandableToDown():
			row, col = self.positions[-1]
			newPosition = ((row + 1, col))
			self.Exchange(newPosition=newPosition)
			self.last_move = 'down'
			return True
		return False

	def Left(self):
		if self.IsExpandableToLeft():
			row, col = self.positions[-1]
			newPosition = ((row, col - 1))
			self.Exchange(newPosition=newPosition)
			self.last_move = 'left'
			return True
		return False

	def Right(self):
		if self.IsExpandableToRight():
			row, col = self.positions[-1]
			newPosition = ((row, col + 1))
			self.Exchange(newPosition=newPosition)
			self.last_move = 'right'
			return True
		return False

	def AccumulatedManhattanDistance(self):
		accumulatedManhattanDistance = 0

		for number, coordinates in self.goalPuzzle.positions.items():
			goalPuzzleRow, goalPuzzleCol = coordinates
			puzzleRow, puzzleColumn = self.positions[number]

			accumulatedManhattanDistance += abs(goalPuzzleRow - puzzleRow) + abs(goalPuzzleCol - puzzleColumn)
		
		return accumulatedManhattanDistance

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

def FindSolution(originPuzzle, goalPuzzle, heuristic_weight=1.0):
    # Reset search statistics
    Puzzle.open_states = 0
    Puzzle.expanded_states = 0
    
    priorityQueue = []
    exploredPuzzles = set()
    
    heapq.heappush(priorityQueue, originPuzzle)

    while priorityQueue:
        # Update search statistics
        Puzzle.open_states = len(priorityQueue)  # P: Current number of open states
        currentBestPuzzle = heapq.heappop(priorityQueue)
        Puzzle.expanded_states += 1  # Q: Increment expanded states counter

        if currentBestPuzzle == goalPuzzle:
            return currentBestPuzzle.getSolutionPath()

        # Right move
        if currentBestPuzzle.IsExpandableToRight():
            rightChild = Puzzle(matrix=currentBestPuzzle.matrix.copy(),
                              parent=currentBestPuzzle,
                              goalPuzzle=goalPuzzle,
                              last_move='right',
							  heuristic_weight=heuristic_weight)
            if rightChild.Right() and rightChild not in exploredPuzzles:
                heapq.heappush(priorityQueue, rightChild)

        # Left move
        if currentBestPuzzle.IsExpandableToLeft():
            leftChild = Puzzle(matrix=currentBestPuzzle.matrix.copy(),
                             parent=currentBestPuzzle,
                             goalPuzzle=goalPuzzle,
                             last_move='left',
							heuristic_weight=heuristic_weight)
            if leftChild.Left() and leftChild not in exploredPuzzles:
                heapq.heappush(priorityQueue, leftChild)

        # Up move
        if currentBestPuzzle.IsExpandableToUp():
            upChild = Puzzle(matrix=currentBestPuzzle.matrix.copy(),
                           parent=currentBestPuzzle,
                           goalPuzzle=goalPuzzle,
                           last_move='up',
						   heuristic_weight=heuristic_weight)
            if upChild.Up() and upChild not in exploredPuzzles:
                heapq.heappush(priorityQueue, upChild)

        # Down move
        if currentBestPuzzle.IsExpandableToDown():
            downChild = Puzzle(matrix=currentBestPuzzle.matrix.copy(),
                             parent=currentBestPuzzle,
                             goalPuzzle=goalPuzzle,
                             last_move='down',
							 heuristic_weight=heuristic_weight)
            if downChild.Down() and downChild not in exploredPuzzles:
                heapq.heappush(priorityQueue, downChild)

        exploredPuzzles.add(currentBestPuzzle)

    return None  # No solution


goalLayout = numpy.array([[1, 2, 3],
				   		  [8, -1, 4],
				   		  [7, 6, 5]])

originLayout = numpy.array([[2, 1, 6],
				   			[4, -1, 8],
				  	 		[7, 5, 3]]) # pag 93


goalPuzzle = Puzzle(matrix=goalLayout)
originPuzzle = Puzzle(matrix=originLayout, goalPuzzle=goalPuzzle)

#goalPuzzle.Display()
print("-"*20)
print("Origin Puzzle")
originPuzzle.Display()
print("-"*20)
print("Goal Puzzle")
goalPuzzle.Display()
print("-"*20)
print("\n")
print("\n")

solution_path = FindSolution(originPuzzle, goalPuzzle, heuristic_weight=1.5)  # Example with weight 1.5

if solution_path:
    final_state = solution_path[-1][0]
    
    # Print search statistics from class attributes
    print(f"\nSearch Statistics:")
    print(f"P (Open states): {Puzzle.open_states}")
    print(f"Q (Expanded states): {Puzzle.expanded_states}")
    print("-" * 20)
    move_sequence = final_state.getMoveSequence()
    print("Move sequence:", '-> '.join(move_sequence))
    print("Total moves:", len(move_sequence))
    print("\nDetailed solution:")
    print('Heuristic Weight: ', final_state.heuristic_weight)

    
    for i, (state, move) in enumerate(solution_path):
        print(f"Step {i}: {'Initial state' if move is None else f'Move: {move}'}")
        state.Display()
        print()
else:
    print("No solution found")

end_time = time()
print("Execution time:", end_time - start_time)
print("-" * 20)
