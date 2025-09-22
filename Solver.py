
import numpy, random, os, copy, heapq

if os.name == 'nt':
	os.system('cls')
elif os.name == 'posix':
	os.system('clear')


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
	def __init__(self, matrix=None, parent=None, goalPuzzle=None):
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
		return (self.g + self.h) < (otherPuzzle.g + otherPuzzle.h)

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
		"""
		Moves the space one row up if possible.
		"""
		row, col = self.positions[-1]
		newPosition = ((row - 1, col))
		self.Exchange(newPosition = newPosition)

	def Down(self):
		"""
		Moves the space one row down if possible.
		"""
		row, col = self.positions[-1]
		newPosition = ((row + 1, col))
		self.Exchange(newPosition = newPosition)

	def Left(self):
		"""
		Moves the space one column to the left if possible.
		"""
		row, col = self.positions[-1]
		newPosition = ((row, col - 1))
		self.Exchange(newPosition = newPosition)

	def Right(self):
		"""
		Moves the space one column to the right if possible.
		"""
		row, col = self.positions[-1]
		newPosition = ((row, col + 1))
		self.Exchange(newPosition = newPosition)

	def AccumulatedManhattanDistance(self):
		accumulatedManhattanDistance = 0

		for number, coordinates in self.goalPuzzle.positions.items():
			goalPuzzleRow, goalPuzzleCol = coordinates
			puzzleRow, puzzleColumn = self.positions[number]

			accumulatedManhattanDistance += abs(goalPuzzleRow - puzzleRow) + abs(goalPuzzleCol - puzzleColumn)
		
		return accumulatedManhattanDistance

def FindSolution(originPuzzle, goalPuzzle):
	priorityQueue = []
	exploredPuzzles = set()

	heapq.heappush(priorityQueue, originPuzzle)

	while priorityQueue: #While not empty
		#Get best f(v) evaluation
		currentBestPuzzle = heapq.heappop(priorityQueue)

		if(currentBestPuzzle == goalPuzzle):
			return currentBestPuzzle

		if currentBestPuzzle.IsExpandableToRight():
			rightChild = Puzzle(matrix=currentBestPuzzle.matrix.copy(), parent=currentBestPuzzle, goalPuzzle=goalPuzzle)
			rightChild.Right()

			if(rightChild not in exploredPuzzles):
				heapq.heappush(priorityQueue, rightChild)

		if currentBestPuzzle.IsExpandableToLeft():
			leftChild = Puzzle(matrix=currentBestPuzzle.matrix.copy(), parent=currentBestPuzzle, goalPuzzle=goalPuzzle)
			leftChild.Left()

			if(leftChild not in exploredPuzzles):
				heapq.heappush(priorityQueue, leftChild)

		if currentBestPuzzle.IsExpandableToUp():
			upChild = Puzzle(matrix=currentBestPuzzle.matrix.copy(), parent=currentBestPuzzle, goalPuzzle=goalPuzzle)
			upChild.Up()

			if(upChild not in exploredPuzzles):
				heapq.heappush(priorityQueue, upChild)

		if currentBestPuzzle.IsExpandableToDown():
			downChild = Puzzle(matrix=currentBestPuzzle.matrix.copy(), parent=currentBestPuzzle, goalPuzzle=goalPuzzle)
			downChild.Down()

			if(downChild not in exploredPuzzles):
				heapq.heappush(priorityQueue, downChild)
				
		exploredPuzzles.add(currentBestPuzzle)

	if len(priorityQueue) == 0:
		return None #No solution



goalLayout = numpy.array([[1, 2, 3],
				   		  [8, -1, 4],
				   		  [7, 6, 5]])

originLayout = numpy.array([[2, 1, 6],
				   			[4, -1, 8],
				  	 		[7, 5, 3]])
							#pag 93

goalPuzzle = Puzzle(matrix=goalLayout)
originPuzzle = Puzzle(matrix=originLayout, goalPuzzle=goalPuzzle)


#goalPuzzle.Display()
goalPuzzle.Display()
print("\n")
solution = FindSolution(originPuzzle = originPuzzle, goalPuzzle = goalPuzzle)
if solution is not None:
	solution.Display()
else:
	print("No solution")