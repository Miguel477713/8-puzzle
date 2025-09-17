
import numpy, random, os

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
	def __init__(self, matrix=None):
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

		indices = numpy.argwhere(self.matrix == -1)
		self.spacePosition = tuple(indices[0]) # finds the position of the space (-1)

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
		spaceRow, spaceCol = self.spacePosition

		if(row > 2 or row < 0 or col > 2 or col < 0):
			return
		
		self.matrix[row, col], self.matrix[spaceRow, spaceCol] = self.matrix[spaceRow, spaceCol], self.matrix[row, col]
		self.spacePosition = newPosition 

	def Up(self):
		"""
		Moves the space one row up if possible.
		"""
		row, col = self.spacePosition
		newPosition = ((row - 1, col))
		self.Exchange(newPosition = newPosition)

	def Down(self):
		"""
		Moves the space one row down if possible.
		"""
		row, col = self.spacePosition
		newPosition = ((row + 1, col))
		self.Exchange(newPosition = newPosition)

	def Left(self):
		"""
		Moves the space one column to the left if possible.
		"""
		row, col = self.spacePosition
		newPosition = ((row, col - 1))
		self.Exchange(newPosition = newPosition)

	def Right(self):
		"""
		Moves the space one column to the right if possible.
		"""
		row, col = self.spacePosition
		newPosition = ((row, col + 1))
		self.Exchange(newPosition = newPosition)





mic = numpy.array([[1, 2, 3],
				   [4, -1, 5],
				   [6, 7, 8]])

matrix = Puzzle(mic)
matrix.Up()
matrix.Up()
matrix.Left()
matrix.Display()
matrix.Right()
matrix.Right()
matrix.Right()
matrix.Right()
matrix.Down()
matrix.Display()
