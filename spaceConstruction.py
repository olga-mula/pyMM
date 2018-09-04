import numpy as np
import random

from space import Space

class BasisConstruction():

	def __init__(self, dictionary):
		self.dictionary = dictionary

class BasisConstructionRandom(BasisConstruction):

	def __init__(self, dictionary, n):
		super().__init__(dictionary)
		self.n = n

	def generateBasis(self):
		# Number of snapshots
		N = len(self.dictionary)
		assert N > 0

		# We use the Fisher-Yates algorithm, which takes O(N) operations
		a = np.arange(N)
		for i in range(N-1, N-self.n, -1):
			j = int(random.random()*i)
			a[i], a[j] = a[j], a[i]
		basis = [self.dictionary[i] for i in a[N-self.n: N]]

		return basis

class BasisConstructionPCA(BasisConstruction):

	def __init__(self, dictionary, n):
		super().__init__(dictionary)
		self.n = n

	def generateBasis(self):
		# TODO
		pass

class BasisConstructionGreedy(BasisConstruction):

	def __init__(self, dictionary, n):
		super().__init__(dictionary)
		self.n = n

	def generateBasis(self):
		# Number of snapshots
		N = len(self.dictionary)
		assert N > 0

		# greedy algorithm
		basis = list()
		for i in range(self.n):
			if i == 0:
				index_max = np.argmax([s.norm() for s in self.dictionary])
				basis.append(self.dictionary[index_max])
			else:
				# Build space V_{i-1}
				V = Space(basis)
				err = list()
				for s in self.dictionary:
					diff = s - V.project(s)
					err.append(diff.norm())
				index_max = np.argmax(err)
				basis.append(self.dictionary[index_max])

				print(i, ' ', np.max(err))

		return basis