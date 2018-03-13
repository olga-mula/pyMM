import numpy as np
import random

from space import Space

class RBconstruction():

	def __init__(self, dictionary):
		self.dictionary = dictionary

class SpaceConstructionRandom(RBconstruction):

	def __init__(self, dictionary, n):
		super().__init__(dictionary)
		self.n = n

	def generateBasis(self):
		# Number of snapshots
		N = len(self.dictionary)
		assert N > 0

		# norm_type
		norm_type = self.dictionary[0].ambient_space.norm_type

		# We use the Fisher-Yates algorithm, which takes O(N) operations
		a = np.arange(N)
		for i in range(N-1, N-self.n, -1):
			j = int(random.random()*i)
			a[i], a[j] = a[j], a[i]
		basis = [self.dictionary[i] for i in a[N-self.n: N]]

		return norm_type, basis

class SpaceConstructionPCA(RBconstruction):

	def __init__(self, dictionary, n):
		super().__init__(dictionary)
		self.n = n

	def generateBasis(self):
		# TODO
		pass