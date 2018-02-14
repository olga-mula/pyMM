import numpy as np
import random

from space import ReducedSpace

class RBconstruction():

	def __init__(self, dictionary):
		self.dictionary = dictionary

class RBconstructionRandom(RBconstruction):

	def __init__(self, dictionary, n):
		super().__init__(dictionary)
		self.n = n

	def generateBasis(self):
		# norm_type
		norm_type = self.dictionary.solver.ambient_space.norm_type

		# basis functions
		N = len(self.dictionary.snapshot_list)
		a = np.arange(N)
		# We use the Fisher-Yates algorithm, which takes O(N) operations
		for i in range(N-1, N-self.n, -1):
			j = int(random.random()*i)
			a[i], a[j] = a[j], a[i]
		basis = [self.dictionary.snapshot_list[i] for i in a[N-self.n: N]]

		return norm_type, basis

class RBconstructionPCA(RBconstruction):

	def __init__(self, dictionary, n):
		super().__init__(dictionary)
		self.n = n

	def generateBasis(self):
		# TODO
		pass