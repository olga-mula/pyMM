from fenics import *
from functions import Snapshot
# import numpy as np
# from snapshot import Snapshot

class Dictionary():
	def __init__(self, solver, samplingStrategy):
		self.samplingStrategy = samplingStrategy
		self.solver = solver
		self.snapshot_list = self.__generateSnapshots()

	def __generateSnapshots(self):
		self.snapshot_list = list()
		for param in self.samplingStrategy:
			self.snapshot_list.append(Snapshot(self.solver, param))
		self.snapshot_list[0].plot()
