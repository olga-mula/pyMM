from fenics import *
from functions import Snapshot, Sensor

class DictionaryFactory():
	def __init__(self, samplingStrategy):
		self.samplingStrategy = samplingStrategy

class DictionaryFactorySnapshots(DictionaryFactory):
	def __init__(self, solver, samplingStrategy):
		super().__init__(samplingStrategy)
		self.solver = solver

	def generateSnapshots(self):
		snapshot_list = list()
		for param in self.samplingStrategy:
			snapshot_list.append(Snapshot(self.solver, param))
		return snapshot_list

class DictionaryFactorySensors(DictionaryFactory):
	def __init__(self, solver, samplingStrategy):
		super().__init__(samplingStrategy)
		self.solver = solver

	def generateSensors(self):
		sensor_list = list()
		for param in self.samplingStrategy:
			sensor_list.append(Sensor(self.solver, param))
		return sensor_list
