from fenics import *
from functions import Snapshot, Sensor

class DictionaryFactory():
	def __init__(self, problem, samplingStrategy, typeFactory='snapshots'):
		self.problem = problem
		self.samplingStrategy = samplingStrategy
		self.typeFactory = typeFactory	

	def create(self):
		dictionary = list()

		if self.typeFactory == 'snapshots':
			for param in self.samplingStrategy:			
				u = self.problem.compute_solution(param)
				dictionary.append(Snapshot(self.problem.ambientSpace(), u, param))
		elif self.typeFactory == 'sensors':
			for param in self.samplingStrategy:	
				w = self.problem.compute_riesz_sensor(param)
				dictionary.append(Sensor(self.problem.ambientSpace(), w, param))
		else:
			raise ValueError('Value of typeFactory not supported.')
			
		return dictionary
