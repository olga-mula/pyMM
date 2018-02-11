from solver import Solver, DiffusionCheckerboard

class FE_function():
	def __init__(self):
		pass

class Snapshot(FE_function):
	def __init__(self, solver, param):
		self.solver = solver # Not sure if we need this
		self.param = param
		self.fun = solver.compute_solution(param)
		self.measure=0 # TODO: measures of solution

	def plot(self):
		return self.solver.plot(self.fun)