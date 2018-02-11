from fenics import Function
from solver import Solver, DiffusionCheckerboard
import copy

class FE_function():
	# Remark: We assume that self and other have the same function_space

	def __init__(self, function):
		self.fun = function  # function should be of type Function from dolfin

	def __add__(self, other):
		V = self.fun.function_space()
		u = Function(V)
		u.vector()[:] = self.fun.vector().get_local() + other.fun.vector().get_local()
		return FE_function(u)

	def __iadd__(self, other):
		self.fun.vector()[:] = self.fun.vector().get_local() + other.fun.vector().get_local()
		return self

	def __sub__(self, other):
		V = self.fun.function_space()
		u = Function(V)
		u.vector()[:] = self.fun.vector().get_local() - other.fun.vector().get_local()
		return FE_function(u)

	def __isub__(self, other):
		self.fun.vector()[:] = self.fun.vector().get_local() - other.fun.vector().get_local()
		return self

	def plot(self, u, seeOnScreen= True, save=True, seeMesh = False):
		# Visualize plot and mesh
		if seeOnScreen:
			plot(u)
			if seeMesh:
				plot(u.function_space().self.mesh())
			import matplotlib.pyplot as plt
			plt.show()
		# Save snapshot to file in VTK format
		if save:
			filename = 'img/fe_function.pvd'
			vtkfile = File(filename)
			vtkfile << u



class Snapshot(FE_function):
	def __init__(self, solver, param):
		self.solver = solver # Not sure if we need this
		self.param = param
		self.fun = solver.compute_solution(param)
		self.measure=0 # TODO: measures of solution

	def plot(self):
		return self.solver.plot(self.fun)