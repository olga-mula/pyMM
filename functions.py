from fenics import *
from solver import Solver, DiffusionCheckerboard
import copy

class FE_function():
	# Remark: We assume that self and other have the same ambient_space and fem function_space

	def __init__(self, ambient_space, function):
		self.ambient_space = ambient_space
		self.fun = function  # function should be of type Function from dolfin

	# Addition of two FE_functions
	def __add__(self, other):
		V = self.fun.function_space()
		u = Function(V)
		u.vector()[:] = self.fun.vector().get_local() + other.fun.vector().get_local()
		return FE_function(self.ambient_space, u)

	def __iadd__(self, other):
		self.fun.vector()[:] = self.fun.vector().get_local() + other.fun.vector().get_local()
		return self

	# Subtraction of two FE_functions
	def __sub__(self, other):
		V = self.fun.function_space()
		u = Function(V)
		u.vector()[:] = self.fun.vector().get_local() - other.fun.vector().get_local()
		return FE_function(self.ambient_space, u)

	def __isub__(self, other):
		self.fun.vector()[:] = self.fun.vector().get_local() - other.fun.vector().get_local()
		return self

	# Multiplication by scalar
	def __mul__(self, other):
		""" other must be a scalar here """
		V = self.fun.function_space()
		u = Function(V)
		c = interpolate( Constant(other), V )
		u.vector()[:] = self.fun.vector().get_local() * c.vector().get_local()
		return FE_function(self.ambient_space, u)

	def __imul__(self, other):
		""" other must be a scalar here """
		V = self.fun.function_space()
		c = interpolate( Constant(other), V )
		self.fun.vector()[:] = self.fun.vector().get_local() * c.vector().get_local()
		return self

	# Division by scalar
	def __truediv__(self, other):
		""" other must be a scalar here """
		V = self.fun.function_space()
		u = Function(V)
		c = interpolate( Constant(other), V )
		u.vector()[:] = self.fun.vector().get_local() / c.vector().get_local()
		return FE_function(self.ambient_space, u)

	def __itruediv__(self, other):
		""" other must be a scalar here """
		V = self.fun.function_space()
		c = interpolate( Constant(other), V )
		self.fun.vector()[:] = self.fun.vector().get_local() / c.vector().get_local()
		return self

	# Inner product
	def dot(self, other):
		return self.ambient_space.inner_product(self.fun, other.fun)

	def norm(self):
		return norm(self.fun, norm_type=self.ambient_space.norm_type)

	def plot(self, seeOnScreen= True, save=True, seeMesh = False):
		# Visualize plot and mesh
		if seeOnScreen:
			plot(self.fun)
			if seeMesh:
				plot(self.fun.function_space().self.mesh())
			import matplotlib.pyplot as plt
			plt.show()
		# Save snapshot to file in VTK format
		if save:
			filename = 'img/fe_function.pvd'
			vtkfile = File(filename)
			vtkfile << self.fun

class Snapshot(FE_function):
	def __init__(self, solver, param):
		super().__init__(solver.ambient_space, solver.compute_solution(param))
		self.solver = solver # Not sure if we need this
		self.param = param
		self.measure=0 # TODO: measures of solution

	def plot(self):
		return self.solver.plot(self.fun)