from dolfin import *
from expression import RadialFunctionExpression

class RieszRadialFunction():
	def __init__(self, ambient_space, fem_degree, mesh):
		self.ambient_space = ambient_space
		self.fem_degree = fem_degree
		self.mesh = mesh

	def compute_solution(self, param):
		norm_type = self.ambient_space.norm_type
		V = FunctionSpace(self.mesh, "P", self.fem_degree)
		assert len(param) > 1  # We need at least one location in 1d and sigma
		pos = param[:-1]
		sigma = param[-1]

		if norm_type == "L2":
			f = RadialFunctionExpression(param=param, degree=self.fem_degree+2)
			u = interpolate(f, V)
			# We normalize function
			c = interpolate( Constant(norm(u, norm_type=norm_type)), V)
			u.vector()[:] = u.vector().get_local() / c.vector().get_local()
			return u
		elif norm_type == "H10":
			u = TrialFunction(V)
			v = TestFunction(V)

			# Source term and boundary condition
			u_bc = Constant(0.)
			f = RadialFunctionExpression(param=param, degree=self.fem_degree+2)

			def boundary(x, on_boundary):
			    return on_boundary
			bc = DirichletBC(V, u_bc, boundary)

			# Define variational form
			a = inner(grad(u), grad(v))*dx
			L = f*v*dx

			# Solve problem
			u = Function(V)
			solve(a == L, u, bc)

			# We normalize function
			c = interpolate( Constant(norm(u, norm_type=norm_type)), V)
			u.vector()[:] = u.vector().get_local() / c.vector().get_local()

			return u
		else:
			print('Unknown norm_type in Riesz compute_solution')
			exit(-1)

	def plot(self, u, seeOnScreen= True, save=True, seeMesh = False):
		# Visualize plot and mesh
		if seeOnScreen:
			plot(u)
			if seeMesh:
				plot(u.function_space().self.mesh())
				legend()
			import matplotlib.pyplot as plt
			plt.show()
		# Save snapshot to file in VTK format
		if save:
			filename = 'img/riesz_representer.pvd'
			vtkfile = File(filename)
			vtkfile << u


