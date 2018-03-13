from dolfin import *
import numpy as np
import itertools

from space import HilbertSpace

# Class for the Hypercube mesh
def UnitHyperCube(divisions):
	mesh_classes = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
	d = len(divisions)
	mesh = mesh_classes[d - 1](*divisions)
	return mesh

class Solver():
	def __init__(self, ambient_space, fem_degree, spatial_dimension, spatial_dofs_per_direction):
		self.ambient_space = ambient_space
		self.fem_degree = fem_degree
		self.spatial_dimension = spatial_dimension
		self.spatial_dofs_per_direction = spatial_dofs_per_direction   # [ dofs_x, dofs_y, ...]

class DiffusionCheckerboard(Solver):

	def __init__(self, fem_degree, spatial_dimension, spatial_dofs_per_direction, diffusion_coef_partition_level):
		super().__init__(HilbertSpace('H10'), fem_degree, spatial_dimension, spatial_dofs_per_direction)
		self.diffusion_coef_partition_level = diffusion_coef_partition_level
		self.mesh = UnitHyperCube(self.spatial_dofs_per_direction)

	def compute_solution(self, param):
		"""
		Poisson equation with Dirichlet conditions.
		Domain: Unit hypercube (any dimension >= 1 supported)

		  \div( \kappa \grad(u) ) = f    in the unit hypercube
		            u = u_D              on the boundary

		  u_D = 0
		  f = 1

		  kappa:
		  	- Piecewise constant on a partition of level l.
		  	- Level l: each coordinate is divided into 2^l segments.
		  	- The value in each subdomain ranges in [1,10] and is generated randomly
		"""

		# Subdomain class
		class Omega(SubDomain):
			def __init__(self, x_min, x_max):
				super().__init__()
				self.x_min = x_min
				self.x_max = x_max
			def inside(self, x, on_boundary):
				b = [ y >= y_min - DOLFIN_EPS and y <= y_max for (y, y_min, y_max) in zip(x, self.x_min, self.x_max)]
				return all(b)

		
		# Initialize mesh function for interior sub-domains
		dim = self.mesh.topology().dim()
		domains = MeshFunction("size_t", self.mesh, dim)
		domains.set_all(0)

		# kappa and sub-domain instances
		L = self.diffusion_coef_partition_level
		level_index_list = \
			list(itertools.product(list(range(L+1)), repeat=dim)) 
		omega = list() # subdomains
		kappa = list()
		for i, level in enumerate(level_index_list):
			# x_min, x_max for each coordinate 
			x_min = [ float(l)/(L+1) for l in level ] 
			x_max = [ float(l+1)/(L+1) for l in level ]
			omega.append( Omega(x_min, x_max) )
			omega[i].mark(domains, i)
			kappa.append(Constant(param[i]))

		# Source term and boundary condition
		u_bc = Constant(0.)
		f   = Constant(1.0)

		# Define function space and basis functions
		V = FunctionSpace(self.mesh, "P", self.fem_degree)
		u = TrialFunction(V)
		v = TestFunction(V)

		def boundary(x, on_boundary):
		    return on_boundary
		bc = DirichletBC(V, u_bc, boundary)

		# Define measures associated with the interior sub-domains
		dx = Measure('dx', domain=self.mesh, subdomain_data=domains)

		# Define variational form
		a = sum([inner(kappa[i]*grad(u), grad(v))*dx(i) for i in list(range(len(kappa)))])
		L = sum([f*v*dx(i) for i in list(range(len(kappa)))])

		# Solve problem
		u = Function(V)
		solve(a == L, u, bc)

		return u


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
			filename = 'img/solution_diffusion_checkerboard.pvd'
			vtkfile = File(filename)
			vtkfile << u