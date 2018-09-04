from dolfin import *
import numpy as np
import itertools

from riesz import RieszRadialFunction
from space import HilbertSpace

# Class for the Hypercube mesh
def UnitHyperCube(divisions):
	mesh_classes = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
	d = len(divisions)
	mesh = mesh_classes[d - 1](*divisions)
	return mesh

class DiffusionCheckerboard():

	"""
		Poisson equation with Dirichlet BC.
		Domain: Unit hypercube (any dimension >= 1 supported)

		  div( kappa grad(u) ) = f    in the unit hypercube
		            		 u = u_D  on the boundary

		  u_D = 0
		  f = 1

		  kappa:
		  	- Piecewise constant on a partition of level l in {0,1,...}
		  	- Level l: each coordinate is divided into 2^l segments.
		  	- The value in each subdomain ranges in [1,10] and is generated randomly
		"""

	def __init__(self, fem_degree=1, dim=2, nx=[100, 100], l=1):
		self.fem_degree = fem_degree # FEM degree
		self.dim  = dim              # Spatial dimension
		self.nx   = nx     # Spatial dofs per direction: [dofs_x, dofs_y, ...]
		self.l    = l      # Partition level for diffusion coefficient
		self.n_param = (1+self.l)**self.dim # Number of parameters
		self.mesh = UnitHyperCube(self.nx)

	@staticmethod
	def problemType():
		return 'DiffusionCheckerboard'

	@staticmethod
	def ambientSpace():
		return HilbertSpace('H10')

	@staticmethod
	def get_info():
		return 'Problem type: ' + DiffusionCheckerboard.problemType() + '\n'

	# Parameters (diffusion coefficient)
	def paramRange(self):
		# Returns [ (param1_min, param1_max), (param2_min, param_2max), ...]
		interval = (1.,10.)
		return list(itertools.repeat(interval, times=self.n_param)) # [(1., 10.), ...]

	def nSamplesPerParameter(self):
		n = 2
		return n * np.ones(self.n_param)

	# Sensors
	def n_sensors_per_direction(self):
		n = 10
		return n * np.ones(self.dim) 

	def position_range_sensor(self):
		"""
			Returns [ (x0_min, x0_max), (x1_min, x1_max), ...]
		"""
		interval = (0.1,0.9)
		return list(itertools.repeat(interval, times=self.dim))

	def n_sigma_sensors(self):
		return 1

	def sigma_range_sensor(self):
		"""
			Returns (sigma_min, sigma_max)
		"""
		return (0.05, 0.1)

	# Solver
	def compute_solution(self, param):
		"""
			Compute solution for certain parameters for the diffusion coefficient
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
		L = self.l
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

	# Riesz representation of the sensors
	def compute_riesz_sensor(self, param_sensor):
		"""
			For the function
				phi_{x, sigma}(y) = C exp( -|| y - xi ||/2 sigma^2 )
			we define
				param_sensor = [ x, sigma ]
		"""

		rieszproblem = RieszRadialFunction(DiffusionCheckerboard.ambientSpace(), self.fem_degree, self.mesh)

		return rieszproblem.compute_solution(param_sensor)

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