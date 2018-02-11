from parameterDomain import ParameterDomain
from sampling import SamplingStrategy, SamplingUniform, SamplingRandom
from solver import DiffusionCheckerboard 
from dictionary import Dictionary
from rbConstruction import RBconstructionRandom
from basis import ReducedBasis

# Define solver
fem_degree = 1
spatial_dofs_per_direction = [100, 100]
spatial_dimension = len(spatial_dofs_per_direction)
diffusion_coef_partition_level = 1
n_partition = (1+diffusion_coef_partition_level)**spatial_dimension

solver = DiffusionCheckerboard(fem_degree, spatial_dimension, spatial_dofs_per_direction, diffusion_coef_partition_level)

# Define parameter domain
parameterDomain = [(1., 10.), (2., 3.), (0.1, 7.), (1., 8.)]
if len(parameterDomain) != n_partition:
	print("Error: parameter domain must have same length as n_partition")
	exit(1)
nSamplesPerParameter = [2, 2, 1, 1]
if len(parameterDomain) != n_partition:
	print("Error: nSamplesPerParameter must have same length as parameterDomain")
	exit(1)
parameterDomain  = ParameterDomain(parameterDomain)
samplingStrategy = SamplingUniform(parameterDomain, nSamplesPerParameter)
# samplingStrategy = SamplingRandom(parameterDomain, 4)

# Define dictionary of snapshots
dictionary = Dictionary(solver, samplingStrategy)

# Define reduced basis
n = 3
rbConstructionStrategy = RBconstructionRandom(dictionary, n)
Vn = ReducedBasis(rbConstructionStrategy)





