from parameterDomain import ParameterDomain
from sampling import SamplingStrategy, SamplingUniform, SamplingRandom
from solver import DiffusionCheckerboard 
from dictionary import Dictionary
from rbConstruction import RBconstructionRandom
from space import ReducedSpace

# Define solver. Here we use the diffusion checkerboard problem
fem_degree = 1
spatial_dofs_per_direction = [100, 100]
spatial_dimension = len(spatial_dofs_per_direction)
diffusion_coef_partition_level = 1
n_partition = (1+diffusion_coef_partition_level)**spatial_dimension

solver = DiffusionCheckerboard(fem_degree, spatial_dimension, spatial_dofs_per_direction, diffusion_coef_partition_level)

# Define parameter domain
parameterDomain = [(1., 10.), (2., 3.), (0.1, 7.), (1., 8.)]
assert len(parameterDomain) == n_partition

nSamplesPerParameter = [3, 3, 1, 1]
assert nSamplesPerParameter != n_partition

parameterDomain  = ParameterDomain(parameterDomain)
samplingStrategy = SamplingUniform(parameterDomain, nSamplesPerParameter)
# samplingStrategy = SamplingRandom(parameterDomain, 4)

# Define dictionary of snapshots
dictionary = Dictionary(solver, samplingStrategy)

# Define reduced basis
n = 1
rbConstructionStrategy = RBconstructionRandom(dictionary, n)
Vn = ReducedSpace(rbConstructionStrategy)

Vn.project(dictionary.snapshot_list[0] * 2.)





