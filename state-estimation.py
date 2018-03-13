from parameterDomain import ParameterDomain
from sampling import SamplingUniform, SamplingRandom
from solver import DiffusionCheckerboard 
from dictionary import DictionaryFactorySnapshots, DictionaryFactorySensors
from rbConstruction import SpaceConstructionRandom
from space import HilbertSpace, Space
from riesz import RieszRadialFunction

# Define solver. (Here we use the diffusion checkerboard problem)
# ==============
fem_degree = 1
spatial_dofs_per_direction = [100, 100]
spatial_dimension = len(spatial_dofs_per_direction)
diffusion_coef_partition_level = 1
n_partition = (1+diffusion_coef_partition_level)**spatial_dimension

solver = DiffusionCheckerboard(fem_degree, spatial_dimension, spatial_dofs_per_direction, diffusion_coef_partition_level)

# Create dictionary of snapshots
# ==============================
# Define parameter domain and sampling strategy
# paramDomainSnapshots = [ (D1min, D1max), (D2min, D2max), ...]
paramDomainSnapshots = [(1., 10.), (2., 3.), (0.1, 7.), (1., 8.)]
nSamplesPerParameter = [2, 2, 1, 1]
assert len(paramDomainSnapshots) == n_partition
assert len(nSamplesPerParameter) == n_partition

paramDomainSnapshots  = ParameterDomain(paramDomainSnapshots)
samplingStrategy = SamplingUniform(paramDomainSnapshots, nSamplesPerParameter)

dict_factory_snapshots = DictionaryFactorySnapshots(solver, samplingStrategy)
dict_snapshots = dict_factory_snapshots.generateSnapshots()

# Create dictionary of sensors
# =============================
# paramDomainSensors = [ (xmin, xmax), (ymin, ymax), (sigma_min, sigma_max) ]
paramDomainSensors = [(0.1, 0.9), (0.1, 0.9), (0.05, 1.)]
assert len(paramDomainSensors)-1 == spatial_dimension
nSamplesPerParameter = [2, 2, 1]
paramDomainSensors = ParameterDomain(paramDomainSensors)
samplingStrategy = SamplingUniform(paramDomainSensors, nSamplesPerParameter)

rieszSolver = RieszRadialFunction(solver.ambient_space, fem_degree, solver.mesh)
dict_factory_sensors = DictionaryFactorySensors(rieszSolver, samplingStrategy)
dict_sensors = dict_factory_sensors.generateSensors()

# Create reduced space of dimension n
# ===================================
n = 2
constructionStrategy = SpaceConstructionRandom(dict_snapshots, n)
V = Space(constructionStrategy)

# Create sensing space of dimension m
# ===================================
m = 3
constructionStrategy = SpaceConstructionRandom(dict_sensors, m)
W = Space(constructionStrategy)







