from parameterDomain import ParameterDomain
from sampling import SamplingUniform, SamplingRandom
from solver import DiffusionCheckerboard 
from dictionary import DictionaryFactorySnapshots, DictionaryFactorySensors
from spaceConstruction import SpaceConstructionRandom
from space import Space
from riesz import RieszRadialFunction
from functions import Snapshot
from stateEstimation import StateEstimation

from fenics import *

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

# Define parameter domain
# paramRange = [ (param1_min, param1_max), (param2_min, param_2max), ...]
paramRange = [(1., 2.), (1., 2.), (1., 2.), (1., 2.)]
assert len(paramRange) == n_partition
paramDomainSnapshots  = ParameterDomain(paramRange)

# Sampling strategy (uniform or random)
nSamplesPerParameter = [2, 2, 1, 1]
assert len(nSamplesPerParameter) == n_partition
samplingStrategy = SamplingUniform(paramDomainSnapshots, nSamplesPerParameter)

# Dictionary
dict_factory_snapshots = DictionaryFactorySnapshots(solver, samplingStrategy)
dict_snapshots = dict_factory_snapshots.generateSnapshots()

# Create dictionary of sensors
# =============================

# paramRange = [ (xmin, xmax), (ymin, ymax), (sigma_min, sigma_max) ]
paramRange = [(0.1, 0.9), (0.1, 0.9), (0.5, 1.)]
assert len(paramRange)-1 == spatial_dimension
paramDomainSensors = ParameterDomain(paramRange)

# Sampling strategy (uniform or random)
nSamplesPerParameter = [3, 3, 1]
samplingStrategy = SamplingUniform(paramDomainSensors, nSamplesPerParameter)

# Dictionary
rieszSolver= RieszRadialFunction(solver.ambient_space, fem_degree, solver.mesh)
dict_factory_sensors = DictionaryFactorySensors(rieszSolver, samplingStrategy)
dict_sensors = dict_factory_sensors.generateSensors()

# Create reduced space of dim n 
# We need to specify construction strategy:
# random, PCA (to do), greedy (to do)
# ========================================
n = 2
constructionStrategy = SpaceConstructionRandom(dict_snapshots, n)
Vn = Space(constructionStrategy)

# Create sensing space of dimension m
# We need to specify construction strategy:
# random, greedy (to do)
# =======================================
m = 3
constructionStrategy = SpaceConstructionRandom(dict_sensors, m)
Wm = Space(constructionStrategy)

# State estimation
# ================
u = Snapshot(solver, (1.5, 1.5, 1.5, 1.5))
se = StateEstimation(Vn, Wm)
u_star, v_star, cond = se.measure_and_reconstruct(u, disp_cond=False)

# Plot the exact function and its optimal reconstruction
u.plot()
u_star.plot()






