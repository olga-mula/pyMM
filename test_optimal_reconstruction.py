import numpy as np
import os
import sys
import argparse
from fenics import *

from parameterDomain import ParameterDomain
from sampling import SamplingUniform, SamplingRandom
from solver import DiffusionCheckerboard 
from dictionary import DictionaryFactory
from spaceConstruction import BasisConstructionRandom, BasisConstructionGreedy
from space import Space
from functions import Snapshot
from stateEstimation import StateEstimation

def save_offline_phase(directoryOffline, dict_snapshots, dict_sensors):
	# Save snapshots
	hdf5file = HDF5File(MPI.comm_world, directoryOffline+"snapshots.hdf5", 'w')
	param = list()
	for i, s in enumerate(dict_snapshots):
		hdf5file.write(s.fun, "snapshot"+str(i))
		param.append(s.param)
	np.savetxt(directoryOffline+'param-snapshots.txt', np.asarray(param))
	del hdf5file

	# Save sensors
	hdf5file = HDF5File(MPI.comm_world, directoryOffline+"sensors.hdf5", 'w')
	param = list()
	for i, s in enumerate(dict_sensors):
		hdf5file.write(s.fun, "sensor"+str(i))
		param.append(s.param)
	np.savetxt(directoryOffline+'param-sensors.txt', np.asarray(param))
	del hdf5file

def load_offline_phase(directoryOffline, problem):
	# Load snapshots
	param = np.loadtxt(directoryOffline+'param-snapshots.txt')
	n_snapshots = len(param)

	hdf5file = HDF5File(MPI.comm_world, directoryOffline+"snapshots.hdf5", 'r')
	V = FunctionSpace(problem.mesh, "P", problem.fem_degree)
	dict_snapshots = list()
	for i in range(n_snapshots):
		u = Function(V)
		hdf5file.read(u, "snapshot"+str(i))
		dict_snapshots.append(Snapshot(problem.ambientSpace(), u, param[i]))
		del u
	del hdf5file
	del param

	# Load sensors
	param = np.loadtxt(directoryOffline+'param-sensors.txt')
	n_snapshots = len(param)

	hdf5file = HDF5File(MPI.comm_world, directoryOffline+"sensors.hdf5", 'r')
	V = FunctionSpace(problem.mesh, "P", problem.fem_degree)
	dict_sensors = list()
	for i in range(n_snapshots):
		u = Function(V)
		hdf5file.read(u, "sensor"+str(i))
		dict_sensors.append(Snapshot(problem.ambientSpace(), u, param[i]))
		del u
	del hdf5file
	del param

	return dict_snapshots, dict_sensors

def offline_phase(problem):
	# Create dictionary of snapshots
	# ==============================

	# Define parameter domain
	# paramRange = [ (param1_min, param1_max), (param2_min, param_2max), ...]
	paramRange = problem.paramRange()
	paramDomainSnapshots  = ParameterDomain(paramRange)

	# Sampling strategy (uniform or random)
	nSamplesPerParameter = problem.nSamplesPerParameter()
	samplingStrategy = SamplingUniform(paramDomainSnapshots, nSamplesPerParameter)

	# Dictionary
	dict_factory_snapshots = DictionaryFactory(problem, samplingStrategy, typeFactory='snapshots')
	dict_snapshots = dict_factory_snapshots.create()

	# Create dictionary of sensors
	# =============================

	# paramRange = [ (xmin, xmax), (ymin, ymax), (sigma_min, sigma_max) ]
	paramRange = problem.position_range_sensor()
	paramRange.append(problem.sigma_range_sensor())
	paramDomainSensors = ParameterDomain(paramRange)

	# Sampling strategy (uniform or random)
	nSamplesPerParameter = problem.n_sensors_per_direction().tolist()
	nSamplesPerParameter.append(problem.n_sigma_sensors())
	samplingStrategy = SamplingUniform(paramDomainSensors, nSamplesPerParameter)

	# Dictionary
	dict_factory_sensors = DictionaryFactory(problem, samplingStrategy, typeFactory='sensors')
	dict_sensors = dict_factory_sensors.create()

	return dict_snapshots, dict_sensors

# MAIN PROGRAM
# ============
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--problemType', help='{DiffusionCheckerboard}')
parser.add_argument('--offline', help='Run offline phase', action='store_true')
parser.add_argument('--id', help='Job ID for output folder')
args = parser.parse_args()

# Folder management
# -----------------
directoryPrefix = ''
if args.id is not None:
    directoryPrefix = args.problemType + '/' + args.id + '/'
else:
    directoryPrefix = args.problemType + '/default/'
directoryOffline = directoryPrefix + 'offline/'

# Dictionary of problems
# ----------------------
problemDict = {'DiffusionCheckerboard': DiffusionCheckerboard}
Problem = None
if args.problemType in problemDict:
    Problem = problemDict[args.problemType]
else:
    Problem = problemDict['DiffusionCheckerboard']

# Define problem. (Here we use the diffusion checkerboard problem)
# ==============
problem = Problem(l=2)

print('General info')
print('============')
print(Problem.get_info())

# Offline phase
# -------------
print('Offline phase')
print('=============')
if args.offline:
	# Offline phase
	dict_snapshots, dict_sensors = offline_phase(problem)
	# Save objects from offline phase
	print('Saving objects from offline phase')
	# Check if directory exists
	if not os.path.exists(directoryOffline):
		os.makedirs(directoryOffline)
	# Save snapshots
	save_offline_phase(directoryOffline, dict_snapshots, dict_sensors)
else:
	# Load objects from offline phase
	print('Loading objects from offline phase')
	dict_snapshots, dict_sensors = load_offline_phase(directoryOffline, problem)

# Create reduced space of dim n 
# We need to specify construction strategy:
# random, PCA (to do), greedy
# ========================================
n = 14
basis_Vn = BasisConstructionGreedy(dict_snapshots, n).generateBasis()
Vn = Space(basis_Vn)

# Create sensing space of dimension m
# We need to specify construction strategy:
# random, greedy (to do)
# =======================================
m = 15
basis_Wm = BasisConstructionRandom(dict_sensors, m).generateBasis()
Wm = Space(basis_Wm)

# State estimation
# ================
u = dict_snapshots[0]
se = StateEstimation(Vn, Wm)
u_star, v_star, cond = se.measure_and_reconstruct(u, disp_cond=False)

# Plot the exact function and its optimal reconstruction
import matplotlib.pyplot as plt
plt.figure
plot(u.fun)
# plt.savefig('u.pdf')

plt.figure
plot(u_star.fun)
# plt.savefig('u_star.pdf')






