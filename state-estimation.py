from parameterDomain import ParameterDomain
from sampling import SamplingStrategy, SamplingUniform, SamplingRandom

parameterDomain  = ParameterDomain([(-10., 10.), (-2., 3.)])
samplingStrategy = SamplingUniform(parameterDomain, (3, 4))

for p in samplingStrategy:
	print(p)



# dictionary = Dictionary()