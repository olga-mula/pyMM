from dolfin import Expression
import numpy as np

class RadialFunctionExpression(Expression):
	def __init__(self, param, **kwargs):
		self.pos = param[:-1]
		self.sigma = param[-1]

	def eval(self, value, x):
		value[0] = np.exp(-np.linalg.norm(x-self.pos)**2/(2*self.sigma**2))/(2*np.pi*self.sigma)
