from dolfin import UserExpression
import numpy as np

class RadialFunctionExpression(UserExpression):
	def __init__(self, param, **kwargs):
		super().__init__(**kwargs)
		self.pos = param[:-1]
		self.sigma = param[-1]

	def eval(self, value, x):
		value[0] = np.exp(-np.linalg.norm(x-self.pos)**2/(2*self.sigma**2))/(2*np.pi*self.sigma)

	def value_shape(self):
		return () # scalar
