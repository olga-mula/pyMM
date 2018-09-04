import numpy as np
import scipy.linalg

from fenics import *

class HilbertSpace():
	"""
		This is the ambient space, which gives the norm and the inner product.
		We can also do linear combinations and orthonormalize a family of functions.
		The functions of the space are of type FE_function. They can also be
		of type Snapshot or Sensor which inherit from FE_function.
	"""
	def __init__(self, norm_type):
		self.norm_type = norm_type

	def inner_product(self, fem_function0, fem_function1):
		ip = getattr(self, "inner_product_" + self.norm_type)
		return ip(fem_function0, fem_function1)

	def inner_product_H10(self, fem_function0, fem_function1):
		return assemble( dot(grad(fem_function0), grad(fem_function1))*dx)

	def inner_product_L2(self, fem_function0, fem_function1):
		return assemble( fem_function0 * fem_function1 *dx )

	def linear_combination(self, vecs, c):
		# Build a FE_function from a vector of coefficients
		if len(c) != len(vecs):
			raise Exception('Coefficients and vectors must be of same length!')

		u_p = None
		for i, c_i in enumerate(c):
			if i==0:
				u_p = c_i * vecs[i]
			else:
				if c_i != 0:
					u_p += c_i * vecs[i]
		return u_p

	# vecs: list of snapshots
	# returns list of FE_function
	def orthonormalize(self, vecs, grammian=None):
		if len(vecs) == 0:
			return vecs
		else:
			"""
				We do a cholesky factorisation rather than a Gram Schmidt, as
				we have a symmetric +ve definite matrix, so this is a cheap and
				easy way to get an orthonormal basis from our previous basis.
				Indeed, A = QR = LL^T and A^TA = R^TR = LL^T so L=R^T
			"""
			if grammian is None:
				grammian = self.make_grammian(vecs)
			L = np.linalg.cholesky(grammian)
			L_inv = scipy.linalg.lapack.dtrtri(L.T)[0]
			ortho_vecs = list()
			for i in range(len(vecs)):
				ortho_vecs.append( self.linear_combination(vecs, L_inv[:, i]) )

			return ortho_vecs

	def make_grammian(self, vecs):
		n = len(vecs)
		G = np.zeros((n, n))
		for i in range(self.dim):
			for j in range(i, n):
				G[i,j] = vecs[i].dot(vecs[j]) # inner product
				G[j,i] = G[i,j]
		return G

class Space(HilbertSpace):
	"""
		A finite dimensional space of the Hilbert space.
	"""
	def __init__(self, basis):
		self.basis     = basis
		self.norm_type = basis[0].ambient_space.norm_type
		self.dim       = len(self.basis)
		self.grammian  = self.make_grammian(self.basis)
		self.onb       = self.orthonormalize(self.basis,grammian=self.grammian)

	def project(self, fe_function):
		c = np.asarray( [fe_function.dot(basisFun) for basisFun in self.onb] )
		proj = self.linear_combination(self.onb, c)
		return self.linear_combination(self.onb, c)







