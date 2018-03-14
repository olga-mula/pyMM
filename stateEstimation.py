import numpy as np
import scipy.linalg

from space import Space

class StateEstimation():
	def __init__(self, Vn, Wm):
		if Vn.norm_type != Wm.norm_type:
			raise Exception('Error - Wm and Vn are not in the same space')
		if Vn.dim > Wm.dim:
			print('Warning: dim Vn > dim Wm in state estimation.')
		self.Vn = Vn
		self.Wm = Wm
		self.GWV = self.cross_grammian()
		self.U = self.S = self.V = None

	def cross_grammian(self):
		"""
			We will work here with the orthonormal bases of Vn and Wm
		"""
		n = self.Vn.dim
		m = self.Wm.dim
		CG = np.zeros((m,n))
		for i in range(m):
			for j in range(n):
				CG[i,j] = self.Wm.onb[i].dot(self.Vn.onb[j])
		return CG

	def beta(self):
		if self.Vn.dim > self.Wm.dim:
			return 0.
		else:
			self.U, self.S, self.V = np.linalg.svd(self.GWV, full_matrices=True)
			return self.S[-1]

	def measure_and_reconstruct(self, u, disp_cond=False):
		w = np.asarray( [u.dot(basisFun) for basisFun in self.Wm.onb])
		return self.optimal_reconstruction(w, disp_cond)

	def optimal_reconstruction(self, w, disp_cond=False):
		"""
			w is an array that contains the coefficients of a function
			of the space Wm expressed in the onb of Wm.

			For the reconstuction of a u in a Hilbert space, w = P_Wm(u)
			expressed in the onb of Wm.
		"""
		if self.Vn.dim > self.Wm.dim:
			raise Exception('Error - Wm must be of higher dimensionality than Vn to be able to do optimal reconstruction')

		try:
			c = scipy.linalg.solve(self.GWV.T @ self.GWV, self.GWV.T @ w, sym_pos=True)
		except np.linalg.LinAlgError as e:
			print('Warning - unstable v* calculation, m={0}, n={1} for Wm and Vn, returning 0 function'.format(self.Wm.n, self.Vn.n))
			c = np.zeros(self.Vn.n)

		v_star = self.Vn.linear_combination(self.Vn.onb, c)

		u_star = self.Wm.linear_combination(self.Wm.onb, w) + v_star - self.Wm.project(v_star)

		cond = np.linalg.cond(self.GWV.T @ self.GWV)

		if disp_cond:
			print('Condition number of GWV.T * GWV = {0}'.format(cond))

		return u_star, v_star, cond







