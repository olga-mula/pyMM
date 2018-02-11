class Basis():

	def __init__(self, dimension):
		pass

class ReducedBasis(Basis):

	def __init__(self, constructionStrategy):
		self.span = constructionStrategy.generateBasis() # list of snapshots
		# self.grammian --> TODO: see with James



# class SamplingBasis(Basis): # TODO
