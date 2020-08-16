# import 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import src.Algorithms._Base.BaseClassifiers as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg


class LDA(BC.BaseClassifiers):
	'''
	Class of algorithm LDA
	'''

	def __init__(self, device='CPU', solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001):
		self.model = LinearDiscriminantAnalysis(solver, shrinkage, priors, n_components, store_covariance, tol)
		self.device = device
		self.typeAlgorithm = enumTypeAlg.SUPERVISED
