# import 
from sklearn.decomposition import LatentDirichletAllocation
import src.Algorithms._Base.BaseUnmixing as BC
from src.Algorithms import enumTypeAlgorithms as enumTypeAlg


class LDA(BC.BaseUnmixing):
    '''
    Class of algorithm Latent Dirichlet Allocation
    '''

    def __init__(self, device='CPU', n_endmembers=10, doc_topic_prior=None, topic_word_prior=None, learning_method='batch',
                learning_decay=0.7, learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=-1,
                total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=None,
                verbose=0, random_state=None):
        self.model = LatentDirichletAllocation(n_endmembers, doc_topic_prior, topic_word_prior, learning_method
                                        , learning_decay, learning_offset, max_iter, batch_size,
                                        evaluate_every, total_samples, perp_tol, mean_change_tol, max_doc_update_iter,
                                        n_jobs, verbose, random_state)
        self.device = device
        self.typeAlgorithm = enumTypeAlg.UNSUPERVISED
