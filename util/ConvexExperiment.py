import pyterrier as pt
from fast_forward.util.pyterrier import FFInterpolate
from pyterrier.measures import RR, nDCG, MAP

from util.FFMinMaxInterpolate import FFMinMaxInterpolate
from util.FFZScoreInterpolate import FFZScoreInterpolate

class ConvexExperiment(object):
    """Object that facilitates in experiments for convex rank fusion functions"""
    def __init__(self, candidates, dataset, num_candidates=100):
        """
        Creates the ConvexExperiment object.
        :param candidates: pd.Dataframe of candidates used for the experiment with their FFScore
        :param dataset: dataset used for the experiment
        :param num_candidates: number of candidate documents retrieved for each query
        """
        self.candidates = candidates
        self.dataset = dataset
        self.num_candidates = num_candidates

    def identity_validation(self, alpha=0.5):
        """
        Validation on FFInterpolate
        :param alpha: parameter for the FFInterpolate transformer
        :param encoding: boolean that indicates if encoding is necessary
        :return: validation result
        """
        return self.validation(FFInterpolate(alpha))

    def min_max_validation(self, alpha=0.5):
        """
        Validation on FFMinMaxInterpolate
        :param alpha: parameter for the FFMinMaxInterpolate transformer
        :param encoding: boolean that indicates if encoding is necessary
        :return: validation result
        """
        return self.validation(FFMinMaxInterpolate(alpha))

    def z_score_validation(self, alpha=0.5):
        """
        Validation on FFZScoreInterpolate
        :param alpha: parameter for the FFZScoreInterpolate transformer
        :param encoding: boolean that indicates if encoding is necessary
        :return: validation result
        """
        return self.validation(FFZScoreInterpolate(alpha))

    def validation(self, ff_int):
        """
        Validation on the ff_int using map, recip_rank, and nDCG@10
        :param ff_int: Interpolate transformer used for validation
        :return: Validation result for all metric
        """
        pt.GridSearch(
            self.candidates >> ff_int,
            {ff_int: {"alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}},
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            "map",
            verbose=True,
        )
        alpha_map = ff_int.alpha
        pt.GridSearch(
            self.candidates >> ff_int,
            {ff_int: {"alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}},
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            "recip_rank",
            verbose=True,
        )
        alpha_RR = ff_int.alpha
        pt.GridSearch(
            self.candidates >> ff_int,
            {ff_int: {"alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}},
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            "ndcg_cut.10",
            verbose=True,
        )
        alpha_nDCG = ff_int.alpha
        return [alpha_map, alpha_RR, alpha_nDCG]