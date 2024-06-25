import pyterrier as pt
from fast_forward.util.pyterrier import FFInterpolate
from pyterrier.measures import RR, nDCG, MAP

from util.FFMinMaxInterpolate import FFMinMaxInterpolate
from util.FFZScoreInterpolate import FFZScoreInterpolate
from util.EncodeTransformer import EncodeTransformer


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

    def identity_validation(self, alpha=0.5, encoding=False):
        """
        Validation on FFInterpolate
        :param alpha: parameter for the FFInterpolate transformer
        :param encoding: boolean that indicates if encoding is necessary
        :return: validation result
        """
        if encoding:
            return self.validation_encoded(FFInterpolate(alpha))
        else:
            return self.validation(FFInterpolate(alpha))

    def min_max_validation(self, alpha=0.5, encoding=False):
        """
        Validation on FFMinMaxInterpolate
        :param alpha: parameter for the FFMinMaxInterpolate transformer
        :param encoding: boolean that indicates if encoding is necessary
        :return: validation result
        """
        if encoding:
            return self.validation_encoded(FFMinMaxInterpolate(alpha))
        else:
            return self.validation(FFMinMaxInterpolate(alpha))

    def z_score_validation(self, alpha=0.5, encoding=False):
        """
        Validation on FFZScoreInterpolate
        :param alpha: parameter for the FFZScoreInterpolate transformer
        :param encoding: boolean that indicates if encoding is necessary
        :return: validation result
        """
        if encoding:
            return self.validation_encoded(FFZScoreInterpolate(alpha))
        return self.validation(FFZScoreInterpolate(alpha))

    def experiment(self, identity_alpha, min_max_alpha, z_alpha):
        """
        Experiment using all convex rank fusion functions
        :param identity_alpha: alpha for FFInterpolate
        :param min_max_alpha: alpha for FFMinMaxInterpolate
        :param z_alpha: alpha for FFZScoreInterpolate
        :return: experiment result
        """
        identity_int = FFInterpolate(identity_alpha)
        min_max_int = FFMinMaxInterpolate(min_max_alpha)
        z_int = FFZScoreInterpolate(z_alpha)
        return pt.Experiment(
            [self.candidates,
             self.candidates >> identity_int,
             self.candidates >> min_max_int,
             self.candidates >> z_int],
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100],
            names=["BM25", "BM25 >> Convex_Identity", "BM25 >> Convex_Min_Max", "BM25 >> Convex_Z"]
        )

    def experiment_encoded(self, identity_alpha, min_max_alpha, z_alpha):
        """
        Experiment with encoding using all convex rank fusion functions
        :param identity_alpha: alpha for FFInterpolate
        :param min_max_alpha: alpha for FFMinMaxInterpolate
        :param z_alpha: alpha for FFZScoreInterpolate
        :return: experiment result
        """
        identity_int = FFInterpolate(identity_alpha)
        min_max_int = FFMinMaxInterpolate(min_max_alpha)
        z_int = FFZScoreInterpolate(z_alpha)
        encoding = EncodeTransformer()
        return pt.Experiment(
            [self.candidates,
             self.candidates >> encoding >> self.ff_score >> identity_int,
             self.candidates >> encoding >> self.ff_score >> min_max_int,
             self.candidates >> encoding >> self.ff_score >> z_int],
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100],
            names=["BM25", "BM25 >> Convex_Identity", "BM25 >> Convex_Min_Max", "BM25 >> Convex_Z"]
        )

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

    def validation_encoded(self, ff_int):
        """
        Validation on the ff_int using map, recip_rank, and nDCG@10 with encoding
        :param ff_int: Interpolate transformer used for validation
        :return: Validation result for all metric
        """
        encoding = EncodeTransformer()
        pt.GridSearch(
            self.candidates >> encoding >> self.ff_score >> ff_int,
            {ff_int: {"alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}},
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            "map",
            verbose=True,
        )
        alpha_map = ff_int.alpha
        pt.GridSearch(
            self.candidates >> encoding >> self.ff_score >> ff_int,
            {ff_int: {"alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}},
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            "recip_rank",
            verbose=True,
        )
        alpha_RR = ff_int.alpha
        pt.GridSearch(
            self.candidates >> encoding >> self.ff_score >> ff_int,
            {ff_int: {"alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}},
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            "ndcg_cut.10",
            verbose=True,
        )
        alpha_nDCG = ff_int.alpha
        return [alpha_map, alpha_RR, alpha_nDCG]
