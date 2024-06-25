import pyterrier as pt

from util.EncodeTransformer import EncodeTransformer
from util.ReciprocalInterpolate import ReciprocalInterpolate

from pyterrier.measures import RR, nDCG, MAP

class ReciprocalExperiment(object):
    """Object that facilitates in experiments for reciprocal rank fusion functions"""
    def __init__(self, candidates, dataset, num_candidates = 100):
        """
        Creates the ReciprocalExperiment object.
        :param candidates: pd.Dataframe of candidates used for the experiment with their FFScore
        :param dataset: dataset used for the experiment
        :param num_candidates: number of candidate documents retrieved for each query
        """
        self.candidates = candidates
        self.dataset = dataset
        self.num_candidates = num_candidates

    def identity_validation(self, encoding=False):
        """
        Validation on ReciprocalInterpolate
        :param alpha: parameter for the ReciprocalInterpolate transformer
        :param encoding: boolean that indicates if encoding is necessary
        :return: validation result
        """
        if encoding:
            return self.validation_encoded(ReciprocalInterpolate(alpha=[10, 10]))
        else:
            return self.validation(ReciprocalInterpolate(alpha=[10, 10]))
    def experiment(self, identity_alpha):
        """
        Experiment using ReciprocalInterpolate
        :param identity_alpha: alpha for ReciprocalInterpolate
        :return: experiment result
        """
        identity_int = ReciprocalInterpolate(alpha=identity_alpha)
        return pt.Experiment(
            [self.candidates >> identity_int],
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100],
            names=["BM25 >> Reciprocal"]
        )
    def experiment_encoded(self, identity_alpha):
        """
        Experiment with encoding using ReciprocalInterpolate
        :param identity_alpha: alpha for ReciprocalInterpolate
        :return: experiment result
        """
        identity_int = ReciprocalInterpolate(alpha=identity_alpha)
        encoding = EncodeTransformer()
        return pt.Experiment(
            [self.candidates >> encoding >> self.ff_score >> identity_int],
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100],
            names=["BM25 >> Reciprocal_Identity"]
        )

    def validation(self, ff_int):
        """
        Validation on the ff_int using map, recip_rank, and nDCG@10
        :param ff_int: Interpolate transformer used for validation
        :return: Validation result for all metric
        """
        pt.GridSearch(
            self.candidates >> ff_int,
            {ff_int: {"alpha": [[1, 1], [1, 100], [5, 10], [20, 80], [40, 60], [60, 60], [60, 40], [80, 20], [100, 1], [10, 5], [100, 100], [1000, 1000]]}},
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            "map",
            verbose=True,
        )
        alpha_map = ff_int.alpha
        pt.GridSearch(
            self.candidates >> ff_int,
            {ff_int: {"alpha": [[1, 1], [1, 100], [5, 10], [20, 80], [40, 60], [60, 60], [60, 40], [80, 20], [100, 1], [10, 5], [100, 100], [1000, 1000]]}},
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            "recip_rank",
            verbose=True,
        )
        alpha_RR = ff_int.alpha
        pt.GridSearch(
            self.candidates >> ff_int,
            {ff_int: {"alpha": [[1, 1], [1, 100], [5, 10], [20, 80], [40, 60], [60, 60], [60, 40], [80, 20], [100, 1], [10, 5], [100, 100], [1000, 1000]]}},
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
            {ff_int: {"alpha": [[1, 1], [1, 100], [5, 10], [20, 80], [40, 60], [60, 60], [60, 40], [80, 20], [100, 1], [10, 5], [100, 100], [1000, 1000]]}},
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            "map",
            verbose=True,
        )
        alpha_map = ff_int.alpha
        pt.GridSearch(
            self.candidates >> encoding >> self.ff_score >> ff_int,
            {ff_int: {"alpha": [[1, 1], [1, 100], [5, 10], [20, 80], [40, 60], [60, 60], [60, 40], [80, 20], [100, 1], [10, 5], [100, 100], [1000, 1000]]}},
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            "recip_rank",
            verbose=True,
        )
        alpha_RR = ff_int.alpha
        pt.GridSearch(
            self.candidates >> encoding >> self.ff_score >> ff_int,
            {ff_int: {"alpha": [[1, 1], [1, 100], [5, 10], [20, 80], [40, 60], [60, 60], [60, 40], [80, 20], [100, 1], [10, 5], [100, 100], [1000, 1000]]}},
            self.dataset.get_topics(),
            self.dataset.get_qrels(),
            "ndcg_cut.10",
            verbose=True,
        )
        alpha_nDCG = ff_int.alpha
        return [alpha_map, alpha_RR, alpha_nDCG]
