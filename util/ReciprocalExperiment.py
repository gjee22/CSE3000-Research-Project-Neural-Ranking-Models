import pyterrier as pt

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

    def identity_validation(self):
        """
        Validation on ReciprocalInterpolate
        :param alpha: parameter for the ReciprocalInterpolate transformer
        :param encoding: boolean that indicates if encoding is necessary
        :return: validation result
        """
        return self.validation(ReciprocalInterpolate(alpha=[10, 10]))

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
