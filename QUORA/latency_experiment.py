import timeit
import os

import pyterrier as pt
from pathlib import Path
import pandas as pd
from fast_forward import OnDiskIndex, Mode
from fast_forward.encoder import TCTColBERTQueryEncoder
from fast_forward.util.pyterrier import FFScore, FFInterpolate

from util.CondorcetFuseInterpolate import CondorcetFuseInterpolate
from util.FFMinMaxInterpolate import FFMinMaxInterpolate
from util.FFZScoreInterpolate import FFZScoreInterpolate
from util.InverseSquareRankInterpolate import InverseSquareRankInterpolate
from util.CombMNZInterpolate import CombMNZInterpolate
from util.ReciprocalInterpolate import ReciprocalInterpolate

from pyterrier.measures import RR, nDCG, MAP


def main():
    """
    Running latency experiment on QUORA
    """
    if not pt.started():
        pt.init()

    cur_dir = os.getcwd()
    new_file_path = os.path.join(cur_dir, 'sparse_index_quora/data.properties')

    dataset = pt.get_dataset('irds:beir/quora/test')
    index_ref = pt.IndexFactory.of(new_file_path)
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")

    index_path = "ffindex_arguana_tct.h5"
    q_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")

    ff_index = OnDiskIndex.load(
        Path(index_path), query_encoder=q_encoder, mode=Mode.MAXP
    )

    ff_index = ff_index.to_memory()
    ff_score = FFScore(ff_index)
    num_candidates = 100
    sample = dataset.get_topics().sample(n=100, random_state=42)

    candidates = (~bm25 % num_candidates)(sample)
    candidates = ff_score(candidates)
    convex_z_time = timeit.repeat(stmt="convex_z(candidates, dataset)",
                                  setup="from __main__ import convex_z",
                                  repeat=4,
                                  number=3,
                                  globals=locals())
    reciprocal_time = timeit.repeat(stmt="reciprocal_identity(candidates, dataset)",
                                    setup="from __main__ import reciprocal_identity",
                                    repeat=4,
                                    number=3,
                                    globals=locals())
    comb_MNZ_time = timeit.repeat(stmt="comb_MNZ(candidates, dataset, num_candidates)",
                                  setup="from __main__ import comb_MNZ",
                                  repeat=4,
                                  number=3,
                                  globals=locals())
    convex_mm_time = timeit.repeat(stmt="convex_mm(candidates, dataset)",
                                   setup="from __main__ import convex_mm",
                                   repeat=4,
                                   number=3,
                                   globals=locals())
    convex_time = timeit.repeat(stmt="convex_identity(candidates, dataset)",
                                setup="from __main__ import convex_identity",
                                repeat=4,
                                number=3,
                                globals=locals())
    inverse_square_rank_time = timeit.repeat(stmt="inverse_square_rank(candidates, dataset)",
                                                 setup="from __main__ import inverse_square_rank",
                                                 repeat=4,
                                                 number=3,
                                                 globals=locals())
    condorcet_time = timeit.repeat(stmt="condorcet(candidates, dataset)",
                                   setup="from __main__ import condorcet",
                                   repeat=4,
                                   number=3,
                                   globals=locals())


    data = {
        'convex_time': convex_time,
        'convex_mm_time': convex_mm_time,
        'convex_z_time': convex_z_time,
        'reciprocal_time': reciprocal_time,
        'condorcet_time': condorcet_time,
        'inverse_square_rank_time': inverse_square_rank_time,
        'comb_MNZ_time': comb_MNZ_time
    }
    df = pd.DataFrame(data)
    output_to_file(df)


def convex_identity(candidates, dataset):
    """
    Experiment using FFInterpolate Transformer
    :param candidates: candidates to run experiment on
    :param dataset: dataset to run experiment on
    """
    ff_int = FFInterpolate(0.1)
    pt.Experiment(
        [candidates >> ff_int],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=[nDCG @ 10],
        names=["Convex"]
    )


def convex_mm(candidates, dataset):
    """
    Experiment using FFMinMaxInterpolate Transformer
    :param candidates: candidates to run experiment on
    :param dataset: dataset to run experiment on
    """
    ff_int = FFMinMaxInterpolate(0.5)
    pt.Experiment(
        [candidates >> ff_int],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=[nDCG @ 10],
        names=["Convex_MM"]
    )


def convex_z(candidates, dataset):
    """
    Experiment using FFZScoreInterpolate Transformer
    :param candidates: candidates to run experiment on
    :param dataset: dataset to run experiment on
    """
    ff_int = FFZScoreInterpolate(0.3)
    pt.Experiment(
        [candidates >> ff_int],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=[nDCG @ 10],
        names=["Convex_Z"]
    )
def reciprocal_identity(candidates, dataset):
    """
    Experiment using ReciprocalInterpolate Transformer
    :param candidates: candidates to run experiment on
    :param dataset: dataset to run experiment on
    """
    ff_int = ReciprocalInterpolate([1, 1])
    pt.Experiment(
        [candidates >> ff_int],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=[nDCG @ 10],
        names=["Reciprocal"]
    )

def condorcet(candidates, dataset):
    """
    Experiment using CondorcetFuseInterpolate Transformer
    :param candidates: candidates to run experiment on
    :param dataset: dataset to run experiment on
    """
    ff_int = CondorcetFuseInterpolate(0.5)
    return pt.Experiment(
        [candidates >> ff_int],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=[nDCG @ 10],
        names=["Condorcet"]
    )

def inverse_square_rank(candidates, dataset):
    """
    Experiment using InverseSquareRankInterpolate Transformer
    :param candidates: candidates to run experiment on
    :param dataset: dataset to run experiment on
    """
    ff_int = InverseSquareRankInterpolate()
    return pt.Experiment(
        [candidates >> ff_int],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=[nDCG @ 10],
        names=["ISR"]
    )


def comb_MNZ(candidates, dataset, num_candidates=100):
    """
    Experiment using CombMNZInterpolate Transformer
    :param candidates: candidates to run experiment on
    :param dataset: dataset to run experiment on
    """
    ff_int = CombMNZInterpolate(num_candidates)
    return pt.Experiment(
        [candidates >> ff_int],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=[nDCG @ 10],
        names=["CombMNZ"]
    )

def output_to_file(res):
    """
    Output the result of latency experiment in a csv file
    :param res: latency experiment result
    """
    res.to_csv("QUORA_latency_experiment.csv", index=False)


if __name__ == '__main__':
    main()
