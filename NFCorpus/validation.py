import pyterrier as pt
from pathlib import Path
import pandas as pd
from fast_forward import Mode
from fast_forward.encoder import TCTColBERTQueryEncoder
from fast_forward.util.pyterrier import FFScore

from util.disk import OnDiskIndex
from util.CondorcetFuseInterpolate import CondorcetFuseInterpolate
from util.ConvexExperiment import ConvexExperiment
from util.ReciprocalExperiment import ReciprocalExperiment
from util.EncodeTransformer import EncodeTransformer

def main():
    """
    Run validation on NFCorpus
    """
    if not pt.started():
        pt.init()

    dataset = pt.get_dataset('irds:beir/nfcorpus/dev')
    max_doc_len = 8

    indexer = pt.IterDictIndexer(
        str(Path.cwd()),  # this will be ignored
        type=pt.index.IndexingType.MEMORY,
        meta={'docno': max_doc_len}
    )

    index_ref = indexer.index(dataset, fields=['text', 'title', 'url'])
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")

    index_path = "ffindex_nfcorpus_tct.h5"
    q_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")

    ff_index = OnDiskIndex.load(
        Path(index_path), query_encoder=q_encoder, mode=Mode.MAXP
    )

    ff_index = ff_index.to_memory()
    ff_score = FFScore(ff_index)
    num_candidates = 100
    sparse = (~bm25 % num_candidates)(dataset.get_topics('text'))
    candidates = ff_score(sparse)
    res = []

    convex = ConvexExperiment(candidates=candidates, dataset=dataset)
    convex_identity = convex.identity_validation()
    res.append({'function': 'convex_identity', 'MAP': convex_identity[0], 'RR': convex_identity[1], 'nDCG@10': convex_identity[2]})
    convex_mm = convex.min_max_validation()
    res.append({'function': 'convex_mm', 'MAP': convex_mm[0], 'RR': convex_mm[1], 'nDCG@10': convex_mm[2]})
    convex_z = convex.z_score_validation()
    res.append({'function': 'convex_z', 'MAP': convex_z[0], 'RR': convex_z[1], 'nDCG@10': convex_z[2]})

    reciprocal = ReciprocalExperiment(candidates=candidates, dataset=dataset)
    reciprocal_identity = reciprocal.identity_validation()
    res.append({'function': 'reciprocal_identity', 'MAP': reciprocal_identity[0], 'RR': reciprocal_identity[1], 'nDCG@10': reciprocal_identity[2]})

    condorcet_int = CondorcetFuseInterpolate(alpha=0.5)
    condorcet_fuse = condorcet_experiment(candidates, condorcet_int, dataset)
    res.append({'function': 'condorcet_fuse', 'MAP': condorcet_fuse[0], 'RR': condorcet_fuse[1], 'nDCG@10': condorcet_fuse[2]})

    output_to_file(res)

def condorcet_experiment(candidates, ff_int, dataset):
    """
    Run validation with CondorcetFuseInterpolate Transformer
    :param candidates: candidates to run interpolation on
    :param ff_int: Interpolate transformer used for validation
    :param dataset: dataset to run validation on
    :return: Validation result for all metric
    """
    pt.GridSearch(
        candidates >> ff_int,
        {ff_int: {"alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}},
        dataset.get_topics(),
        dataset.get_qrels(),
        "map",
        verbose=True,
    )
    alpha_map = ff_int.alpha
    pt.GridSearch(
        candidates >> ff_int,
        {ff_int: {"alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}},
        dataset.get_topics(),
        dataset.get_qrels(),
        "recip_rank",
        verbose=True,
    )
    alpha_RR = ff_int.alpha
    pt.GridSearch(
        candidates >> ff_int,
        {ff_int: {"alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}},
        dataset.get_topics(),
        dataset.get_qrels(),
        "ndcg_cut.10",
        verbose=True,
    )
    alpha_nDCG = ff_int.alpha
    return [alpha_map, alpha_RR, alpha_nDCG]

def output_to_file(res):
    df = pd.DataFrame(res)
    df.to_csv("NFCorpus_validation.csv", index=False)

if __name__ == '__main__':
    main()