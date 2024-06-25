import pyterrier as pt
from pathlib import Path
import pandas as pd
from fast_forward import Mode
from fast_forward.encoder import TCTColBERTQueryEncoder
from fast_forward.util.pyterrier import FFScore, FFInterpolate

from util.disk import OnDiskIndex
from util.CondorcetFuseInterpolate import CondorcetFuseInterpolate
from util.EncodeTransformer import EncodeTransformer
from util.FFMinMaxInterpolate import FFMinMaxInterpolate
from util.FFZScoreInterpolate import FFZScoreInterpolate
from util.InverseSquareRankInterpolate import InverseSquareRankInterpolate
from util.CombMNZInterpolate import CombMNZInterpolate

from pyterrier.measures import RR, nDCG, MAP

from util.ReciprocalInterpolate import ReciprocalInterpolate

def transform_index(dataset):
    """
    Encode docno
    :param dataset: dataset to be encoded
    :return: dataset with encoded docno
    """
    for d in dataset.get_corpus_iter():
        d['docno'] = str(d['docno'].encode('utf-8'))
        yield d
def main():
    """
    Running ranking effectiveness experiment on DBPedia
    """
    if not pt.started():
        pt.init()

    dataset = pt.get_dataset('irds:beir/dbpedia-entity/test')
    max_doc_len = 206

    indexer = pt.IterDictIndexer(
        str(Path.cwd()),  # this will be ignored
        type=pt.index.IndexingType.MEMORY,
        meta={'docno': max_doc_len}
    )

    index_ref = indexer.index(transform_index(dataset), fields=['text', 'title', 'url'])
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")

    index_path = "ffindex_dbpedia_entity_tct.h5"
    q_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")

    ff_index = OnDiskIndex.load(
        Path(index_path), query_encoder=q_encoder, mode=Mode.MAXP
    )

    ff_index = ff_index.to_memory()
    ff_score = FFScore(ff_index)
    num_candidates = 100
    sparse = (~bm25 % num_candidates)(dataset.get_topics())
    encoding = EncodeTransformer()
    encoded = encoding(sparse)
    candidates = ff_score(encoded)

    convex = FFInterpolate(alpha=0.1)
    convex_mm = FFMinMaxInterpolate(alpha=0.4)
    convex_z = FFZScoreInterpolate(alpha=0.4)
    reciprocal = ReciprocalInterpolate(alpha=[20, 80])
    condorcet = CondorcetFuseInterpolate(alpha=0.3)
    isr = InverseSquareRankInterpolate()
    combMNZ = CombMNZInterpolate(num_candidates)

    experiment = pt.Experiment(
            [encoded,
             candidates >> convex,
             candidates >> convex_mm,
             candidates >> convex_z,
             candidates >> reciprocal,
             candidates >> condorcet,
             candidates >> isr,
             candidates >> combMNZ],
            dataset.get_topics(),
            dataset.get_qrels(),
            eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100],
            names=["BM25", "BM25 >> Convex", "BM25 >> Convex_MM", "BM25 >> Convex_Z", "BM25 >> Reciprocal",
                   "BM25 >> Condorcet", "BM25 >> ISR", "BM25 >> combMNZ"],
            baseline=1,
            correction='bonferroni'
        )

    res = []
    res.append(experiment)

    output_to_file(res)

def output_to_file(res):
    """
    Converts the result to a csv file
    :param res: list of dictionaries storing the scores
    """
    pd.concat(res).to_csv("DBPedia_experiment.csv", index=False)

if __name__ == '__main__':
    main()
