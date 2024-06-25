import pyterrier as pt
from pathlib import Path
from fast_forward.encoder import TCTColBERTQueryEncoder, TCTColBERTDocumentEncoder
import torch
from fast_forward import Mode, Indexer
from util.disk import OnDiskIndex


def docs_iter(dataset):
    """
        Save the corpus data as a dictionary
        :param dataset: dataset to index
        :return: Dictionary of documents in the corpus
    """
    for d in dataset.get_corpus_iter():
        yield {"doc_id": d["docno"], "text": d["text"], "title": d["title"], "url": d["url"]}


def main():
    """
        Create Fast-Forward index for DBPedia Entity
    """
    if not pt.started():
        pt.init(tqdm="notebook")

    dataset = pt.get_dataset("irds:beir/dbpedia-entity")

    q_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")
    d_encoder = TCTColBERTDocumentEncoder(
        "castorini/tct_colbert-msmarco",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    ff_index = OnDiskIndex(
        Path("ffindex_dbpedia_entity_tct.h5"), dim=768, query_encoder=q_encoder, mode=Mode.MAXP, max_id_length=206
    )

    ff_indexer = Indexer(ff_index, d_encoder, batch_size=8)
    ff_indexer.index_dicts(docs_iter(dataset))

if __name__ == '__main__':
    main()
