import ast

import pyterrier as pt
import pandas as pd


class EncodeTransformer(pt.Transformer):
    """PyTerrier transformer that provides decoding on the encoded document information."""

    def __init__(self) -> None:
        """Create an EncodeTransformer transformer.
        """
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PyTerrier Transformer providing decoding for the docno
        :param df: pd.DataFrame
        :return: pd.DataFrame
        """
        df['docno'] = df['docno'].apply(lambda x: ast.literal_eval(x).decode("utf-8"))
        return df




