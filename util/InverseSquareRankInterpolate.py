import pyterrier as pt
import pandas as pd


class InverseSquareRankInterpolate(pt.Transformer):
    """PyTerrier transformer that interpolates scores computed by `FFScore`."""

    def __init__(self) -> None:
        """Create an InverseSquareRankInterpolate transformer.
        """
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate the scores for all query-document pairs in the data frame as
        `2 * [(1 / l_rank ^ 2) + (1 / s_rank ^ 2)]`.

        Args:
            df (pd.DataFrame): The PyTerrier data frame.

        Returns:
            pd.DataFrame: A new data frame with the interpolated scores.
        """
        new_df = df[["qid", "docno", "query"]].copy()
        df['l_rank'] = df.groupby('qid')['score_0'].rank(ascending=False)
        df['s_rank'] = df.groupby('qid')['score'].rank(ascending=False)
        new_df["score"] = 2 * ((1/df['l_rank']**2) + (1/df['s_rank']**2))
        return new_df