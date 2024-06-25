import pyterrier as pt
import pandas as pd


class CombMNZInterpolate(pt.Transformer):
    """PyTerrier transformer that interpolates scores computed by `FFScore`."""

    def __init__(self, num_candidates: int) -> None:
        """Create an CombMNZInterpolate transformer.
        """
        self.num_candidates = num_candidates
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate the scores for all query-document pairs in the data frame as
        `2 * [(num_candidates - l_rank + 1) + (num_candidates - s_rank + 1)]`.

        Args:
            df (pd.DataFrame): The PyTerrier data frame.

        Returns:
            pd.DataFrame: A new data frame with the interpolated scores.
        """
        new_df = df[["qid", "docno", "query"]].copy()
        df['s_rank'] = df.groupby('qid')['score_0'].rank(ascending=False)
        df['l_rank'] = df.groupby('qid')['score'].rank(ascending=False)
        new_df["score"] = 2 * ((self.num_candidates - df['l_rank'] + 1) + (self.num_candidates - df['s_rank'] + 1))
        return new_df