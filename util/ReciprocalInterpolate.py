import pyterrier as pt
import pandas as pd


class ReciprocalInterpolate(pt.Transformer):
    """PyTerrier transformer that interpolates scores computed by `FFScore`."""

    def __init__(self, alpha: [float]) -> None:
        """Create an ReciprocalInterpolate.py transformer.

        Args:
            alpha ([float]): The interpolation parameter.
        """
        # attribute name needs to be exactly this for pyterrier.GridScan to work
        self.alpha = alpha
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate the scores for all query-document pairs in the data frame as
        `1 / (alpha[0] + l_rank) + 1 / (alpha[1] + s_rank)`.

        Args:
            df (pd.DataFrame): The PyTerrier data frame.

        Returns:
            pd.DataFrame: A new data frame with the interpolated scores.
        """
        new_df = df[["qid", "docno", "query"]].copy()
        df['l_rank'] = df.groupby('qid')['score_0'].rank(ascending=False)
        df['s_rank'] = df.groupby('qid')['score'].rank(ascending=False)
        new_df["score"] = (1 / (self.alpha[0] + df["l_rank"])) + (1 / (self.alpha[1] + df["s_rank"]))
        return new_df