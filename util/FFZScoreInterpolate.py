import pyterrier as pt
import pandas as pd


class FFZScoreInterpolate(pt.Transformer):
    """PyTerrier transformer that interpolates scores computed by `FFScore`."""

    def __init__(self, alpha: float) -> None:
        """Create an FFZScoreInterpolate.py transformer.

        Args:
            alpha (float): The interpolation parameter.
        """
        # attribute name needs to be exactly this for pyterrier.GridScan to work
        self.alpha = alpha
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate the scores for all query-document pairs in the data frame as
        `alpha * [(score_0 - l_mean) / l_std]  + (1 - alpha) * [(score - s_mean) / s_std]`.

        Args:
            df (pd.DataFrame): The PyTerrier data frame.

        Returns:
            pd.DataFrame: A new data frame with the interpolated scores.
        """

        l_mean = df['score_0'].mean()
        s_mean = df['score'].mean()

        l_std = df['score_0'].std()
        s_std = df['score'].std()

        new_df = df[["qid", "docno", "query"]].copy()
        new_df["score"] = self.alpha * ((df["score_0"] - l_mean)/l_std) + (1 - self.alpha) * ((df["score"] - s_mean)/s_std)
        return new_df