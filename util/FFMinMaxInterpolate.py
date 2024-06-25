import pyterrier as pt
import pandas as pd


class FFMinMaxInterpolate(pt.Transformer):
    """PyTerrier transformer that interpolates scores computed by `FFScore`."""

    def __init__(self, alpha: float) -> None:
        """Create an FFMinMaxInterpolate.py transformer.

        Args:
            alpha (float): The interpolation parameter.
        """
        # attribute name needs to be exactly this for pyterrier.GridScan to work
        self.alpha = alpha
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate the scores for all query-document pairs in the data frame as
        `alpha * [(score_0 - l_min) / (l_max - l_min)] + (1 - alpha) * [(score - s_min) / (s_max - s_min)]`.

        Args:
            df (pd.DataFrame): The PyTerrier data frame.

        Returns:
            pd.DataFrame: A new data frame with the interpolated scores.
        """
        l_max = max(df['score_0'])
        s_max = max(df['score'])

        l_min = min(df['score_0'])
        s_min = min(df['score'])

        new_df = df[["qid", "docno", "query"]].copy()
        new_df["score"] = self.alpha * ((df["score_0"] - l_min) / (l_max - l_min)) + (1 - self.alpha) * (
                    (df["score"] - s_min) / (s_max - s_min))
        return new_df
