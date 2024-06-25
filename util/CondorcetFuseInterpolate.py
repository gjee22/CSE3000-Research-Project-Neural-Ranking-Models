import numpy as np
import pandas as pd
import pyterrier as pt


class CondorcetFuseInterpolate(pt.Transformer):
    """PyTerrier transformer that interpolates scores computed by `FFScore`."""

    def __init__(self, alpha: float) -> None:
        """Create an CondorcetFuseInterpolate transformer.

        Args:
            alpha (float): The interpolation parameter.
        """
        # attribute name needs to be exactly this for pyterrier.GridScan to work
        self.alpha = alpha
        super().__init__()

    def sortCondorcet(self, group, l_max, s_max, l_min, s_min):
        """Computes the preference relationship of the documents and returns the aggregated score

                    Args:
                        group (pd.DataFrame): The PyTerrier data frame.
                        l_max (float): maximum lexical score
                        s_max (float): maximum semantic score
                        l_mim (float): minimum lexical score
                        s_min (float): minimum semantic score

                    Returns:
                        pd.DataFrame: A new data frame with the aggregated scores.
        """
        score_mat = pd.DataFrame(np.zeros((len(group), len(group))), index=group['docno'], columns=group['docno'])
        score_mat['interpolation'] = np.zeros(len(group))
        for i, doc1 in group.iterrows():
            score_mat.at[doc1['docno'], 'interpolation'] = self.alpha * (
                        (doc1['score_0'] - l_min) / (l_max - l_min)) + (1 - self.alpha) * (
                                                                       (doc1['score'] - s_min) / (s_max - s_min))
            for j, doc2 in group.iterrows():
                if i < j:
                    if doc1['score_0'] < doc2['score_0'] and doc1['score'] < doc2['score']:
                        score_mat.at[doc2['docno'], doc1['docno']] += 1
                    if doc1['score_0'] > doc2['score_0'] and doc1['score'] > doc2['score']:
                        score_mat.at[doc1['docno'], doc2['docno']] += 1
        score_mat['score'] = score_mat.sum(axis=1)
        return score_mat.loc[:, 'score']

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate the scores for all query-document pairs in the data frame as
        `number of wins in the preference relationship + normalized convex rank fusion score`.

        Args:
            df (pd.DataFrame): The PyTerrier data frame.

        Returns:
            pd.DataFrame: A new data frame with the interpolated scores.
        """

        l_max = max(df['score_0'])
        s_max = max(df['score'])

        l_min = min(df['score_0'])
        s_min = min(df['score'])

        new_rows = []
        for _, group in df.groupby('qid'):
            scores = self.sortCondorcet(group, l_max, s_max, l_min, s_min)
            new_rows.extend(zip(group['qid'], group['docno'], group['query'], scores))
        new_df = pd.DataFrame(new_rows, columns=['qid', 'docno', 'query', 'score'])
        return new_df
