import re
import unittest
from typing import List

import scipy
from packaging import version

import pandas as pd
import seaborn as sns

from statannotations.Annotation import Annotation
from statannotations.Annotator import Annotator
from statannotations.utils import DEFAULT, InvalidParametersError


# noinspection DuplicatedCode
class TestAnnotator(unittest.TestCase):
    """Test validation of parameters"""

    def setUp(self):
        self.data = [[1, 2, 3], [2, 5, 7]]
        self.data2 = pd.DataFrame([[1, 2], [2, 5], [3, 7]], columns=["X", "Y"])
        self.ax = sns.boxplot(data=self.data)
        self.df = pd.DataFrame.from_dict(
            {1: {'x': "a", 'y': 15, 'color': 'blue'},
             2: {'x': "a", 'y': 16, 'color': 'blue'},
             3: {'x': "b", 'y': 17, 'color': 'blue'},
             4: {'x': "b", 'y': 18, 'color': 'blue'},
             5: {'x': "a", 'y': 15, 'color': 'red'},
             6: {'x': "a", 'y': 16, 'color': 'red'},
             7: {'x': "b", 'y': 17, 'color': 'red'},
             8: {'x': "b", 'y': 18, 'color': 'red'}
             }).T

        self.pairs_for_df = [(("a", "blue"), ("b", "blue")),
                             (("a", "blue"), ("a", "red"))]
        self.df.y = self.df.y.astype(float)
        self.params_df = {
            "data": self.df,
            "x": "x",
            "y": "y",
            "hue": "color",
            "order": ["a", "b"],
            "hue_order": ['red', 'blue']}

    def test_plot_and_annotate(self):
        annotations: List[Annotation]
        ax, annotations, annotator = Annotator.plot_and_annotate(
            plot="boxplot", pairs=self.pairs_for_df,
            plot_params=self.params_df,
            configuration={'test': 'Mann-Whitney', 'text_format': 'simple'},
            annotation_func='apply_test',
            ax_op_after=[['set_xlabel', ['Group'], None]],
            annotation_params={'num_comparisons': 'auto'})

        expected = (['M.W.W. p = 0.25', 'M.W.W. p = 0.67']
                    if version.parse(scipy.__version__) < version.parse("1.7")
                    else ['M.W.W. p = 0.33', 'M.W.W. p = 1.00'])
        self.assertEqual(expected, annotator.get_annotations_text())
