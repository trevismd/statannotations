import re
import unittest

import scipy
from packaging import version

import pandas as pd
import seaborn as sns

from statannotations.Annotator import Annotator, _ensure_ax_operation_format
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

        self.x_pairs = [("a", "b")]
        self.x_pairs_tuples = [(("a",), ("b",))]
        self.pairs = [(("a", "blue"), ("b", "blue")),
                      (("a", "blue"), ("a", "red"))]
        self.df.y = self.df.y.astype(float)
        self.params_df = {
            "data": self.df,
            "x": "x",
            "y": "y",
            "hue": "color",
            "order": ["a", "b"],
            "hue_order": ['red', 'blue']}

        self.params_df_no_hue = {
            "data": self.df,
            "x": "x",
            "y": "y",
            "order": ["a", "b"],
        }

        self.params_df_redundant_hue = {
            "data": self.df,
            "x": "x",
            "y": "y",
            "hue": "x",
            "order": ["a", "b"],
            "hue_order": ["b", "a"]}

        self.df_x_float = pd.DataFrame(
            data={
                "x_axis": [1.01, 1.02, 1.03, 1.01, 1.02, 1.03, 1.01, 1.02,
                           1.03, 1.01, 1.02, 1.03, 1.01, 1.02, 1.03, 1.01,
                           1.02, 1.03, 1.01, 1.02, 1.03, 1.01, 1.02, 1.03],
                "y_axis": [16.99, 10.34, 21.01, 23.68, 24.59, 25.29, 8.77,
                           26.88, 15.04, 14.78, 10.27, 35.26, 15.42, 18.43,
                           14.83, 21.58, 10.33, 16.29, 16.97, 20.65, 17.92,
                           20.29, 15.77, 39.42],
                "hue": ["hue_2", "hue_2", "hue_1", "hue_1", "hue_2", "hue_2",
                        "hue_2", "hue_1", "hue_2", "hue_1", "hue_1", "hue_2",
                        "hue_1", "hue_1", "hue_0", "hue_0", "hue_2", "hue_1",
                        "hue_0", "hue_0", "hue_2", "hue_0", "hue_2", "hue_1"]
            }
        )
        self.params_float = {
            "data": self.df_x_float,
            "x": "x_axis",
            "y": "y_axis",
            "hue": "hue",
            "hue_order": ["hue_0", "hue_1", "hue_2"],
            "order": [1.01, 1.02, 1.03]
        }
        self.ax_float = sns.boxplot(**self.params_float)

        self.params_arrays = {
            "data": None,
            "x": self.df['x'],
            "y": self.df['y'],
            "hue": self.df['color'],
            "order": ["a", "b"],
            "hue_order": ['red', 'blue']}

        self.params_arrays_redundant_hue = {
            "data": None,
            "x": self.df['x'],
            "y": self.df['y'],
            "hue": self.df['x'],
            "order": ["a", "b"],
            "hue_order": ['b', 'a']}

    def tearDown(self) -> None:
        sns.mpl.pyplot.clf()
        return super().tearDown()

    def test_init_simple(self):
        self.annot = Annotator(self.ax, [(0, 1)], data=self.data)

    def test_init_float(self):
        self.annot_float = Annotator(
            self.ax_float,
            pairs=[
                ((1.01, "hue_0"), (1.01, "hue_1")),
                ((1.02, "hue_0"), (1.02, "hue_1"))
            ],
            **self.params_float
        )

    def test_init_df_no_hue(self):
        self.ax = sns.boxplot(**self.params_df_no_hue)
        self.annot = Annotator(self.ax, pairs=self.x_pairs, **self.params_df_no_hue)

    def test_init_df(self):
        self.ax = sns.boxplot(**self.params_df)
        self.annot = Annotator(self.ax, pairs=self.pairs, **self.params_df)

    def test_init_df_with_redundant_hue(self):
        self.ax = sns.boxplot(**self.params_df_redundant_hue)
        self.annot = Annotator(self.ax, pairs=self.x_pairs, **self.params_df_redundant_hue)

    def test_init_df_with_redundant_hue_singleton_tuples(self):
        self.ax = sns.boxplot(**self.params_df_redundant_hue)
        self.annot = Annotator(self.ax, pairs=self.x_pairs_tuples, **self.params_df_redundant_hue)

    def test_init_df_violinplot_with_redundant_hue(self):
        self.ax = sns.violinplot(**self.params_df_redundant_hue)
        self.annot = Annotator(self.ax, plot="violinplot", pairs=self.x_pairs, **self.params_df_redundant_hue)

    def test_init_df_violinplot_horizontal_with_redundant_hue(self):
        horizontal_params = {**self.params_df_redundant_hue, "x": "y", "y": "x", "orient": "h"}
        self.ax = sns.violinplot(**horizontal_params)
        self.annot = Annotator(self.ax, plot="violinplot", pairs=self.x_pairs, **horizontal_params)

    def test_init_arrays(self):
        self.ax = sns.boxplot(**self.params_arrays)
        self.annot = Annotator(self.ax, pairs=self.pairs, **self.params_arrays)

    def test_init_arrays_with_redundant_hue(self):
        self.ax = sns.boxplot(**self.params_arrays_redundant_hue)
        self.annot = Annotator(self.ax, pairs=self.x_pairs, **self.params_arrays_redundant_hue)

    def test_init_barplot(self):
        ax = sns.barplot(data=self.data)
        self.annot = Annotator(ax, [(0, 1)], plot="barplot", data=self.data)

    def test_init_stripplot(self):
        ax = sns.stripplot(**self.params_df)
        self.annot = Annotator(ax, plot="stripplot", pairs=self.pairs, **self.params_df)

    def test_test_name_provided(self):
        self.test_init_simple()
        with self.assertRaisesRegex(ValueError, "test"):
            self.annot.apply_test()

    def test_unmatched_x_in_box_pairs_without_hue(self):
        with self.assertRaisesRegex(ValueError, "(specified in `pairs`)"):
            self.annot = Annotator(self.ax, [(0, 2)], data=self.data)

    def test_order_in_x(self):
        with self.assertRaisesRegex(ValueError, "(specified in `order`)"):
            self.annot = Annotator(self.ax, [(0, 2)], data=self.data,
                                   order=[0, 1, 2])

    def test_working_hue_orders(self):
        self.annot = Annotator(self.ax, [(("a", "blue"), ("b", "blue"))],
                               data=self.df, x="x", y="y",
                               order=["a", "b"], hue='color',
                               hue_order=['red', 'blue'])

    def test_unmatched_hue_in_hue_order(self):
        with self.assertRaisesRegex(ValueError, "(specified in `hue_order`)"):
            self.annot = Annotator(self.ax, [(("a", "blue"), ("b", "blue"))],
                                   data=self.df, x="x", y="y",
                                   order=["a", "b"], hue='color',
                                   hue_order=['red', 'yellow'])

    def test_unmatched_hue_in_box_pairs(self):
        with self.assertRaisesRegex(ValueError, "(specified in `pairs`)"):
            self.annot = Annotator(self.ax, [(("a", "yellow"), ("b", "blue"))],
                                   data=self.df, x="x", y="y",
                                   order=["a", "b"], hue='color',
                                   hue_order=['red', 'blue'])

    def test_unmatched_hue_in_box_pairs_arrays(self):
        with self.assertRaisesRegex(ValueError, "(specified in `pairs`)"):
            self.annot = Annotator(self.ax, [(("a", "yellow"), ("b", "blue"))],
                                   data=None, x=self.df['x'], y=self.df['y'],
                                   order=["a", "b"], hue='color',
                                   hue_order=['red', 'blue'])

    def test_unmatched_x_in_box_pairs_with_hue(self):
        with self.assertRaisesRegex(ValueError, "(specified in `pairs`)"):
            self.annot = Annotator(self.ax, [(("c", "blue"), ("b", "blue"))],
                                   data=self.df, x="x", y="y",
                                   order=["a", "b"], hue='color',
                                   hue_order=['red', 'blue'])

    def test_location(self):
        self.test_init_simple()
        with self.assertRaisesRegex(ValueError, "argument `loc`"):
            self.annot.configure(loc="somewhere")

    def test_unknown_parameter(self):
        self.test_init_simple()
        with self.assertRaisesRegex(
                InvalidParametersError, re.escape("parameter(s) \"that\"")):
            self.annot.configure(that="this")

    def test_format(self):
        self.test_init_simple()

        with self.assertRaisesRegex(ValueError, "argument `text_format`"):
            self.annot.configure(pvalue_format={'text_format': 'that'})

    def test_apply_comparisons_correction(self):
        self.test_init_simple()
        self.assertIsNone(self.annot._apply_comparisons_correction([]))

    def test_get_configuration(self):
        self.test_init_simple()
        conf = self.annot.get_configuration()
        assert "pvalue_format" in conf

    def test_correct_num_custom_annotations(self):
        self.test_init_simple()
        self.annot.annotate_custom_annotations(["Annotation"])

    def test_wrong_num_custom_annotations(self):
        self.test_init_simple()
        with self.assertRaisesRegex(ValueError, "same length"):
            self.annot.set_custom_annotations(["One", "Two"])

    def test_not_implemented_plot(self):
        with self.assertRaises(NotImplementedError):
            Annotator(self.ax, [(0, 1)], data=self.data, plot="thatplot")

    def test_reconfigure_alpha(self):
        self.test_init_simple()
        with self.assertWarnsRegex(UserWarning, "pvalue_thresholds"):
            self.annot.configure(alpha=0.1)
        self.annot.reset_configuration()
        self.assertEqual(0.05, self.annot.alpha)

    def test_reconfigure_alpha_with_thresholds(self):
        self.test_init_simple()
        self.annot.configure(alpha=0.1,
                             pvalue_format={"pvalue_thresholds": DEFAULT})
        self.annot.reset_configuration()
        self.assertEqual(0.05, self.annot.alpha)

    def test_get_annotation_text_undefined(self):
        self.test_init_simple()
        self.assertIsNone(self.annot.get_annotations_text())

    def test_get_annotation_text_calculated(self):
        self.test_init_simple()
        self.annot.configure(test="Mann-Whitney", verbose=2)
        self.annot.apply_test()
        self.assertEqual(["ns"], self.annot.get_annotations_text())

    def test_support_int_labels(self):
        self.test_init_simple()
        self.annot.configure(test="Mann-Whitney")
        self.annot.apply_and_annotate()

    def test_support_float_labels(self):
        self.test_init_float()
        self.annot_float.configure(test='Mann-Whitney')
        self.annot_float.apply_and_annotate()

    def test_configure_hide_non_significant(self):
        self.test_init_simple()
        self.annot.configure(test='Mann-Whitney', hide_non_significant=True)
        self.annot.apply_and_annotate()

    def test_get_annotation_text_in_input_order(self):
        self.test_init_df()
        self.annot.configure(test="Mann-Whitney", text_format="simple")
        self.annot.apply_test()

        expected = (['M.W.W. p = 0.25', 'M.W.W. p = 0.67']
                    if version.parse(scipy.__version__) < version.parse("1.7")
                    else ['M.W.W. p = 0.33', 'M.W.W. p = 1.00'])
        self.assertEqual(expected, self.annot.get_annotations_text())

    def test_init_df_inverted(self):
        box_pairs = self.pairs[::-1]
        self.ax = sns.boxplot(**self.params_df)
        self.annot = Annotator(self.ax, pairs=box_pairs, **self.params_df)

    def test_get_annotation_text_in_input_order_inverted(self):
        self.test_init_df_inverted()
        self.annot.configure(test="Mann-Whitney", text_format="simple")
        self.annot.apply_test()

        expected = (['M.W.W. p = 0.67', 'M.W.W. p = 0.25']
                    if version.parse(scipy.__version__) < version.parse("1.7")
                    else ['M.W.W. p = 1.00', 'M.W.W. p = 0.33'])
        self.assertEqual(expected, self.annot.get_annotations_text())

    def test_apply_no_apply_warns(self):
        self.test_init_df_inverted()
        self.annot.configure(test="Mann-Whitney", text_format="simple")
        self.annot.apply_and_annotate()

        self.ax = sns.boxplot(**self.params_df)
        self.annot.new_plot(self.ax, self.pairs, **self.params_df)
        self.annot.configure(test="Levene", text_format="simple")
        with self.assertWarns(UserWarning):
            self.annot.annotate()

    def test_apply_apply_no_warns(self):
        self.test_init_df_inverted()
        self.annot.configure(test="Mann-Whitney", text_format="simple")
        self.annot.apply_and_annotate()

        self.ax = sns.boxplot(**self.params_df)
        self.annot.new_plot(self.ax, self.pairs, **self.params_df)
        self.annot.configure(test="Mann-Whitney-gt", text_format="simple")
        self.annot.apply_and_annotate()

    def test_valid_parameters_df_data_only(self):
        self.ax = sns.boxplot(ax=self.ax, data=self.data2)
        annot = Annotator(self.ax, pairs=[("X", "Y")],
                          data=self.data2)
        annot.configure(test="Mann-Whitney").apply_and_annotate()

    def test_comparisons_correction_by_name(self):
        self.ax = sns.boxplot(ax=self.ax, data=self.data2)
        annot = Annotator(self.ax, pairs=[("X", "Y")],
                          data=self.data2)
        annot.configure(test="Mann-Whitney", comparisons_correction="BH")
        annot.apply_and_annotate()

    def test_empty_annotator_wo_new_plot_raises(self):
        annot = Annotator.get_empty_annotator()
        with self.assertRaises(RuntimeError):
            annot.configure(test="Mann-Whitney")

    def test_empty_annotator_then_new_plot_ok(self):
        annot = Annotator.get_empty_annotator()
        self.ax = sns.boxplot(ax=self.ax, data=self.data2)
        annot.new_plot(self.ax, pairs=[("X", "Y")],
                       data=self.data2)
        annot.configure(test="Mann-Whitney")

    def test_ensure_ax_operation_format_args_not_ok(self):
        with self.assertRaises(ValueError):
            _ensure_ax_operation_format(["func", "param", None])

    def test_ensure_ax_operation_format_op_not_ok(self):
        with self.assertRaises(ValueError):
            _ensure_ax_operation_format(["func", ["param"]])

    def test_ensure_ax_operation_format_kwargs_not_ok(self):
        with self.assertRaises(ValueError):
            _ensure_ax_operation_format(["func", ["param"], {"that"}])

    def test_ensure_ax_operation_format_func_not_ok(self):
        with self.assertRaises(ValueError):
            _ensure_ax_operation_format([sum, ["param"], {"that": "this"}])

    def test_apply_no_hue(self):
        self.test_init_df_no_hue()
        self.annot.configure(test="Mann-Whitney").apply_and_annotate()
