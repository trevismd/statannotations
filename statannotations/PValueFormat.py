from statannotations.format_annotations import pval_annotation_text, \
    simple_text
from statannotations.utils import DEFAULT, check_valid_text_format, \
    InvalidParametersError

CONFIGURABLE_PARAMETERS = [
    'fontsize',
    'pvalue_format_string',
    'simple_format_string',
    'text_format',
    'pvalue_thresholds'
]


class Formatter:
    def __init__(self):
        pass

    def config(self, *args, **kwargs):
        pass

    def format_data(self, data):
        pass


class PValueFormat(Formatter):
    def __init__(self):
        Formatter.__init__(self)
        self._pvalue_format_string = '{:.3e}'
        self._simple_format_string = '{:.2f}'
        self._text_format = "star"
        self.fontsize = 'medium'
        self._default_pvalue_thresholds = True
        self._pvalue_thresholds = self._get_pvalue_thresholds(DEFAULT)

    def config(self, **parameters):

        unmatched_parameters = parameters.keys() - set(CONFIGURABLE_PARAMETERS)
        if unmatched_parameters:
            raise InvalidParametersError(unmatched_parameters)

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        new_threshold = parameters.get("pvalue_thresholds")
        if new_threshold is not None:
            self._default_pvalue_thresholds = new_threshold == DEFAULT

        if parameters.get("text_format") is not None:
            self._update_pvalue_thresholds()

    @property
    def text_format(self):
        return self._text_format

    @text_format.setter
    def text_format(self, text_format):
        """
        :param text_format: `star`, `simple` or `full`
        """
        check_valid_text_format(text_format)
        self._text_format = text_format

    def _get_pvalue_thresholds(self, pvalue_thresholds):
        if self._default_pvalue_thresholds:
            if self.text_format == "star":
                return [[1e-4, "****"], [1e-3, "***"],
                        [1e-2, "**"], [0.05, "*"], [1, "ns"]]
            else:
                return [[1e-5, "1e-5"], [1e-4, "1e-4"],
                        [1e-3, "0.001"], [1e-2, "0.01"],
                        [5e-2, "0.05"]]

        return pvalue_thresholds

    @property
    def pvalue_format_string(self):
        return self._pvalue_format_string

    @property
    def simple_format_string(self):
        return self._simple_format_string

    @pvalue_format_string.setter
    def pvalue_format_string(self, pvalue_format_string):
        """
        :param pvalue_format_string: By default is `"{:.3e}"`, or `"{:.2f}"`
            for `"simple"` format
        """
        self._pvalue_format_string, self._simple_format_string = (
            self._get_pvalue_and_simple_formats(pvalue_format_string)
        )

    @property
    def pvalue_thresholds(self):
        return self._pvalue_thresholds

    @pvalue_thresholds.setter
    def pvalue_thresholds(self, pvalue_thresholds):
        """
        :param pvalue_thresholds: list of lists, or tuples.
            Default is:
            For "star" text_format: `[
                [1e-4, "****"],
                [1e-3, "***"],
                [1e-2, "**"],
                [0.05, "*"],
                [1, "ns"]
            ]`.

            For "simple" text_format : `[
                [1e-5, "1e-5"],
                [1e-4, "1e-4"],
                [1e-3, "0.001"],
                [1e-2, "0.01"],
                [5e-2, "0.05"]
            ]`
        """
        if pvalue_thresholds != DEFAULT:
            self._default_pvalue_thresholds = False

        self._pvalue_thresholds = self._get_pvalue_thresholds(
            pvalue_thresholds)

    @staticmethod
    def _get_pvalue_and_simple_formats(pvalue_format_string):
        if pvalue_format_string is DEFAULT:
            pvalue_format_string = '{:.3e}'
            simple_format_string = '{:.2f}'
        else:
            simple_format_string = pvalue_format_string

        return pvalue_format_string, simple_format_string

    def print_legend_if_used(self):
        if self._text_format == 'star':
            print("p-value annotation legend:")

            pvalue_thresholds = sort_pvalue_thresholds(self.pvalue_thresholds)
            print(f"{pvalue_thresholds[-1][1]}: ".rjust(10)
                  + f"p <= {pvalue_thresholds[-1][0]:.2e}")

            for i in range(len(pvalue_thresholds)-2, 0, -1):
                print(f"{pvalue_thresholds[i][1]}: ".rjust(10)
                      + f"{pvalue_thresholds[i-1][0]:.2e} < p "
                        f"<= {pvalue_thresholds[i][0]:.2e}")

            print(f"{pvalue_thresholds[0][1]}: ".rjust(10) +
                  f"p <= {pvalue_thresholds[0][0]:.2e}")

            print()

    def _update_pvalue_thresholds(self):
        self._pvalue_thresholds = self._get_pvalue_thresholds(
            self._pvalue_thresholds)

    def format_data(self, result):
        if self.text_format == 'full':
            return ("{} p = {}{}"
                    .format('{}', self.pvalue_format_string, '{}')
                    .format(result.test_short_name, result.pvalue,
                            result.significance_suffix))

        elif self.text_format == 'star':
            return pval_annotation_text(result, self.pvalue_thresholds)

        elif self.text_format == 'simple':
            return simple_text(result, self.simple_format_string,
                               self.pvalue_thresholds)

    def get_configuration(self):
        return {key: getattr(self, key) for key in CONFIGURABLE_PARAMETERS}


def sort_pvalue_thresholds(pvalue_thresholds):
    return sorted(pvalue_thresholds,
                  key=lambda threshold_notation: threshold_notation[0])
