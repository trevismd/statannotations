from typing import Union

from statannotations.PValueFormat import Formatter
from statannotations.stats.StatResult import StatResult


class Annotation:
    """
    Holds data, linked structs and an optional Formatter.
    """
    def __init__(self, structs, data: Union[str, StatResult],
                 formatter: Formatter = None):
        """
        :param structs: plot structures concerned by the data
        :param data: a string or StatResult to be formatted by a formatter
        :param formatter: A Formatter object. Statannotations provides a
            PValueFormatter for StatResult objects.
        """
        self.structs = structs
        self.data = data
        self.is_custom = isinstance(data, str)
        self.formatter = formatter

    @property
    def text(self):
        if self.is_custom:
            return self.data
        else:
            if self.formatter is None:
                raise ValueError("Missing a PValueFormat object to "
                                 "format the statistical result.")
            return self.formatter.format_data(self.data)

    @property
    def formatted_output(self):
        if isinstance(self.data, str):
            return self.data
        else:
            return self.data.formatted_output

    def print_labels_and_content(self, sep=" vs. "):
        labels_string = sep.join([struct["label"]
                                  for struct in self.structs])

        print(f"{labels_string}: {self.formatted_output}")
