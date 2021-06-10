from statannotations.stats.utils import check_alpha


class StatResult:
    def __init__(self, test_description, test_short_name, stat_str, stat, pval,
                 alpha=0.05):
        """Alpha is used to know if the pvalue is non-significant as a result
        of comparison correction or even without in cases it doesn't change the
        pvalue"""
        check_alpha(alpha)
        self.test_description = test_description
        self.test_short_name = test_short_name
        self.stat_str = stat_str
        self.stat_value = stat
        self.pval = pval
        self._corrected_significance = None
        self._correction_method = None
        self.alpha = alpha

    @property
    def correction_method(self):
        return self._correction_method

    @correction_method.setter
    def correction_method(self, correction_method: str):
        self._correction_method = correction_method

    @property
    def corrected_significance(self):
        return self._corrected_significance

    @corrected_significance.setter
    def corrected_significance(self, significance: bool):
        if self._correction_method is None:
            raise ValueError("Correction method must first be set.")
        self._corrected_significance = significance and True or False

    @property
    def formatted_output(self):
        description = self.test_description

        if self._correction_method is not None:
            description += f' with {self._correction_method} correction'

        stat_summary = '{}, P_val:{:.3e}'.format(description, self.pval)

        stat_summary += self.significance_suffix

        if self.stat_str is not None or self.stat_value is not None:
            stat_summary += ' {}={:.3e}'.format(self.stat_str, self.stat_value)

        return stat_summary

    @property
    def significance_suffix(self):
        # will add this only if a correction method is specified
        if self._corrected_significance is False and self.pval <= self.alpha:
            return ' (ns)'
        return ""

    def __str__(self):
        return self.formatted_output
