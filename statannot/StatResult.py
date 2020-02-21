class StatResult:
    def __init__(self, test_str, test_short_name, stat_str, stat, pval):
        self.test_str = test_str
        self.test_short_name = test_short_name
        self.stat_str = stat_str
        self.stat = stat
        self.pval = pval

    @property
    def formatted_output(self):
        if self.stat_str is None and self.stat is None:
            stat_summary = '{}, P_val:{:.3e}'.format(self.test_str, self.pval)
        else:
            stat_summary = '{}, P_val={:.3e} {}={:.3e}'.format(
                self.test_str, self.pval, self.stat_str, self.stat
            )
        return stat_summary

    def __str__(self):
        return self.formatted_output


