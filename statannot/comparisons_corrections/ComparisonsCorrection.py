from statannot.comparisons_corrections.funcs import bonferroni, holm_bonferroni, benjamin_hochberg

methods_names = {'bonferroni':        'Bonferroni',
                 'bonf':              'Bonferroni',
                 'Bonferroni':        'Bonferroni',
                 'holm-bonferroni':   'Holm-Bonferroni',
                 'HB':                'Holm-Bonferroni',
                 'Holm-Bonferroni':   'Holm-Bonferroni',
                 'benjamin-hochberg': 'Benjamin-Hochberg',
                 'BH':                'Benjamin-Hochberg',
                 'Benjamin-Hochberg': 'Benjamin-Hochberg'}

# methods_data represents function and method type:
# Type 0 means per-pvalue correction with modification
# Type 1 means correction of interpretation of set of (ordered) pvalues

methods_data = {'Bonferroni':        [bonferroni, 0],
                'Holm-Bonferroni':   [holm_bonferroni, 1],
                'Benjamin-Hochberg': [benjamin_hochberg, 1]}

IMPLEMENTED_METHODS = [name for name in methods_names.keys()]


def get_correction_method(name):
    if name is None:
        return None

    if name not in IMPLEMENTED_METHODS:
        raise NotImplementedError(f"Comparison {str(name)} not implemented.")

    correction_method = methods_names[name]

    return ComparisonsCorrection(correction_method,
                                 *methods_data[correction_method])


class ComparisonsCorrection:

    def __init__(self, name, func, method_type):
        self.name = name
        self.func = func
        self.type = method_type
        self.__doc__ = f"ComparisonCorrection wrapper for the function " \
                       f"{func.__name__}, providing the attributes\n" \
                       f"    `name`: {self.name}\n" \
                       f"    `type`: {self.type} \n" \
                       f"With\n" \
                       f"    type 0 function: Per-pvalue correction with modification\n" \
                       f"    type 1 function: Correction of interpretation of set of (ordered) pvalues\n" \
                       f"------ Original function documentation ------ \n" \
                       f"{func.__doc__}"
               
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
