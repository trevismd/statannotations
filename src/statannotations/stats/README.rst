1. Write your function that takes in two sets of data, and outputs a test statistic and a p-value:

.. code-block:: python

    import numpy as np
    from scipy.stats import ttest_ind

    def log_ttest(group_data1, group_data2, **stats_params):
        group_data1_log = np.log(group_data1)
        group_data2_log = np.log(group_data2)

        return ttest_ind(group_data1_log, group_data2_log, **stats_params)

2. Initialize a ``statannotations.stats.StatTest.StatTest`` :ref:`object <StatTest_module>` using your function:

.. code-block:: python

    from statannotations.stats.StatTest import StatTest

    custom_long_name = 'Log t-test'
    custom_short_name = 'log-t'
    custom_func = log_ttest
    custom_test = StatTest(custom_func, custom_long_name, custom_short_name)

3. When you configure the ``statannotations.Annotator.Annotator`` :ref:`object <Annotator_module>`, you can pass your ``StatTest``:

.. code-block:: python

    annot = Annotator(<ax>, <pairs>)
    annot.configure(test=custom_test, comparisons_correction=None,
                    text_format='star')
