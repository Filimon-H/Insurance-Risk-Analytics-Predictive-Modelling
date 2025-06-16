# ab_testing.py
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, ttest_ind

def annova_test(dependent_col: str, independent_col: str, data: pd.DataFrame):
    """
    Perform ANOVA test for comparing means of dependent variable across multiple groups.
    """
    try:
        groups = data.groupby(independent_col)[dependent_col].apply(list)
        f_stat, p_value = f_oneway(*groups)
        return f_stat, p_value
    except Exception as e:
        raise ValueError(f"ANOVA test failed: {str(e)}")

def ab_test(dependent_col: str, independent_col: str, data: pd.DataFrame):
    """
    Perform A/B test (t-test for independent samples) on two groups.
    """
    try:
        groups = data[independent_col].unique()
        if len(groups) != 2:
            raise ValueError("A/B test requires exactly 2 groups.")
        group1 = data[data[independent_col] == groups[0]][dependent_col]
        group2 = data[data[independent_col] == groups[1]][dependent_col]
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
        return t_stat, p_value
    except Exception as e:
        raise ValueError(f"A/B test failed: {str(e)}")

def test_hypothesis(null_hypothesis: str, p_value: float, alpha: float = 0.05):
    """
    Interpret p-value and print result of hypothesis testing.
    """
    if p_value < alpha:
        print(f"\033[1mRejected the null hypothesis:\033[0m {null_hypothesis} ")
    else:
        print(f"\033[1mAccepted the null hypothesis:\033[0m {null_hypothesis} ")
    print(f"p_value: {p_value}")
