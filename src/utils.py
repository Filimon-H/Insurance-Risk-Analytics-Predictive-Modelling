# utils.py

def count_group_contribution(data, grouping_col):
    """
    Prints the percentage contribution of each group in a column.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data
    grouping_col (str): The name of the column to group by

    Returns:
    dict: A dictionary with group names and their percentage contribution
    """
    group_counts = data[grouping_col].value_counts(normalize=True) * 100
    group_dict = {}

    for group, pct in group_counts.items():
        print(f"{group}: {pct:.2f}% of the data")
        group_dict[group] = pct

    return group_dict
