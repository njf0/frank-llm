"""Functions for analysing survey data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def t2i(
    text: str,
    point_scale: int = 5,
) -> int:
    """Convert text to integer.

    Parameters
    ----------
    text : str
        Text to convert.
    point_scale : int, optional
        Point scale of the survey, by default 5.

    Returns
    -------
    int
        Integer representation of the text.

    """
    if point_scale == 5:
        mapping = {
            'Strongly disagree': 1,
            'Somewhat disagree': 2,
            'Neither agree nor disagree': 3,
            'Somewhat agree': 4,
            'Strongly agree': 5,
        }
    elif point_scale == 3:
        mapping = {
            'Strongly disagree': 1,
            'Somewhat disagree': 1,
            'Neither agree nor disagree': 2,
            'Somewhat agree': 3,
            'Strongly agree': 3,
        }

    return mapping.get(text, None)


def i2i(
    integer: int,
) -> int:
    """Convert 5-point scale integer to 3-point scale integer.

    Parameters
    ----------
    integer : int
        Integer to convert.

    Returns
    -------
    int
        Integer on 3-point scale.

    """
    mapping = {
        1: 1,
        2: 1,
        3: 2,
        4: 3,
        5: 3,
    }

    return mapping.get(integer)


def i2t(
    integer: int,
    point_scale: int = 5,
) -> str:
    """Convert integer to text.

    Parameters
    ----------
    integer : int
        Integer to convert.
    point_scale : int, optional
        Point scale of the survey, by default 5.

    Returns
    -------
    str
        Text representation of the integer.

    """
    integer = int(integer)
    if point_scale == 5:
        mapping = {
            1: 'SD',
            2: 'sD',
            3: 'N',
            4: 'sA',
            5: 'SA',
        }
    elif point_scale == 3:
        mapping = {
            1: 'SD',
            2: 'N',
            3: 'SA',
        }

    return mapping.get(integer, None)


def remove_outliers(
    df_dict: dict[tuple, pd.DataFrame],
) -> dict[tuple, pd.DataFrame]:
    """Remove outliers from a dataframe using the IQR method.

    Parameters
    ----------
    df_dict: dict[tuple, pd.DataFrame]
        A dictionary containing dataframes with the key being a tuple of the dataset and model.

    Returns
    -------
    dict[tuple, pd.DataFrame]
        A dictionary containing dataframes with the key being a tuple of the dataset and model with outliers removed.

    """

    def get_outliers(df):
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1

        outlier_columns = df[(df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))]

        return outlier_columns

    outliers = {}
    outliers_removed = {}

    for (dataset, model), df in df_dict.items():
        outliers_df = get_outliers(df)
        outliers_removed[(dataset, model)] = df[~outliers_df.notna().any(axis=1)]
        outliers[(dataset, model)] = outliers_df
        # percentage of outliers for a given dataset and model
        # outlier_info.append(
        #     {
        #         'dataset': dataset,
        #         'model': model,
        #         'percentage': outliers.count().sum() / df.shape
        #     }
        # )

    return outliers_removed, outliers


def majority_vote(
    df_dict: dict[tuple, pd.DataFrame],
) -> pd.DataFrame:
    """Majority vote for a dataframe.

    Parameters
    ----------
    df_dict : dict[tuple, pd.DataFrame]
        A dictionary containing dataframes with the key being a tuple of the dataset and model.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the majority vote for each dataset and model.

    """
    for (dataset, model), df in df_dict.items():
        # for each column in df
        for column in df.columns:
            # get the most common value(s) in the column
            most_common = df[column].mode().to_numpy()[0]
            # set all values in the column to the most common value
            df[column] = most_common

        df_dict[(dataset, model)] = df

    return df_dict


def compute_column_sem(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """Compute the standard error of the mean for each column in a dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A dataframe containing the data.

    Returns
    -------
    pd.Series
        A series containing the standard error of the mean for each column.

    """
    # Remove NaN values and compute SEM for each column
    df = [col[~np.isnan(col)] for col in dataframe.to_numpy().T]
    sem = pd.Series([pd.Series(col).sem() for col in df])
    return sem


def compute_average_sem(
    df_dict: dict[tuple, pd.DataFrame],
) -> pd.DataFrame:
    """Compute the average standard error of the mean for each dataset and model.

    Parameters
    ----------
    df_dict : dict[tuple, pd.DataFrame]
        A dictionary containing dataframes with the key being a tuple of the dataset and model.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the average standard error of the mean for each dataset and model.

    """
    rows = []
    for (dataset, model), dataframe in df_dict.items():
        sem = compute_column_sem(dataframe)
        sem_mean = sem.mean()
        # Get mean value for all columns
        mean = pd.Series([pd.Series(col).mean() for col in dataframe.to_numpy().T]).mean()
        # Get std value for all columns
        std = pd.Series([pd.Series(col).std() for col in dataframe.to_numpy().T]).mean()

        rows.append(
            {
                'dataset': dataset,
                'model': model,
                'mean': mean,
                'std': std,
                'mean sem': sem_mean,
            }
        )

    # Convert to DataFrame and round floats to 2 decimal places
    df = pd.DataFrame(rows).round(2)
    return df


def plot_histograms(
    df_dict: dict[tuple, pd.DataFrame],
    point_scale: int = 5,
) -> None:
    """Plot histograms for each dataset and model in a 4x4 grid.

    Parameters
    ----------
    df_dict : dict[tuple, pd.DataFrame]
        A dictionary containing dataframes with the key being a tuple of the dataset and model.
    point_scale : int, optional
        The point scale of the survey, by default 5.

    """
    # subplots in a 4x4 grid
    fig, axs = plt.subplots(4, 4, figsize=(12, 7))
    # set a value of max counts for y axis
    datasets = sorted({k[0] for k in df_dict})
    models = sorted({k[1] for k in df_dict})
    max_counts = 0
    for d, dataset in enumerate(datasets):
        for m, model in enumerate(models):
            values = df_dict[(dataset, model)].to_numpy()
            # remove nan values from numpy array
            values = values[~pd.isna(values)]
            # frequency of each value
            counts = pd.Series(values).value_counts()
            # set the maximum count
            max_counts = max(max_counts, counts.max())
            # plot the frequency of each value
            axs[d, m].bar(counts.index, counts.values)
            # set x axis to i2t for each value
            axs[d, m].set_xticks(range(1, point_scale + 1))
            axs[d, m].set_xticklabels([i2t(i, point_scale) for i in range(1, point_scale + 1)], ha='center')
            # set y axis to 0-max frequency
            axs[d, m].set_ylim(0, counts.max())
            # set x axis label to the model
            axs[d, m].set_xlabel(model)
            # set y axis label to the dataset
            axs[d, m].set_ylabel(dataset)

    for ax in axs.flat:
        # set y limits for all subgraphs
        ax.set_ylim(0, max_counts)
        # set 4 y ticks
        ax.set_yticks(range(0, max_counts + 1, max_counts // 4))
        # set outer labels only
        ax.label_outer()

    # set the title of the entire plot
    # fig.suptitle('"The response answers the question"')
    plt.show()


def counts_gt_n(
    df_dict: dict[tuple, pd.DataFrame],
    n: int,
) -> pd.DataFrame:
    """Get the counts of values greater than n for each dataset and model.

    Parameters
    ----------
    df_dict : dict[tuple, pd.DataFrame]
        A dictionary containing dataframes with the key being a tuple of the dataset and model.
    n : int
        The value to compare against.


    Returns
    -------
    pd.DataFrame
        A dataframe containing the marginalised data.

    """
    results = []
    for (dataset, model), df in df_dict.items():
        # get counts for each value in the dataframe
        counts = pd.Series(df.to_numpy().flatten()).value_counts()
        # get total counts
        total_counts = counts.sum()
        # get counts greater than n
        counts_gt = counts[counts.index > n].sum()
        # get proportion of counts greater than n
        prop_gt = round(counts_gt / total_counts, 2)
        # append to results
        results.append(
            {
                'dataset': dataset,
                'model': model,
                f'counts_greater_than_{n}': prop_gt,
            }
        )

    return pd.DataFrame(results)


# def satisfaction_with_plan(
#     df_dict_plan: dict[tuple, pd.DataFrame],
#     df_dict_sat: dict[tuple, pd.DataFrame],
#     plan=True,
# ) -> pd.DataFrame:
#     """Compute the satisfaction for a response depending on presence of plan.

#     Parameters
#     ----------
#     df_dict_plan: dict[tuple, pd.DataFrame]
#         A dictionary with the dataset and model as the key and the dataframe as the value.
#     df_dict_sat: dict[tuple, pd.DataFrame]
#         A dictionary with the dataset and model as the key and the dataframe as the value.
#     plan: bool
#         Whether to compute satisfaction with plan or satisfaction with answer.

#     Returns
#     -------
#     pd.DataFrame
#         A dataframe with the satisfaction with plan for each dataset and model combination.

#     """
