# Exploratory Data Analysis Functions

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display


def head_and_tail(df, n=5):
    print("Head:")
    display(df.head(n))
    print("\nTail:")
    display(df.tail(n))


# TODO: add five-number summary annotations to boxplot (min, Q1, median, Q3, max)
def histogram_boxplot(data, feature, figsize=(15, 10), kde=False, bins=None):
    """
    Boxplot and histogram combined.

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (15,10))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )
    sns.boxplot(
        data=data,
        x=feature,
        ax=ax_box2,
        showmeans=True,
        meanline=True,
        meanprops={"linestyle": "--", "linewidth": 1.5, "color": "red"},
    )
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins
    ) if bins else sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2)
    ax_hist2.axvline(
        data[feature].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {data[feature].mean():,.0f}",
    )
    ax_hist2.axvline(
        data[feature].median(),
        color="black",
        linestyle="-",
        label=f"Median: {data[feature].median():,.0f}",
    )
    ax_hist2.legend()


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)+
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 2, 6))
    else:
        plt.figure(figsize=(n + 2, 6))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        order=data[feature].value_counts().index[:n],
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(
        loc="lower left",
        frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


### function to plot distributions wrt target


def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0])

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
    )

    plt.tight_layout()
    plt.show()


def categorical_overview(data, cat_cols):
    """
    Horizontal barplots for all categorical variables
    with a shared count axis for honest cross-feature comparison.
    """
    counts_list = []
    for var in cat_cols:
        vc = data[var].value_counts().reset_index()
        vc.columns = ["value", "count"]
        vc["variable"] = var
        vc["percent"] = vc["count"] / vc["count"].sum() * 100
        counts_list.append(vc)
    counts = pd.concat(counts_list)

    variables = counts["variable"].unique()
    heights = [data[v].nunique() for v in variables]

    fig, axes = plt.subplots(
        len(variables),
        1,
        figsize=(10, sum(heights) * 0.8 + len(variables)),
        sharex=True,
        gridspec_kw={"height_ratios": heights},
    )

    for i, (ax, var) in enumerate(zip(axes, variables)):
        # Alternating background
        if i % 2 == 0:
            ax.set_facecolor("#4242422C")
        else:
            ax.set_facecolor("white")

        subset = counts[counts["variable"] == var].sort_values("count", ascending=True)
        sns.barplot(data=subset, y="value", x="count", ax=ax, alpha=0.7)
        ax.set_title(var, fontsize=11, pad=6)
        ax.set_xlabel("")
        ax.set_ylabel("")

        for i, row in enumerate(subset.itertuples()):
            # Count inside bar
            ax.text(
                row.count * 0.02,
                i,
                f"{row.count}",
                va="center",
                ha="left",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
            # Percentage outside bar
            ax.text(
                row.count + ax.get_xlim()[1] * 0.01,
                i,
                f"{row.percent:.1f}%",
                va="center",
                ha="left",
                fontsize=9,
                color="black",
            )

        # Box around each subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("lightgrey")

    axes[-1].set_xlabel("Count")
    plt.tight_layout()
    plt.show()
