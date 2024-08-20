from data_handling.features import *
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns


def edge_type_barchart():
    global categories, cat, fig
    # barplot for each edge type
    categories = edge_df[EDGE_COLUMN_TYPE].cat.categories
    for cat in categories:
        edge_cat_df = edge_df[edge_df[EDGE_COLUMN_TYPE] == cat]

        occurrences_source = edge_cat_df[EDGE_COLUMN_SOURCE_TYPE].value_counts().sort_values(ascending=False)
        occurrences_target = edge_cat_df[EDGE_COLUMN_TARGET_TYPE].value_counts().sort_values(ascending=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

        # bar_width = 0.35
        # index = range(len(occurrences_source))

        ax1.bar(list(occurrences_source.index), occurrences_source, color='b')
        ax2.bar(list(occurrences_target.index), occurrences_target, color='r')

        fig.suptitle('Types of nodes with edge type " ' + str(cat) + '"')
        ax1.set_ylabel('Number of edges')
        ax2.set_ylabel('Number of edges')
        ax1.set_xlabel('Type of source node')
        ax2.set_xlabel('Type of target node')
        ax1.semilogy()
        ax2.semilogy()
        ax1.set_xticklabels(list(occurrences_source.index), rotation=90)
        ax2.set_xticklabels(list(occurrences_target.index), rotation=90)
        plt.tight_layout()  # for complete x axis labels
        fig.show()


"""
def multi_edges_scatterplot():
    for target in top_targets_2:
        filtered_df = merged_edges_info_2[merged_edges_info_2[EDGE_INDEX_TARGET] == target]
        target_type = filtered_df.iloc[0][EDGE_COLUMN_TARGET_TYPE]
        person = filtered_df.iloc[0][EDGE_INDEX_SOURCE]
        # print(filtered_df)

        sns.relplot(filtered_df, x=EDGE_COLUMN_START_DATE, y=EDGE_INDEX_SOURCE, hue=EDGE_COLUMN_TYPE)
        plt.title(target + ' (' + target_type + ')')

        plt.tight_layout()
        plt.xlabel('start date')
        plt.ylabel('')
        plt.show()
    for target in top_targets_3:
        filtered_df = merged_edges_info_3[merged_edges_info_3[EDGE_INDEX_TARGET] == target]
        target_type = filtered_df.iloc[0][EDGE_COLUMN_TARGET_TYPE]
        person = filtered_df.iloc[0][EDGE_INDEX_SOURCE]
        # print(filtered_df)

        sns.relplot(filtered_df, x=EDGE_COLUMN_START_DATE, y=EDGE_COLUMN_END_DATE, hue=EDGE_COLUMN_TYPE)
        plt.title(target + ' (' + target_type + ') \n with ' + person)

        plt.tight_layout()
        plt.xlabel('start date')
        plt.ylabel('')
        plt.show()
"""


def multi_edges_barchart():
    global fig
    occurences = edge_df[EDGE_COLUMN_MULTI_EDGE_COUNT].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(1, 1)
    ax.bar(list(occurences.index), occurences)
    # ax.semilogy()
    ax.set_title('Multi Edges')
    ax.set_ylabel('Number of edges')
    ax.set_xlabel('Multi egde')
    fig.show()


def multi_edge_types_barchart():
    global occurences
    occurences = multi_edges_df['multi_edges_types'].value_counts().sort_values(ascending=False)
    occurences.index = occurences.index.map(lambda x: ', '.join(x))
    fig, ax = plt.subplots(1, 1)
    ax.bar(list(occurences.index), occurences)
    ax.semilogy()
    ax.set_title('Multi Edge Types')
    ax.set_ylabel('Amount of combinations')
    ax.set_xlabel('Edge combinations')
    ax.set_xticks(range(len(occurences)))
    ax.set_xticklabels(list(occurences.index), rotation=45, ha='right')
    plt.tight_layout()
    fig.show()


if __name__ == '__main__':
    graph, node_df, edge_df = get_graph_attributes()

    # edge_type_barchart()

    multi_edges_barchart()

    grouped = edge_df.groupby([EDGE_COLUMN_MULTI_EDGE_ID])[EDGE_COLUMN_TYPE].apply(lambda x: set(x)).reset_index(
        name='multi_edges_types')

    grouped_sorted = edge_df.sort_values(by=EDGE_COLUMN_START_DATE).groupby(EDGE_COLUMN_MULTI_EDGE_ID)[
                EDGE_COLUMN_TYPE].apply(list).reset_index(name='multi_edges_types_sorted')
    type_combinations = grouped_sorted['multi_edges_types_sorted'].value_counts()
    #print(type_combinations.head(10))

    multi_edges_df = pd.merge(grouped, edge_df.loc[edge_df[EDGE_COLUMN_MULTI_EDGE_COUNT] > 1],
                on=[EDGE_COLUMN_MULTI_EDGE_ID], how='left')
    multi_edges_df_sorted = pd.merge(grouped_sorted, edge_df.loc[edge_df[EDGE_COLUMN_MULTI_EDGE_COUNT] > 1],
                on=[EDGE_COLUMN_MULTI_EDGE_ID], how='left')

    # there are edges with the same type, same nodes, same dates
    size_filtered = multi_edges_df.loc[
                multi_edges_df['multi_edges_types'].str.len() < multi_edges_df[EDGE_COLUMN_MULTI_EDGE_COUNT]]
    # print(size_filtered)

    multi_edges_df = multi_edges_df.loc[
        multi_edges_df['multi_edges_types'].str.len() >= multi_edges_df[EDGE_COLUMN_MULTI_EDGE_COUNT]]

    multi_edge_types_barchart()

