from matplotlib import pyplot as plt
import seaborn as sns

from data_handling.network import *

def vis_works_for_exception():
    grouped_sorted = edge_df.sort_values(by=EDGE_COLUMN_START_DATE).groupby(EDGE_COLUMN_MULTI_EDGE_ID)[
        EDGE_COLUMN_TYPE].apply(list).reset_index(name='multi_edges_types_sorted')

    # Filter cases where 'Works for' is the last entry in the list
    works_for_last = grouped_sorted[grouped_sorted['multi_edges_types_sorted'].apply(lambda x: x[-1] == 'Works for')]
    works_for_last_df = pd.merge(works_for_last, edge_df, on=[EDGE_COLUMN_MULTI_EDGE_ID], how='left')
    #print(works_for_last_df.head(25))
    sns.set_style("whitegrid")
    fig1 = sns.relplot(works_for_last_df, x=EDGE_COLUMN_START_DATE, y=EDGE_INDEX_TARGET, hue=EDGE_COLUMN_TYPE).fig.suptitle('Companies with strange multi edges (works-for last)')
    plt.xticks(rotation=45)
    plt.show()


def vis_top_companies(sort_by_column, top_i, size_column, title):
    company_df = node_df.loc[node_df[NODE_COLUMN_TYPE].apply(lambda x: 'Company' in x)]
    top_companies = company_df.sort_values(by=sort_by_column, ascending=False).head(top_i)
    #print(top_companies)
    sns.set_style("whitegrid")
    sns.relplot(top_companies, x=NODE_COLUMN_FOUNDING_DATE, y=NODE_INDEX_ID, hue=NODE_COLUMN_TYPE, size=size_column).fig.suptitle(title)
    plt.xticks(rotation=45)
    plt.show()


def vis_edge_type_dist_companies():
    global fig
    company_source_df = edge_df.loc[(edge_df[EDGE_COLUMN_SOURCE_TYPE].apply(lambda x: 'Company' in x))]
    company_target_df = edge_df.loc[(edge_df[EDGE_COLUMN_TARGET_TYPE].apply(lambda x: 'Company' in x))]
    source_counts = company_source_df[EDGE_INDEX_SOURCE].value_counts()
    target_counts = company_target_df[EDGE_INDEX_TARGET].value_counts()
    occurences_source = company_source_df[EDGE_COLUMN_TYPE].value_counts().sort_values(ascending=False)
    occurences_target = company_target_df[EDGE_COLUMN_TYPE].value_counts().sort_values(ascending=False)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.bar(list(occurences_source.index), occurences_source)
    ax2.bar(list(occurences_target.index), occurences_target)
    ax1.set_title('Company as source')
    ax1.set_ylabel('Amount')
    ax1.set_xlabel('Edge type')
    ax2.set_title('Company as target')
    ax2.set_ylabel('Amount')
    ax2.set_xlabel('Edge type')
    fig.show()


def vis_works_for_multiple_companies():
    works_for_multiple_df = edge_df.loc[edge_df[EDGE_COLUMN_TYPE].apply(lambda x: 'Works' in x)].groupby(
        EDGE_INDEX_SOURCE).size().reset_index(name='works_for_count')
    works_for_multiple_df = works_for_multiple_df.loc[works_for_multiple_df['works_for_count'] > 1]
    works_for_multiple_df = pd.merge(works_for_multiple_df, node_df, how='left',
                                     left_on=EDGE_INDEX_SOURCE,
                                     right_index=True)
    # print(works_for_multiple_df.sort_values(by='works_for_count', ascending=False).head(15))
    print(works_for_multiple_df)
    sns.set_style("whitegrid")
    sns.relplot(works_for_multiple_df, x=NODE_COLUMN_CONNECTED_COMPONENT, y=EDGE_INDEX_SOURCE,
                size=NODE_COLUMN_OUT_DEGREE).fig.suptitle('Person works for multiple companies')
    plt.xticks(rotation=45)
    plt.show()


if __name__ == '__main__':
    graph, node_df, edge_df = get_graph_attributes()

    vis_top_companies(NODE_COLUMN_CONNECTED_COMPONENT, 15, NODE_COLUMN_IN_DEGREE, 'Top 15 companies by connected component')
    vis_top_companies(NODE_COLUMN_IN_DEGREE, 15, NODE_COLUMN_IN_DEGREE, 'Top 15 companies by in degree')
    vis_top_companies(NODE_COLUMN_OUT_DEGREE, 15, NODE_COLUMN_OUT_DEGREE, 'Top 15 companies by out degree')

    vis_edge_type_dist_companies()

    vis_works_for_exception()

    vis_works_for_multiple_companies()


