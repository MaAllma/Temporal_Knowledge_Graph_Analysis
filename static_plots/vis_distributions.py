import pandas as pd

from data_handling.network import *
from data_handling.features import *
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    graph, node_df, edge_df = get_graph_attributes()

    sns.set_theme()

    # ----------------------- Bar Plot: Amount of edges per type -----------------------
    categories = edge_df[EDGE_COLUMN_TYPE].cat.categories
    occurences = edge_df[EDGE_COLUMN_TYPE].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(1,1)
    ax.bar(list(occurences.index), occurences)
    #ax.semilogy()
    ax.set_title('Types of edges')
    ax.set_ylabel('Number of edges')
    ax.set_xlabel('Type of edge')
    fig.show()

    # ----------------------- Bar Plot: Amount of nodes per type -----------------------
    node_type = edge_df[NODE_COLUMN_TYPE].cat.categories
    amount = node_df[NODE_COLUMN_TYPE].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.bar(list(amount.index), amount)
    ax.semilogy()
    ax.set_title('Types of nodes')
    ax.set_ylabel('Number of nodes')
    ax.set_xlabel('Type of node')
    plt.xticks(rotation=45)
    fig.show()

    # ----------------------- Hist Plot: Amount of edges per time -----------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    sns.histplot(edge_df[EDGE_COLUMN_START_DATE], bins=100, kde=True, ax=ax1, color='g')
    sns.histplot(edge_df[EDGE_COLUMN_END_DATE], bins=100, kde=True, ax=ax2, color='r')

    ax1.set_title('New edges over time')
    ax1.set_ylabel('Density')
    ax1.set_xlabel('Time')
    #ax1.tick_params(axis='x', rotation=45)

    ax2.set_title('Deleted edges over time')
    ax2.set_ylabel('Density')
    ax2.set_xlabel('Time')
    #ax2.tick_params(axis='x', rotation=45)

    # Highlight min and max of second x-axis on the first x-axis
    min_end_date = edge_df[EDGE_COLUMN_END_DATE].min()
    max_end_date = edge_df[EDGE_COLUMN_END_DATE].max()
    ax1.axvline(min_end_date, color='r', linestyle='-',
                label=f'Min Deleted edges: {pd.to_datetime(min_end_date).date()}')
    ax1.axvline(max_end_date, color='r', linestyle='--',
                label=f'Max Deleted edges: {pd.to_datetime(max_end_date).date()}')
    ax1.legend()

    # Set x-axis limits
    ax1.set_xlim(min(edge_df[EDGE_COLUMN_START_DATE]), max(edge_df[EDGE_COLUMN_START_DATE]))

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    # ----------------------- Relplot: Start vs End edges -----------------------
    categories = node_df[NODE_COLUMN_TYPE].cat.categories
    occurences = node_df[NODE_COLUMN_TYPE].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(1, 1)
    ax.bar(list(occurences.index), occurences)
    ax.semilogy()
    ax.set_xticklabels(list(occurences.index), rotation=45)
    ax.set_title('Types of nodes')
    ax.set_ylabel('Number of nodes')
    ax.set_xlabel('Type of node')
    fig.show()

    edge_df = edge_df.loc[np.logical_and(edge_df[EDGE_COLUMN_START_DATE] != edge_df[EDGE_COLUMN_START_DATE].min(),
                                         edge_df[EDGE_COLUMN_END_DATE] != edge_df[EDGE_COLUMN_END_DATE].max())]
    print(len(edge_df))
    sns.relplot(edge_df, x=EDGE_COLUMN_START_DATE, y=EDGE_COLUMN_END_DATE, hue=EDGE_COLUMN_TYPE)
    plt.show()

    # ----------------------- KDE Plot: Density over time of all events -----------------------
    all_dates = pd.concat([edge_df[EDGE_COLUMN_START_DATE], edge_df[EDGE_COLUMN_END_DATE]])#.to_frame(name='Dates').reset_index()
    # Note: length is propably identical for both, but I don't really care...
    #start_end_selector = pd.concat([pd.Series(['Start']*len(edge_df[EDGE_COLUMN_START_DATE])),
    #                                pd.Series(['End']*len(edge_df[EDGE_COLUMN_END_DATE]))])
    #all_dates['Event-Type'] = start_end_selector
    sns.kdeplot(all_dates)#, x='Dates', hue='Event-Type')
    plt.show()

    # ----------------------- Hist Plot: Amount of edges per time -----------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    sns.histplot(edge_df[EDGE_COLUMN_START_DATE], bins=50, kde=True, ax=ax1, color='g')
    sns.histplot(edge_df[EDGE_COLUMN_END_DATE], bins=50, kde=True, ax=ax2, color='r')

    ax1.set_title('New edges over time (without first one)')
    ax1.set_ylabel('Density')
    ax1.set_xlabel('Time')
    #ax1.tick_params(axis='x', rotation=45)
    ax1.set_xlim(min(edge_df[EDGE_COLUMN_START_DATE]), max(edge_df[EDGE_COLUMN_END_DATE]))

    ax2.set_title('Deleted edges over time (without last one)')
    ax2.set_ylabel('Density')
    ax2.set_xlabel('Time')
    #ax2.tick_params(axis='x', rotation=45)

    # Highlight min and max of second x-axis on the first x-axis
    min_end_date = edge_df[EDGE_COLUMN_END_DATE].min()
    max_end_date = edge_df[EDGE_COLUMN_END_DATE].max()
    ax1.axvline(min_end_date, color='r', linestyle='-',
                label=f'Min Deleted edges: {pd.to_datetime(min_end_date).date()}')
    ax1.axvline(max_end_date, color='r', linestyle='--',
                label=f'Max Deleted edges: {pd.to_datetime(max_end_date).date()}')
    ax1.legend()



    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()