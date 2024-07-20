import networkx as nx
import numpy as np
import pandas as pd
from data_handling.network import *
import itertools as itert
import functools as funct
from utils import get_logger, TRACE

CONSIDERED_QUANTILES = [0.0, 0.25, 0.5, 0.75, 1.0]
CONSIDERED_QUANTILES_NAMES = ['Min', '25th Perc.', 'Median', '75th Perc.', 'Max']
LOGGER = get_logger(__name__)
NEIGHBOR_COLUMN_EDGE_TYPE = 'Relation-type'
NEIGHBOR_COLUMN_DISTANCE = 'distance'
def node_edge_neighbors(graph: nx.MultiDiGraph, node_df: pd.DataFrame, edge_df: pd.DataFrame, neighborhood: int = 1, as_undirected: bool = True) -> pd.DataFrame:
    # Note: It is not checked that the path to neighbors actually forms a valid time-frame
    # To ensure that only neighbors are used which are valid at any point in time, please use graph_at_time_step first
    def merge_op(neighbor_df: pd.DataFrame, edge_df: pd.DataFrame) -> pd.DataFrame:
        neighbor_df = pd.merge(neighbor_df, edge_df, how='left', left_on=NODE_INDEX_ID, right_on=EDGE_INDEX_SOURCE)
        neighbor_df[NEIGHBOR_COLUMN_DISTANCE] = 1
        return neighbor_df

    def get_distance_neighbors(node_df: pd.DataFrame, edge_df: pd.DataFrame, graph: Union[nx.MultiDiGraph, nx.MultiGraph], distance: int) -> pd.DataFrame:
        def apply_neighbor_search(node: str):
            tree: nx.DiGraph = nx.bfs_tree(graph, node, depth_limit=distance)

            res = []
            for d, layer in enumerate(nx.bfs_layers(tree, node)):
                if d == 0:
                    continue
                res.extend((source, target, d) for node in layer for source, target in tree.in_edges(node))
            return res

        descendants = node_df[NODE_INDEX_ID].apply(apply_neighbor_search).explode()
        descendant_nodes = pd.DataFrame(descendants.tolist(),
                                        index=descendants.index,
                                        columns=[EDGE_INDEX_SOURCE, EDGE_INDEX_TARGET, NEIGHBOR_COLUMN_DISTANCE])
        descendant_nodes = pd.merge(node_df, descendant_nodes, how='inner', left_index=True, right_index=True)
        descendant_nodes = pd.merge(descendant_nodes, edge_df, how='inner',
                                    left_on=[EDGE_INDEX_SOURCE, EDGE_INDEX_TARGET],
                                    right_on=[EDGE_INDEX_SOURCE, EDGE_INDEX_TARGET])
        return descendant_nodes

    def adjust_df(neighbor_df: pd.DataFrame) -> pd.DataFrame:
        to_rename = {
            NODE_COLUMN_TYPE + '_x': NODE_COLUMN_TYPE,
            EDGE_COLUMN_TYPE + '_y': NEIGHBOR_COLUMN_EDGE_TYPE,
        }
        neighbor_df = neighbor_df.rename(columns=to_rename)

        return neighbor_df


    if neighborhood < 1:
        raise ValueError(f'Cannot extract neighbor based features with neighborhood = {neighborhood} < 1, since that would imply not to consider neighbors!')

    node_df = node_df.reset_index()
    edge_df = edge_df.reset_index()
    edge_df = edge_df.drop(columns=['key'])

    if as_undirected:
        LOGGER.debug('Requested node-edge neighbor calculation in a undirected setting. Converting to undirected graph.')
        reversed_edge_df = edge_df.copy()
        reversed_edge_df[EDGE_INDEX_SOURCE] = edge_df [EDGE_INDEX_TARGET]
        reversed_edge_df[EDGE_INDEX_TARGET] = edge_df[EDGE_INDEX_SOURCE]
        edge_df = pd.concat([edge_df, reversed_edge_df], axis='rows')
        graph = graph.to_undirected()
        LOGGER.debug('Converting to undirected graph completed.')


    if neighborhood > 1:
        LOGGER.debug('Computing neighbors with neighborhood size of %d which can be slow.', neighborhood)
        neighbor_df = get_distance_neighbors(node_df, edge_df, graph, neighborhood)
    else:
        LOGGER.debug('Computing neighbors with neighborhood size of %d using a fast data-frame merge.', neighborhood)
        neighbor_df = merge_op(node_df, edge_df)
    LOGGER.debug('Computing neighbors with neighborhood size of %d completed. Fixing data-frame column names and returning result.', neighborhood)
    neighbor_df = adjust_df(neighbor_df)


    print(neighbor_df.describe(include='all'))
    print(np.any(neighbor_df[EDGE_COLUMN_START_DATE] > neighbor_df[EDGE_COLUMN_END_DATE]))
    return neighbor_df


def connected_component_statistics(graph: nx.MultiDiGraph, node_df: pd.DataFrame, edge_df: pd.DataFrame, shortest_paths: bool = True) -> pd.DataFrame:
    def get_statistic_names(features: Iterable[str], statistic: Iterable[str]) -> List[str]:
        names = [f'{feat} ({stat})' for stat, feat in itert.product(statistic, features)]
        return names

    def get_objs_with_multi_index(df: pd.DataFrame, level: int = 1) -> List[Union[pd.Series, pd.DataFrame]]:
        objs = [df.xs(val, level=level) for val in df.index.levels[level]]
        return objs

    def min_max_component_normalize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        grouped = df.groupby(by=[NODE_COLUMN_CONNECTED_COMPONENT], dropna=False)[columns].agg(['min', 'max'])
        for column in columns:
            data = df[[column, NODE_COLUMN_CONNECTED_COMPONENT]]
            combined = pd.merge(data, grouped[column], how='inner', left_on=NODE_COLUMN_CONNECTED_COMPONENT, right_index=True)
            normalized = (combined[column] - combined['min'] ) / (combined['max'] - combined['min'])
            # if min == max, then we have a densly connected component where all nodes are pretty similiar
            # => use 0.5 to indicate that the node is connected in a similiar manner as everyone else
            normalized.loc[normalized.isna()] = 0.5
            df[column] = normalized
        return df

    if shortest_paths:
        LOGGER.debug('Requested shortest paths for connected component statistics. Computing.')
        apsps = get_shortest_path_distances(graph)
        eccentricities = apsps.eccentricities()
        med_distances = apsps.med_distance()
        del apsps
        node_df = pd.merge(node_df, eccentricities, how='inner', left_index=True, right_index=True)
        node_df = pd.merge(node_df, med_distances, how='inner', left_index=True, right_index=True)
        node_df = min_max_component_normalize(node_df, [APSP_COLUMN_ECCENTRICITY, APSP_COLUMN_MED_DISTANCE])
        del eccentricities
        del med_distances
        LOGGER.debug('Computed shortest path statistics.')

    node_df['degree'] = node_df[NODE_COLUMN_IN_DEGREE] + node_df[NODE_COLUMN_OUT_DEGREE]
    LOGGER.debug('Grouping Node-dataframe for feature extraction')
    grouped = node_df.groupby([NODE_COLUMN_CONNECTED_COMPONENT], observed=True)

    all_quantiles = grouped.quantile(CONSIDERED_QUANTILES, interpolation='linear', numeric_only=True)
    mean = grouped[list(node_df.select_dtypes(include=[np.number]))].mean()
    std = grouped[list(node_df.select_dtypes(include=[np.number]))].std()
    # to get a specific quantile: all_quantiles.xs(your_quantile, level=1)
    num_elements = grouped.size()
    types = grouped.value_counts(subset=['type'], normalize=True)
    objs = get_objs_with_multi_index(all_quantiles)
    names = get_statistic_names(all_quantiles.columns, CONSIDERED_QUANTILES_NAMES)

    objs.append(mean)
    names.extend(get_statistic_names(mean.columns, ['Mean']))

    objs.append(std)
    names.extend(get_statistic_names(std.columns, ['Std.']))

    objs.append(num_elements)
    names.append('#nodes')

    objs.extend(get_objs_with_multi_index(types))
    names.extend(get_statistic_names(['Type-Perc.'], types.index.levels[1]))
    res = pd.concat(objs, axis='columns')
    res.columns = names
    LOGGER.debug('Extracted features for node-only dataframe. Computing edge-features.')

    return res


if __name__ == '__main__':
    graph, node_df, edge_df = get_graph_attributes()
    res = connected_component_statistics(graph, node_df, edge_df)
    print(res.head())
    print(res.describe(include='all'))
    #node_edge_neighbors(graph, node_df, edge_df)