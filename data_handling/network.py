import random
import re
from collections import defaultdict
from functools import partial
from typing import Optional, Tuple, Dict, Any, Union, List, Set, cast, Iterable, Callable

import networkx as nx
import os.path as path
import json
import pickle
import datetime
import time
import os

import numba
import numpy as np
import pandas as pd
import numba as nb
from utils import get_logger, TRACE
import hashlib  as hlib
LOGGER = get_logger(__name__)

def _f_name_with_opt_prefix(f_name: str, prefix: Optional[str] = None) -> str:
    if prefix is None:
        f = path.join('data', f_name)
    else:
        f = path.join(prefix, 'data', f_name)
    return f

def get_challenge_json(prefix: Optional[str] = None) -> Dict[str, Any]:
    challenge_json = _f_name_with_opt_prefix('mc3.json', prefix)
    LOGGER.log(TRACE, 'Loading challenge json file at %s', challenge_json)
    with open(challenge_json, 'r', encoding='utf-8') as f:
        challenge_dict = json.load(f)
    return challenge_dict

def get_challenge_graph(prefix: Optional[str] = None) -> nx.MultiDiGraph:
    """
    Load the network graph for the challenge from json.

    :param prefix: Prefix for the file-path s.t. the file is located at `prefix/data/mc3.json`.
                  If not set, working directory will be used.
    :return: Graph for Mini-Challenge-3
    """
    challenge_dict = get_challenge_json(prefix)
    return nx.node_link_graph(challenge_dict)

def _convert_to_datetime(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    def convert_invalid_format(time_string: str) -> str:
        if 'T' not in time_string:
            return time_string+'T00:00:00'
        return time_string

    for column in columns:
        fixed_strings = df[column].map(convert_invalid_format, na_action='ignore')
        df[column] = pd.to_datetime(fixed_strings)

    return df

NODE_COLUMN_TYPE = 'type'
NODE_COLUMN_COUNTRY = 'country'
NODE_COLUMN_DATE_OF_BIRTH = 'dob'
NODE_COLUMN_FOUNDING_DATE = 'founding_date'
NODE_COLUMN_GENERIC_DATE = 'date'
NODE_COLUMN_REVENUE = 'revenue'
NODE_COLUMN_TRADE_DESC = 'TradeDescription'
NODE_COLUMN_POINT_OF_CONTACT = 'PointOfContact'
NODE_COLUMN_HEAD_OF_ORG = 'HeadOfOrg'
NODE_COLUMN_PRODUCT_SERVICE = 'ProductServices'

NODE_COLUMN_CONNECTED_COMPONENT = 'connected_component'
NODE_COLUMN_IN_DEGREE = 'in_degree'
NODE_COLUMN_OUT_DEGREE = 'out_degree'
NODE_INDEX_ID = 'id'
NODE_TYPE_MAP = {
    'Entity.Person': 'Person',
    'Entity.Person.CEO': 'Ceo',
    'Entity.Organization': 'Generic Org.',
    'Entity.Organization.Company': 'Generic Company',
    'Entity.Organization.FishingCompany': 'Fishing Company',
    'Entity.Organization.LogisticsCompany': 'Logistics Company',
    'Entity.Organization.NewsCompany': 'News Company',
    'Entity.Organization.FinancialCompany': 'Financial Company',
    'Entity.Organization.NGO': 'NGO'
}

EDGE_COLUMN_START_DATE = 'start_date'
EDGE_COLUMN_END_DATE = 'end_date'
EDGE_COLUMN_TYPE = 'type'
EDGE_INDEX_SOURCE = 'source'
EDGE_INDEX_TARGET = 'target'
EDGE_INDEX_KEY = 'key'
EDGE_COLUMN_SOURCE_TYPE = 'source_type'
EDGE_COLUMN_TARGET_TYPE = 'target_type'
EDGE_COLUMN_MULTI_EDGE_ID = 'multi_edge_id'
EDGE_COLUMN_MULTI_EDGE_COUNT = 'multi_edge_count'
EDGE_TYPE_MAP = {
    'Event.Owns.BeneficialOwnership': 'Beneficial Ownership',
    'Event.Owns.Shareholdership': 'Shareholder',
    'Event.WorksFor': 'Works for',
    'Relationship.FamilyRelationship': 'Family Relation'
}
def get_graph_attributes(graph: Optional[nx.MultiDiGraph] = None, prefix: Optional[str] = None, enforce_start_date: bool = True, enforce_end_date: bool = True, merge_node_dates: bool = False) \
        -> Tuple[nx.MultiDiGraph, pd.DataFrame, pd.DataFrame]:
    """
    Note that the node-dataframe is enhanced with a connected component annotation.
    Node-dataframe is indexed by node id and edge_dataframe is indexed by (source, target, key)

    :param prefix: where data is located
    :param enforce_start_date: Whether or not to enforce start date to be present on every edge by imputing with the minimum.
    :param enforce_end_date: Whether or not to enforce end date to be present on every edge by imputing with the maximum.
    :return: A tuple of dataframes describing the nodes and edges in the graph respectively.
    """
    if graph is None:
        challenge_graph = get_challenge_graph(prefix)
    else:
        challenge_graph = graph
    connected_components = list(nx.weakly_connected_components(challenge_graph))
    cc_map = {node: i for i, c in enumerate(connected_components) for node in c}

    # For some reason, networkx discards all attributes in the graph
    # We restore them, by manually loading the json and then ensuring only things that are also present in the given graph are used
    challenge_dict = get_challenge_json(prefix)
    node_list = [d for d in challenge_dict['nodes'] if challenge_graph.has_node(d[NODE_INDEX_ID])]
    edge_list = [d for d in challenge_dict['links'] if challenge_graph.has_edge(d[EDGE_INDEX_SOURCE], d[EDGE_INDEX_TARGET], d[EDGE_INDEX_KEY])]

    node_df = pd.DataFrame(node_list)
    node_df[NODE_COLUMN_CONNECTED_COMPONENT] = pd.Series(pd.Categorical(node_df[NODE_INDEX_ID].map(cc_map), categories=list(range(len(connected_components))), ordered=True))
    node_df[NODE_COLUMN_IN_DEGREE] = node_df[NODE_INDEX_ID].map(challenge_graph.in_degree)
    node_df[NODE_COLUMN_OUT_DEGREE] = node_df[NODE_INDEX_ID].map(challenge_graph.out_degree)

    node_df = node_df.set_index([NODE_INDEX_ID])
    node_df = node_df.drop(columns=['_raw_source', '_algorithm', '_date_added', '_last_edited_date', '_last_edited_by'])
    node_df = _convert_to_datetime(node_df, [NODE_COLUMN_DATE_OF_BIRTH, NODE_COLUMN_FOUNDING_DATE])
    node_df[NODE_COLUMN_TYPE] = node_df[NODE_COLUMN_TYPE].astype('category')
    node_df[NODE_COLUMN_COUNTRY] = node_df[NODE_COLUMN_COUNTRY].astype('category')

    pd.set_option('display.max_columns', None)
    edge_df = pd.DataFrame(edge_list)
    edge_df = edge_df.set_index([EDGE_INDEX_SOURCE , EDGE_INDEX_TARGET, EDGE_INDEX_KEY])
    edge_df = edge_df.drop(columns=['_last_edited_by', '_last_edited_date', '_date_added', '_raw_source', '_algorithm'])
    edge_df = _convert_to_datetime(edge_df, [EDGE_COLUMN_START_DATE, EDGE_COLUMN_END_DATE])
    edge_df[EDGE_COLUMN_TYPE] = edge_df[EDGE_COLUMN_TYPE].astype('category')
    if enforce_start_date:
        mask = edge_df[EDGE_COLUMN_START_DATE].isna()
        edge_df.loc[mask, EDGE_COLUMN_START_DATE] = edge_df[EDGE_COLUMN_START_DATE].min()
    if enforce_end_date:
        mask = edge_df[EDGE_COLUMN_END_DATE].isna()
        edge_df.loc[mask, EDGE_COLUMN_END_DATE] = edge_df[EDGE_COLUMN_END_DATE].max()

    edge_df[EDGE_COLUMN_TYPE] = edge_df[EDGE_COLUMN_TYPE].cat.rename_categories(EDGE_TYPE_MAP)
    node_df[NODE_COLUMN_TYPE] = node_df[NODE_COLUMN_TYPE].cat.rename_categories(NODE_TYPE_MAP)

    if merge_node_dates:
        node_df[NODE_COLUMN_GENERIC_DATE] = node_df[NODE_COLUMN_FOUNDING_DATE]
        mask = node_df[NODE_COLUMN_GENERIC_DATE].isna()
        node_df.loc[mask, NODE_COLUMN_GENERIC_DATE] = node_df.loc[mask, NODE_COLUMN_DATE_OF_BIRTH]

    edge_df = edge_df.merge(node_df[[NODE_COLUMN_TYPE]].rename(columns={NODE_COLUMN_TYPE: EDGE_COLUMN_SOURCE_TYPE}),
                            how='left',
                            left_on=EDGE_INDEX_SOURCE,
                            right_index=True)

    edge_df = edge_df.merge(node_df[[NODE_COLUMN_TYPE]].rename(columns={NODE_COLUMN_TYPE: EDGE_COLUMN_TARGET_TYPE}),
                            how='left',
                            left_on=EDGE_INDEX_TARGET,
                            right_index=True)

    multi_edges_df = edge_df.groupby([EDGE_INDEX_SOURCE, EDGE_INDEX_TARGET]).size().reset_index(name=EDGE_COLUMN_MULTI_EDGE_COUNT)
    multi_edges_df[EDGE_COLUMN_MULTI_EDGE_ID] = 0
    multi_edges_df.loc[multi_edges_df[EDGE_COLUMN_MULTI_EDGE_COUNT] > 1, EDGE_COLUMN_MULTI_EDGE_ID] = range(1, (multi_edges_df[EDGE_COLUMN_MULTI_EDGE_COUNT] > 1).sum() + 1)

    edge_df = pd.merge(multi_edges_df[[EDGE_INDEX_SOURCE, EDGE_INDEX_TARGET, EDGE_COLUMN_MULTI_EDGE_ID, EDGE_COLUMN_MULTI_EDGE_COUNT]], edge_df, on=[EDGE_INDEX_SOURCE, EDGE_INDEX_TARGET], how='left')

    return challenge_graph, node_df, edge_df

class LateIndexedNode:
    def __init__(self, id: Any):
        self.id = id
        self.nbors: Set['LateIndexedNode'] = set()
        self.index: Optional[int] = None

    def add_nbor(self, other: 'LateIndexedNode'):
        self.nbors.add(other)

    def set_index(self, index: int):
        assert self.index is None
        self.index = index

    def __eq__(self, other):
        return isinstance(other, LateIndexedNode) and other.id == self.id

    def __hash__(self):
        return hash(self.id)

def graph_to_undirected_edge_array(graph: nx.MultiDiGraph) -> Tuple[List[str], np.ndarray, np.ndarray]:
    ud_graph = graph.to_undirected()
    known_nodes: Dict[str, LateIndexedNode] = {}
    for node, adj_dict in ud_graph.adjacency():
        node_instance = known_nodes.setdefault(node, LateIndexedNode(node))
        known_nodes[node] = node_instance
        for target in adj_dict.keys():
            target_instance = known_nodes.setdefault(target, LateIndexedNode(target))
            node_instance.add_nbor(target_instance)

    nbor_ls = []
    index_ls = []
    node_ls = []
    cur_idx = 0
    for cc in sorted(nx.connected_components(ud_graph), key=len):
        for node in cast(Iterable[str], cc):
            node_instance = known_nodes[node]
            node_instance.set_index(len(node_ls))
            node_ls.append(node_instance.id)
            index_ls.append(len(nbor_ls))
            if node_instance.nbors:
                for nbor in node_instance.nbors:
                    nbor_ls.append(nbor)
            else:
                nbor_ls.append(-1)

    nbor_indices = [node_instance.index for node_instance in nbor_ls]
    index_ls.append(len(nbor_ls)) #to make access easier add a dummy in the end...
    return node_ls, np.array(index_ls), np.array(nbor_indices)

import numba.typed as nt
@nb.njit(parallel=True)
def parallel_eccentricity(index_ar: np.ndarray, nbor_ar: np.ndarray) -> np.ndarray:
    # should only be used within a single cc to avoid excessive memory use
    eccentricities = np.empty(index_ar.shape[0]-1, dtype=np.int32)
    for i in nb.prange(index_ar.shape[0]-1): # remember: index_ar contains a dummy in the end
        to_search = nt.List([nbor_ar[j] for j in range(index_ar[i], index_ar[i+1]) if j >= 0])
        next_search = nt.List()
        visited = {i}
        ecc = 0
        while len(to_search):
            ecc+=1
            for index in to_search:
                visited.add(index)
                for j in range(index_ar[index], index_ar[index+1]):
                    nbor = nbor_ar[j]
                    if nbor not in visited:
                        next_search.append(nbor)
            to_search = next_search.copy()
            next_search.clear()
        eccentricities[i] = ecc
    return eccentricities

APSP_COLUMN_ECCENTRICITY = 'eccentricity'
APSP_COLUMN_MED_DISTANCE = 'median_distance'

class DisconnectedShortestPathMap:
    @staticmethod
    def from_graph(graph: nx.MultiDiGraph) -> 'DisconnectedShortestPathMap':
        import graphblas_algorithms as ga
        from tqdm import tqdm
        LOGGER.debug('Computing All-Pairs Shortest Paths from graph.')
        t = time.time()
        ud_graph = graph.to_undirected()
        node_to_ccs = {}
        path_map = []
        ccs: List[Set[str]] = cast(List[Set[str]], list(sorted(nx.connected_components(ud_graph), key=len)))
        for i, cc in tqdm(enumerate(ccs), total=len(ccs)):
            node_to_ccs.update({node: i for node in cc})
            cc_graph = ud_graph.subgraph(cc)
            ga_graph = ga.MultiGraph.from_networkx(cc_graph)
            id_node_map = list(cc_graph.nodes)
            del cc_graph # free as much memory as possible
            apspl_matrix = ga.all_pairs_shortest_path_length(ga_graph).to_dense()
            del ga_graph #free more memory
            node_id_map = {node: j for j, node in enumerate(id_node_map)}
            path_map.append((node_id_map, apspl_matrix))
        t1 = time.time() - t
        LOGGER.debug('Computing All-Pairs Shortest Paths completed for %d connected components in %.3f seconds.',
                     len(ccs), t1
                     )
        return DisconnectedShortestPathMap(node_to_ccs, path_map)

    KEY_NODE_TO_CC = 'Node to Connected Component (ID)'
    KEY_PATH_MAP = 'Shortest Path Map'
    KEY_NODE_ID_MAP = '(Local) Node ID by Name'
    KEY_APSPL_MATRIX = 'All Pairs Shortest Path Lengths Matrix'
    @staticmethod
    def from_json(json: Dict[str, Union[Dict[str, int], List[Dict[str, Union[Dict[str, int], List[List[int]]]]]]]) \
            -> 'DisconnectedShortestPathMap':
        node_to_cc = json[DisconnectedShortestPathMap.KEY_NODE_TO_CC]
        path_map = json[DisconnectedShortestPathMap.KEY_PATH_MAP]
        return DisconnectedShortestPathMap(
            node_to_cc=node_to_cc,
            path_map=[(d[DisconnectedShortestPathMap.KEY_NODE_ID_MAP],
                       np.array(d[DisconnectedShortestPathMap.KEY_APSPL_MATRIX]))
                      for d in path_map]
        )

    def __init__(self, node_to_cc: Dict[str, int], path_map: List[Tuple[Dict[str, int], np.ndarray]]):
        self.node_to_cc = node_to_cc
        self.path_map = path_map

    def __getitem__(self, nodes: Tuple[str, str]) -> int:
        cc_id1, cc_id2 = self.node_to_cc[nodes[0]], self.node_to_cc[nodes[1]]
        if cc_id1 != cc_id2:
            return 0
        node_id_map, apspl_matrix = self.path_map[cc_id1]
        node_1, node_2 = node_id_map[nodes[0]], node_id_map[nodes[1]]
        return int(apspl_matrix[node_1, node_2])

    def to_json(self) -> Dict[str, Union[Dict[str, int], List[Tuple[Dict[str, int], List[List[int]]]]]]:
        return {
            self.KEY_NODE_TO_CC: self.node_to_cc,
            self.KEY_PATH_MAP: [{self.KEY_NODE_ID_MAP: d, self.KEY_APSPL_MATRIX: ar.tolist()}
                                for d, ar in self.path_map]
        }

    def eccentricities(self) -> pd.Series:
        return self.aggregate_shortest_paths(partial(np.max, axis=0), APSP_COLUMN_ECCENTRICITY)

    def med_distance(self) -> pd.Series:
        return self.aggregate_shortest_paths(partial(np.median, axis=0), APSP_COLUMN_MED_DISTANCE)

    def aggregate_shortest_paths(self, func: Callable[[np.ndarray], np.ndarray], label: str) -> pd.Series:
        agg_by_cc = [(d, func(ar)) for d, ar in self.path_map]
        all_aggs = pd.DataFrame([{'id': id, label : ar[index]} for d, ar in agg_by_cc for id, index in d.items()])
        all_aggs = all_aggs.set_index(['id'])
        return all_aggs[label]


def get_shortest_path_distances(graph: Optional[nx.MultiDiGraph] = None, prefix: Optional[str] = None, cache: bool = True) -> DisconnectedShortestPathMap:
    if graph is None:
        challenge_graph = get_challenge_graph(prefix)
    else:
        challenge_graph = graph
    hash_val = hlib.sha256(json.dumps(nx.node_link_data(challenge_graph), sort_keys=True).encode('utf-16'), usedforsecurity=False)
    shortest_paths_file = _f_name_with_opt_prefix(path.join('apsp_'+hash_val.hexdigest(), 'shortest_paths.pth'), prefix)
    shortest_paths_file = path.abspath(shortest_paths_file)
    if cache and path.exists(shortest_paths_file) and path.isfile(shortest_paths_file):
        LOGGER.debug('Found stored shortest_paths at %s', shortest_paths_file)
        with open(shortest_paths_file, 'rb') as fd:
            shortest_paths = pickle.load(fd)
        LOGGER.debug('Load completed. Returning shortest paths.')
    else:
        from tqdm import tqdm
        import graphblas_algorithms as ga

        LOGGER.info('Shortest paths are not available, computing.')
        shortest_paths = DisconnectedShortestPathMap.from_graph(challenge_graph)
        if cache:
            LOGGER.debug('Creating shortest path cache at %s.', shortest_paths_file)
            os.makedirs(path.dirname(shortest_paths_file), exist_ok=True)
            with open(shortest_paths_file, 'wb') as fd:
                pickle.dump(shortest_paths, fd)
            LOGGER.debug('Save completed. Returning shortest paths.')
    return shortest_paths

def graph_at_time_step(graph: nx.MultiDiGraph, time: datetime.datetime, edge_df: pd.DataFrame, copy: bool = True) -> Tuple[nx.MultiDiGraph, pd.DataFrame]:
    """
    Filter the given graph to represent the graph at the given time step.

    Note: It is expected that the input graph was not filtered before! No edges will be added, solely edges removed.
    :param graph: The graph to filter.
    :param time: The time step that the graph should represent after the modification
    :param edge_df: Edge-dataframe of the given graph, which is used to enable selection.
    :param copy: Whether to copy the graph prior to removing edges. Setting this to false will modify the input graph!!!
    :return:
    """
    to_delete = edge_df.loc[np.logical_or(edge_df[EDGE_COLUMN_START_DATE] > time, edge_df[EDGE_COLUMN_END_DATE] < time)]
    if copy:
        graph = graph.copy()
    graph.remove_edges_from(to_delete.index)
    edge_df = edge_df.drop(to_delete.index)
    return graph, edge_df


if __name__ == '__main__':
    challenge_graph, node_df, edge_df = get_graph_attributes()
    print(challenge_graph.name)
    print(challenge_graph)

    print(node_df.describe(include='all'))
    print(node_df.head(10))
    print(edge_df.describe(include='all'))

    #ud_graph = challenge_graph.to_undirected()
    #all_ccs = list(sorted(nx.connected_components(ud_graph), key=len))
    #to_check = len(all_ccs)-1#random.randint(0, len(all_ccs)-100)
    #chosen_cc = all_ccs[to_check]#
    #LOGGER.info('Testing with cc %d of size %d', to_check, len(chosen_cc))
    #nx_eccs = nx.eccentricity(challenge_graph.to_undirected().subgraph(chosen_cc))
    #print('nx eccs:', nx_eccs)
        #print(path_length)
    #ecc_dict = {id: ecc for id, ecc in zip(node_ids, parallel_eccentricity(index_ar, nbor_ar))}
    #LOGGER.info('own eccs computed! ')
    #print(ecc_dict)
    #assert tuple(sorted(nx_eccs.items())) == tuple(sorted(ecc_dict.items())), f'{tuple(sorted(nx_eccs.items()))} does not equal {tuple(sorted(ecc_dict.items()))}'
    re_time = datetime.datetime.fromisoformat('2035-06-01T00:00:00')
    t = time.time()
    new_graph, edge_df = graph_at_time_step(challenge_graph, re_time, edge_df)
    t = time.time() - t
    print('Removal took', t)
    challenge_graph, node_df, edge_df = get_graph_attributes(new_graph)
    print(challenge_graph.name)
    print(challenge_graph)

    print(node_df.describe(include='all'))
    print(node_df.head(10))
    print(edge_df.describe(include='all'))
    node_id = 'Abbott-Gomez'
    print(edge_df.xs(node_id, axis='rows', level=1).head())


