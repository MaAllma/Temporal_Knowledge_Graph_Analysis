import time

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import cycles
from tqdm import tqdm
from data_handling.network import *
from data_handling.network import _f_name_with_opt_prefix

LOGGER = get_logger(__name__)
OUTLIER_FAMILY_RELATION_CYCLE = 'In cycle with Family Relation'
OUTLIER_RELATION_SWITCH = 'Relation switched to other Entities'
def _compute_entities_in_cycles(graph: nx.MultiDiGraph, node_df: pd.DataFrame, edge_df: pd.DataFrame) -> Dict[str, Set[str]]:
    LOGGER.debug('Preparing graph to identify cycles')
    window = pd.Timedelta(value=15, unit='D')
    last_time = edge_df[EDGE_COLUMN_END_DATE].max()
    rel_cycles: List[List[str]] = []
    switch_cycles: List[List[str]] = []
    ud_graph = graph.to_undirected()
    #graph = nx.Graph(ud_graph)
    t = time.time()
    d_to_set = {(name[0], name[1], i):
                {EDGE_COLUMN_START_DATE: edge_df.loc[edge_df.index[pos], EDGE_COLUMN_START_DATE],
                 EDGE_COLUMN_END_DATE: edge_df.loc[edge_df.index[pos], EDGE_COLUMN_END_DATE],
                 EDGE_COLUMN_TYPE: edge_df.loc[edge_df.index[pos], EDGE_COLUMN_TYPE]}
                for name, values in edge_df.groupby([EDGE_INDEX_SOURCE, EDGE_INDEX_TARGET]).groups.items()
                for i, pos in enumerate(values)}
    nx.set_edge_attributes(ud_graph, d_to_set)
    LOGGER.debug('Computing all_cycles')
    all_cycles = list(nx.chordless_cycles(nx.Graph(ud_graph), length_bound=7))
    LOGGER.debug('Starting cycle identification from %d candidates', len(all_cycles))
    for cycle in tqdm(all_cycles):
        sub_graph: nx.MultiGraph = ud_graph.subgraph(cycle)
        family_rel_edges = [k for k, v in nx.get_edge_attributes(sub_graph, EDGE_COLUMN_TYPE).items() if v == 'Family Relation']
        if len(family_rel_edges) > 0 and len(family_rel_edges) < len(cycle):
            rel_cycles.append(cycle)
            continue
        relevant_times = [v for v in nx.get_edge_attributes(sub_graph, EDGE_COLUMN_END_DATE).values() if v < last_time]
        if not relevant_times:
            #if 'Acevedo, Miller and Edwards' in cycle:
            #    LOGGER.debug('Cycle %s contains Acevedo, Miller and Edwards but is rejected since there are no relevant end-date edges %s', cycle, list(nx.get_edge_attributes(sub_graph, EDGE_COLUMN_END_DATE).values()))
            continue
        valid_neighbours: pd.Series = pd.Series(data=([1]) * len(relevant_times), index=sorted(relevant_times)).rolling(window=window).sum()
        pos = valid_neighbours.argmax()
        if valid_neighbours.iloc[pos] >= 2:
            val = valid_neighbours.index[pos]
            lower_bound = val - window
            upper_bound = val + window
            in_bounds_edges = [v for v in nx.get_edge_attributes(sub_graph, EDGE_COLUMN_START_DATE).values() if lower_bound <= v <= upper_bound]
            if len(in_bounds_edges) >= 2:
                switch_cycles.append(cycle)
            #elif 'Acevedo, Miller and Edwards' in cycle:
            #    LOGGER.debug('Cycle %s contains Acevedo, Miller and Edwards but is rejected since there are insufficient edges within the given time bound (%s [%s], %s [%s]): %s %s', cycle, lower_bound, type(lower_bound), upper_bound, type(upper_bound), list(nx.get_edge_attributes(sub_graph, EDGE_COLUMN_START_DATE).values()))
        #elif 'Acevedo, Miller and Edwards' in cycle:
        #    LOGGER.debug('Cycle %s contains Acevedo, Miller and Edwards but is rejected since there are insufficient valid neighbours.', cycle, valid_neighbours)

    LOGGER.debug('Cycle identification completed with %d relevant relation cycles and %d relevant switch cycles.', len(rel_cycles), len(switch_cycles))
    res_entities = {
        OUTLIER_FAMILY_RELATION_CYCLE: set(node for cycle in rel_cycles for node in cycle),
        OUTLIER_RELATION_SWITCH: set(node for cycle in switch_cycles for node in cycle)
    }
    LOGGER.debug('Found %d relation cycle entities and %d switch cycle entities', len(res_entities[OUTLIER_FAMILY_RELATION_CYCLE]), len(res_entities[OUTLIER_RELATION_SWITCH]))
    return res_entities

def entities_in_cycles(graph: nx.MultiDiGraph, node_df: pd.DataFrame, edge_df: pd.DataFrame, prefix: Optional[str] = None, cache: bool = True) -> Dict[str, Set[str]]:
    cycle_entities = path.abspath(_f_name_with_opt_prefix( 'cycle_entities.json', prefix))
    if cache and path.exists(cycle_entities):
        LOGGER.debug('Loading cached entities from %s', cycle_entities)
        with open(cycle_entities, 'r', encoding='utf-16') as fd: #
            cached_entities = json.load(fd)
        LOGGER.debug('Loaded %d cached entities from %s', len(cached_entities), cycle_entities)
        return {k: set(values) for k, values in cached_entities.items()}
    else:
        entities = _compute_entities_in_cycles(graph, node_df, edge_df)
        if cache:
            LOGGER.debug('Caching entities into %s', cycle_entities)
            with open(cycle_entities, 'w', encoding='utf-16') as fd: #
                json.dump({k: list(values) for k, values in entities.items()}, fd)
        return entities



if __name__ == '__main__':
    graph, node_df, edge_df = get_graph_attributes()
    _compute_entities_in_cycles(graph, node_df, edge_df)

