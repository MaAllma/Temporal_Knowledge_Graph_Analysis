import argparse
import dataclasses
import datetime
import math
from collections import deque

import networkx as nx
import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, column
import bokeh.palettes as palettes
from bokeh.plotting import figure, from_networkx
from bokeh.models import MultiLine, Scatter, GraphRenderer, DatetimePicker, ColumnDataSource, DataTable, DateSlider, \
    DatePicker, CustomJS, TableColumn, NumberFormatter, DateFormatter, NumericInput, StaticLayoutProvider, HoverTool, \
    EdgesAndLinkedNodes, NodesOnly, TapTool, Select, AutocompleteInput, GlyphRenderer, Bezier, Range1d, Scroll, \
    WheelZoomTool, Legend, Line, LegendItem, LabelSet, Button, Spacer
from data_handling.network import *
from data_handling.network_changes import *
from utils import get_logger
from interactive_plots.select_time import frequency_time_selection, VALID_KERNELS
from interactive_plots.vis_interesting_nodes_interactive import create_interesting_nodes_tabs
from numba.typed import Dict as TDict
from numba import njit, prange
import numba
import itertools as itert

parser = argparse.ArgumentParser()
args = parser.parse_args()

LOGGER = get_logger(__name__)


g, node_df, edge_df = get_graph_attributes(merge_node_dates=True)
interesting_nodes = entities_in_cycles(g, node_df, edge_df)

ug = g.to_undirected()

LOGGER.debug('Constructed challenge graph')
node_color_map = {
    t: c for t, c in zip(filter(lambda t: t != 'Person' and t != 'Ceo', NODE_TYPE_MAP.values()), palettes.RdPu7)
}
node_color_map.update({
    t: c for t, c in zip(['Person', 'Ceo'], palettes.Greens3)
})
node_df['color'] = node_df[NODE_COLUMN_TYPE].map(node_color_map)
node_df['interesting_node_reason'] = '-'

edge_color_map = {
    t: c for t, c in zip(EDGE_TYPE_MAP.values(), palettes.RdBu4)
}
edge_df['color'] = edge_df[EDGE_COLUMN_TYPE].map(edge_color_map)

edge_df_reset = edge_df.reset_index().rename(columns={EDGE_INDEX_SOURCE: 'start', EDGE_INDEX_TARGET: 'end'})
edge_df_reset = edge_df_reset.sort_values(by=[EDGE_COLUMN_START_DATE, EDGE_COLUMN_END_DATE], ascending=True)

node_df_reset = node_df.reset_index().rename(columns={NODE_INDEX_ID: 'index'})
node_df_reset['degree'] = node_df_reset[NODE_COLUMN_IN_DEGREE] + node_df_reset[NODE_COLUMN_OUT_DEGREE]
node_df_reset[NODE_COLUMN_PRODUCT_SERVICE] = node_df_reset[NODE_COLUMN_PRODUCT_SERVICE].fillna('-')
node_df_reset[NODE_COLUMN_TRADE_DESC] = node_df_reset[NODE_COLUMN_TRADE_DESC].fillna('-')
node_df_reset[NODE_COLUMN_HEAD_OF_ORG] = node_df_reset[NODE_COLUMN_HEAD_OF_ORG].fillna('-')
node_df_reset[NODE_COLUMN_POINT_OF_CONTACT] = node_df_reset[NODE_COLUMN_POINT_OF_CONTACT].fillna('-')


all_nodes_source = ColumnDataSource(node_df_reset)


def default_selection():
    all_nodes_source.selected.indices = [60463]

def last_selection():
    if len(all_nodes_source.selected.indices) >= 1:
        all_nodes_source.selected.indices = [all_nodes_source.selected.indices[-1]]
    else:
        default_selection()

@dataclasses.dataclass
class GraphState:
    cur_time: datetime.datetime
    selection_data_source: ColumnDataSource
    node_data: pd.DataFrame
    edge_data: pd.DataFrame
    last_coords: Dict[str, Tuple[float, float]]
    labeled_nodes: Set[str]
    interesting_nodes: Dict[str, Set[str]]
    all_interesting_nodes: Set[str]
    cur_range: Tuple[int] = tuple(-1 for _ in range(4))
    cur_bounds: Tuple[int] = tuple(-1 for _ in range(4))
    update_listeners: Tuple[Callable[['GraphState'], None], ...] = tuple()
    undirected_graph: nx.MultiGraph = ug
    last_depth: int = -1

    # neighborhood_size not included for re-usability with non-neighborhood plots

    def __post_init__(self):
        LOGGER.debug('Adding selection on change listener')
        def sel_update(attr, old, new):
            LOGGER.debug('Node selection changed to %s', new)
            self.notify_update()

        self.selection_data_source.selected.on_change('indices', sel_update)

    def get_selection_data(self, columns: Optional[Union[str, Iterable[str]]] = None) -> Union[pd.DataFrame, pd.Series]:
        if columns is None:
            return self.node_data.loc[all_nodes_source.selected.indices]
        else:
            return self.node_data.loc[all_nodes_source.selected.indices, columns]

    def on_select_node(self, node_identifier: Union[int, str, Iterable[int], Iterable[str]]):
        if isinstance(node_identifier, (int, str)):
            node_identifier = [node_identifier]
        elif not isinstance(node_identifier, list):
            node_identifier = list(node_identifier)
        if not node_identifier: # attempting to select an empty list...
            LOGGER.warning('Attempt to select an empty list is ignored!')
            return
        if isinstance(node_identifier[0], str):
            candidates: pd.Index = self.node_data.loc[self.node_data['index'].isin(node_identifier)].index
            if len(candidates) > 1:
                LOGGER.warning('Multiple nodes (ids = %s) match identifier %s in the target data. Retaining only the first.', candidates.to_list(), node_identifier)
            elif len(candidates) == 0:
                LOGGER.warning('No node matches identifier %s in the target data. Skipping selection.', node_identifier)
                return
            index = candidates
        else:
            index = node_identifier
        #print(index)
        component_id: pd.Series = self.node_data.loc[index, NODE_COLUMN_CONNECTED_COMPONENT].unique()
        cur_component = cast(pd.Series, self.get_selection_data(NODE_COLUMN_CONNECTED_COMPONENT)).unique()
        if len(cur_component) > 1:
            LOGGER.error('More than one connected component in current selection!!!')
            cur_component = pd.Series(cur_component)
        #print(component_id)
        #print(cur_component)
        # ensure uniqueness, but retain order by using an (ordered) dictionary
        index = list({val: False for val in index}.keys())
        if not component_id.isin(cur_component) or not cur_component.isin(component_id):
            LOGGER.info('Selecting a different connected component (%s instead of %s) by selecting node %s with id %s- Selection is cleared!', component_id, cur_component, node_identifier, index)
            self.selection_data_source.selected.indices = index
        else:
            LOGGER.info('Adding node %s with id(s) %s to selection of connected component %s', node_identifier, index, component_id)
            ls = list(self.selection_data_source.selected.indices)
            ls.extend(index)
            self.selection_data_source.selected.indices = ls

    def on_deselect_node(self, node_identifier: Union[int, str, Iterable[int], Iterable[str]]):
        if isinstance(node_identifier, (int, str)):
            node_identifier = [node_identifier]
        elif not isinstance(node_identifier, list):
            node_identifier = list(node_identifier)
        if isinstance(node_identifier[0], str):
            candidates: pd.Index = self.node_data.loc[self.node_data['index'].isin(node_identifier)].index
            if len(candidates) > 1:
                LOGGER.warning('Multiple nodes (ids = %s) match identifier %s in the target data. Retaining only the first.', candidates.to_list(), node_identifier)
            elif len(candidates) == 0:
                LOGGER.warning('No node matches identifier %s in the target data. Skipping selection.', node_identifier)
                return
            index = candidates
        else:
            index = set(node_identifier)
        new_list = [selected for selected in self.selection_data_source.selected.indices if selected not in index]
        if not new_list:
            LOGGER.error('Attempted to clear selection completely which will reset the selection!!!')
            last_selection()
            #self.selection_data_source.selected.indices = list(self.selection_data_source.selected.indices)
            return
        self.selection_data_source.selected.indices = new_list

    def get_last_selected_node(self) -> str:
        return self.selection_data_source.data['index'][self.selection_data_source.selected.indices[-1]]

    def on_update_time(self, t: datetime.datetime):
        if t != self.cur_time:
            self.cur_time = t
            self.notify_update()

    def notify_update(self):
        for callback in self.update_listeners:
            callback(self)

    def add_time_update_listener(self, callback: Callable[['GraphState'], None]):
        self.update_listeners = self.update_listeners + (callback,)


companies = set(v for v in NODE_TYPE_MAP.values() if v != 'Person' and v != 'Ceo')
all_interesting_nodes = set(node for node_set in interesting_nodes.values() for node in node_set)
labeled_nodes = set(node_df_reset.loc[node_df_reset['index'].isin(all_interesting_nodes) & node_df_reset[NODE_COLUMN_TYPE].isin(companies), 'index'])
neighborhood_size_input = NumericInput(value=5, low=2, title='Neighborhood size to display')
state = GraphState(cur_time=datetime.datetime.fromisoformat('2035-01-01'),
                   selection_data_source=all_nodes_source,
                   node_data=node_df_reset,
                   edge_data=edge_df_reset,
                   last_coords={},
                   labeled_nodes=labeled_nodes,
                   interesting_nodes=interesting_nodes,
                   all_interesting_nodes=all_interesting_nodes)


time_select_layout, date_picker, on_kde_change = frequency_time_selection(node_df,
                                                             edge_df,
                                                             state.on_update_time,
                                                             initial_time=state.cur_time)


graph_renderer = GraphRenderer()

graph_renderer.node_renderer.data_source = ColumnDataSource({c: [] for c in node_df_reset.columns})
graph_renderer.edge_renderer.data_source = ColumnDataSource({c: [] for c in edge_df_reset.columns})

def construct_graph_legend(p: figure, graph_renderer: GraphRenderer) -> Legend:
    dotted_line = Line(x='x', y='y', line_width=4, line_color='black',line_dash='dotted')
    dashed_line = Line(x='x', y='y', line_width=4, line_color='black',line_dash='dashed')
    simple_line = Line(x='x', y='y', line_width=4, line_color='black')

    dotted_renderer = p.add_glyph(dotted_line)
    dashed_renderer = p.add_glyph(dashed_line)
    simple_renderer = p.add_glyph(simple_line)

    items = [
        LegendItem(label='Future Relation', renderers=[dotted_renderer]),
        LegendItem(label='Active Relation', renderers=[simple_renderer]),
        LegendItem(label='Past Relation', renderers=[dashed_renderer])
    ]

    circular_scatter = Scatter(x='x', y='y', size=20, fill_color='black', line_width=0, marker='circle')
    hex_scatter = Scatter(x='x', y='y', size=20, fill_color='black', line_width=0, marker='hex_dot')

    circular_renderer = p.add_glyph(circular_scatter)
    hex_renderer = p.add_glyph(hex_scatter)

    for type, color in edge_color_map.items():
        glyph = Line(x='x', y='y', line_width=4, line_color=color)
        renderer = p.add_glyph(glyph)
        items.append(LegendItem(label=type, renderers=[renderer]))

    items.extend([
        LegendItem(label='Entity', renderers=[circular_renderer]),
        LegendItem(label='Abnormal Entity', renderers=[hex_renderer])
    ])

    for type, color in node_color_map.items():
        glyph = Scatter(x='x', y='y', size=20, fill_color=color, line_width=0)
        renderer = p.add_glyph(glyph)
        items.append(LegendItem(label=type, renderers=[renderer]))

    legend = Legend(
        items=items,
        location='center',
        orientation='vertical',
        title='Symbols'
    )
    for item in legend.items:
        for renderer in item.renderers:
            renderer.visible = False
    return legend

# Adapted from https://docs.bokeh.org/en/latest/docs/user_guide/topics/graph.html#node-and-edge-attributes
def new_graph_plot(graph_renderer: GraphRenderer) -> figure:
    p = figure(width=1000, height=800, #x_range=tuple(state.cur_range[:2]), y_range=tuple(state.cur_range[2:]),
               x_axis_location=None, y_axis_location=None, sizing_mode='stretch_both',
               title="Node Neighbourhood Inspector", background_fill_color="#efefef",
               tools="pan,wheel_zoom,reset") #, x_range=state.cur_range[:2], y_range=state.cur_range[2:])
    hover = HoverTool(line_policy='interp',
                      tooltips=[
                          ('Name', '@index'),
                          ('Type', '@type'),
                          ('First Existence', '@date{%F}'),
                          ('Country of Origin', '@country'),
                          ('Total Neighbours', '@degree'),
                          ('Visible Neighbours', '@visible_degree'),
                          ('Atypical Behaviour', '@interesting_node_reason{safe}')
                      ],
                      formatters={'@date': 'datetime'},
                      # lastly we need to specify the renderer to hover since otherwise bokeh will also hover
                      # the label...
                      renderers=[graph_renderer.node_renderer])
    p.toolbar.active_scroll = next(tool for tool in p.tools if isinstance(tool, Scroll))
    p.add_tools(hover)
    p.add_tools(TapTool())
    p.grid.grid_line_color = None
    p.renderers.append(graph_renderer)
    p.add_layout(construct_graph_legend(p, graph_renderer), 'right')
    used_label_nodes = state.labeled_nodes.copy()
    used_label_nodes.update(state.get_selection_data('index')) # everything that's selected should get a label in order to more easily find it...
    relevant_labels = {node: (x, y) for node, (x, y) in state.last_coords.items() if node in used_label_nodes}
    if relevant_labels:
        print(relevant_labels)
        p.text(x=[x for x, _ in relevant_labels.values()],
               y=[y for _, y in relevant_labels.values()],
               text=list(relevant_labels.keys()),
               x_offset=5,
               y_offset=5,
               anchor='bottom_left')

    if any(map(lambda val: val != -1, state.cur_range)):
        p.x_range = Range1d(*state.cur_range[:2])
        p.y_range = Range1d(*state.cur_range[2:])
    return p

graph_layout = row(new_graph_plot(graph_renderer), sizing_mode='stretch_both')


def _spring_layout_algorithm(pos: np.ndarray, neighbourhood: TDict, k: float = 0.1, threshold: float = 1e-4, iterations = 100) -> Tuple[np.ndarray, int]:
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / (iterations + 1)

    #displacement = np.zeros((pos.shape[0], pos.shape[1]))
    delta_pos = np.empty(pos.shape[0], dtype=pos.dtype)
    converged = iterations <= 0
    converge_counter = 0
    #print('Converge loop')
    while not converged:
        #print('Not converged with counter', converge_counter, 'target iterations',int(iterations * np.log2(converge_counter + 2)))
        for iteration in range(int(iterations * np.log2(converge_counter + 2))):#iterations):#
            # loop over rows
            for i in prange(pos.shape[0]):
                #if i in fixed:
                #    continue
                # difference between this row's node position and all others
                delta = (pos[i] - pos)
                # distance between points
                distance = np.sqrt((delta ** 2).sum(axis=1))
                # enforce minimum distance of 0.01
                distance = np.where(distance < 0.01, 0.01, distance)
                # networkx here gets the adjacency matrix row and uses that for a mask

                Ai = np.zeros(pos.shape[0], dtype=np.int32)
                n_ar = neighbourhood[i]
                for j in n_ar:
                    Ai[j] += 1
                # we could instead directly set only the forces we need, but it doesn't really make a difference
                # attracting_force = np.zeros(pos.shape[0], dtype=np.int32)
                # n_ar = neighbourhood[i]
                # for j in n_ar:
                #     attracting_force[j] += distance[j] / k

                # displacement "force"
                f = delta.T * (k * k / distance ** 2 - Ai * distance / k) # - attracting_force
                displacement = f.sum(axis=1)
                # update positions
                length = np.linalg.norm(displacement) #np.sqrt((displacement ** 2).sum(axis=1))
                if length < 0.01:
                    length = 0.01
                #length = np.where(length < 0.01, 0.01, length)
                delta_pos_local = displacement * t / length
                pos[i, :] += delta_pos_local
                delta_pos[i] = np.max(delta_pos_local)
            # cool temperature
            t -= dt
            converged = (np.max(delta_pos) / pos.shape[0]) < threshold
            if converged:
                break
        converge_counter += 1
    return pos, converge_counter

_spring_layout_accelerated = njit(parallel=False)(_spring_layout_algorithm)
#_spring_layout_parallel = njit(parallel=True)(_spring_layout_algorithm)

def compute_spring_layout(graph: nx.MultiGraph, last_coords: Dict[str, Tuple[float, float]], k=0.1, seed=42) -> Dict[str, Tuple[float, float]]:
    node_id_map = {node: i for i, node in enumerate(last_coords.keys()) if node in graph.nodes}
    available_nodes = len(node_id_map)
    counter = available_nodes
    for node in graph.nodes:
        if node not in node_id_map:
            node_id_map[node] = counter
            counter += 1

    if len(node_id_map) == 0:
        return {}
    elif len(node_id_map) == 1:
        return {node: (0.0, 0.0) for node in node_id_map.keys()}

    neighbourhood = TDict([
        (node_id_map[node], np.array([node_id_map[other_node] for _, other_node in graph.edges(node)]))
        for node in node_id_map.keys()
    ])
    node_pos = np.array([(list(last_coords[node]) if node in last_coords else [0, 0])for node in node_id_map.keys()], dtype=np.float32)
    node_flags = np.full(node_pos.shape[0], True)
    if available_nodes < node_flags.shape[0]:
        node_flags[available_nodes:] = False
    node_flags[0] = True # if nothing else, the first node will be in the center... :shrug:

    numba.set_num_threads(4)
    gen = np.random.default_rng(seed)
    nodes = deque([next(iter(node_id_map.keys()))])
    visited = set()
    while nodes:
        next_node = nodes.popleft()
        next_node_id = node_id_map[next_node]
        visited.add(next_node)
        if not node_flags[next_node_id]:
            neighbor_indices = np.array([node_id_map[node] for node in graph.neighbors(next_node)])
            relevant_pos = (node_pos[neighbor_indices])[node_flags[neighbor_indices]]
            #print(relevant_pos)
            random_positions: np.ndarray = gen.normal(loc=relevant_pos, scale=k)
            #print(random_positions)
            #print('-----------')
            node_pos[next_node_id] = np.mean(random_positions, axis=0)
            node_flags[next_node_id] = True

        nodes.extend(n for n in graph.neighbors(next_node) if n not in visited)

    LOGGER.debug('Calling numba accelerated spring layout computation.')
    res, converge_iter = _spring_layout_accelerated(node_pos, neighbourhood, k)
    LOGGER.debug('Spring layout computation completed in %d iterations.', converge_iter)


    return cast(Dict[str, Tuple[float, float]], {node: tuple(res[i].tolist()) for node, i in node_id_map.items()})


BEND_STRENGTH = 1/6
NODE_THRESHOLD = 1500
MIN_DEPTH = 3
def update_graph():
    def find_relevant_nodes(sel_set: Set[str], d: int, last_d: int) -> Tuple[Set[str], int]:
        if last_d < MIN_DEPTH:
            #print('Last depth was', last_d)
            last_d = d
        #note that this assumes that all nodes are automatically in the same connected component
        all_nodes: Set[str] = set()
        LOGGER.info('Graph is updated with source nodes %s are and neighborhood size %d at time step %s and last '
                    'recorded neighbourhood size was %d', sel_set, d, str(t), last_d)
        selection_map = {}
        if sel_set:  # when not working with the null_graph...
            to_test = list(sel_set)
            #print(to_test)
            while to_test:
                next_node = to_test.pop()
                #print(to_test)
                for other_node in to_test:
                    shortest_paths = nx.all_shortest_paths(state.undirected_graph, source=next_node, target=other_node)
                    #LOGGER.debug('Adding nodes on path between "%s" and "%s": %s', next_node, other_node, shortest_paths)
                    selection_map.update({
                        node: max(min(d // 2, last_d), MIN_DEPTH) for node in set(itert.chain.from_iterable(shortest_paths))
                    })
        max_d = d
        d = min(d, last_d)
        selection_map.update({node: d for node in sel_set})
        last_was_decrease = False
        #LOGGER.debug('Running search with selection map %s', selection_map)
        while True:
            for node, depth in selection_map.items():
                tree: nx.DiGraph = nx.bfs_tree(state.undirected_graph, node, depth_limit=depth)
                all_nodes.update(cast(Iterable[str], tree.nodes))

            if len(all_nodes) < NODE_THRESHOLD or d <= MIN_DEPTH:
                if d < max_d and not last_was_decrease:
                    d += 1
                    selection_map = {n: max(nd, (d if n in sel_set else d//2)) for n, nd in selection_map.items()}
                    LOGGER.debug('Increased search depth by 1 from %d to %d since %d nodes were discovered and depth < %d.', d-1, d, len(all_nodes), max_d)
                else:
                    break
            else:
                d -= 1
                selection_map = {n: min(nd, (d if n in sel_set else d//2)) for n, nd in selection_map.items()}
                last_was_decrease = True
                LOGGER.debug('Reduced search depth by 1 from %d to %d since %d nodes were discovered.', d+1, d, len(all_nodes))
            all_nodes.clear()
        return all_nodes, d

    def compute_bezier_control_points(node_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # get the lines between the nodes and the corresponding magnitudes
        vectors = node_coords[1] - node_coords[0]
        magnitudes = np.linalg.norm(vectors, axis=1)
        # construct orthogonal vectors by solving a^Tx = 0 and a^Ta = x^Tx
        # aka find the orthogonal vector with the same magnitude as the original vector
        relation = (vectors[:, 1] / vectors[:, 0])
        a_2 = magnitudes / np.sqrt(1 + relation ** 2)
        a_1 = - a_2 * relation
        a = np.array([a_1, a_2]).T
        x_axis_aligned_mask = np.isclose(vectors[:, 0], 0)
        if np.any(x_axis_aligned_mask):
            n_axis_aligned = np.sum(x_axis_aligned_mask)
            LOGGER.debug('Computed bézier control points in presence of %d x-axis parallel vectors', n_axis_aligned)
            a[x_axis_aligned_mask, :] = np.array([np.zeros(n_axis_aligned), magnitudes[x_axis_aligned_mask]]).T
        y_axis_algined_mask = np.isclose(vectors[:, 1], 0)
        if np.any(y_axis_algined_mask):
            n_axis_aligned = np.sum(y_axis_algined_mask)
            LOGGER.debug('Computed bézier control points in presence of %d x-axis parallel vectors', n_axis_aligned)
            a[y_axis_algined_mask, :] = np.array([magnitudes[y_axis_algined_mask], np.zeros(n_axis_aligned)]).T
        if np.any(x_axis_aligned_mask & y_axis_algined_mask):
            n_axis_aligned = np.sum(x_axis_aligned_mask & y_axis_algined_mask)
            LOGGER.warning('Found %d vectors where each direction has a magnitude < 1e-8 and which where thus '
                           'determined axis-aligned. This will most likely lead to incorrect rendering and is '
                           'presumably caused by the layout not converging!', n_axis_aligned)
        multi_edges = [(i, relevant_edges['start'].iloc[i], relevant_edges['end'].iloc[i]) for i, meid in
                       enumerate(relevant_edges[EDGE_COLUMN_MULTI_EDGE_ID]) if meid > 0]
        multi_edge_dict = {}
        for i, start, end in multi_edges:
            ls = multi_edge_dict.setdefault((start, end), [])
            ls.append(i)
        mask = np.zeros_like(a)
        for ls in multi_edge_dict.values():
            # multiply by 25% in order to bend it by "25%" (BEND_STRENGTH == 0.25)
            if len(ls) > 3:
                LOGGER.warning('More than 3 Multi Edges detected!!!')
            for bend_index, i in enumerate(ls):
                if bend_index > 1:
                    break
                mask[i] = BEND_STRENGTH * ((-1) ** bend_index)

        a = mask * a
        c0 = node_coords[0] + 1 / 3 * vectors + a
        c1 = node_coords[0] + 2 / 3 * vectors + a
        return c0, c1


    d = neighborhood_size_input.value
    # division by 1000 as per https://stackoverflow.com/questions/68583446/bokeh-update-map-based-on-dates-filtered-via-dateslider
    # yields the correct result... Javascript works with milliseconds whereas python interprets timestamps in seconds
    t = state.cur_time
    node_data = state.node_data
    edge_data = state.edge_data
    last_coords = state.last_coords
    sel_data = state.get_selection_data('index')
    sel_set = set(state.get_selection_data('index'))

    all_nodes, state.last_depth = find_relevant_nodes(sel_set, d, state.last_depth)

    sub_graph: nx.MultiGraph = state.undirected_graph.subgraph(all_nodes)

    # the value 0.0010515 originates from the approximate slope of log2(log2(x)) at x=250
    # hence increasing the amount nodes and edges are made smaller by that results in rather continous transition to
    # this much slower growing function at that point
    # The value was obtained with help from wolframalpha...

    # Different approach scale by sqrt(ax) until x = 266 to a reduction of 2 and then scale with log2(log2(x - 250)) - 1
    # For this to work we must have a = 2/133 and to make it not scale to quickly start at x = 133/2
    if len(all_nodes) <= 258:
        node_size_red = 1.0
    else:
        node_size_red = math.log2(math.log2(len(all_nodes) - 242)) - 1
    if len(all_nodes) <= 258 / 2:
        edge_size_red = 1.0
    elif len(all_nodes) <= 516:
        edge_size_red = math.sqrt(len(all_nodes) * 2.0 / 258.0 )
    elif len(all_nodes) > 516:
        edge_size_red = math.log2(math.log2(len(all_nodes) - 500.0))
    else:
        edge_size_red = 1.0 # should never be reached, but makes LINT shut up
    LOGGER.debug('Slicing Graph since BFS completed with %d nodes resulting in graph for which is_connected()==%s '
                 'and which will divide size by a factor of %.3f for nodes and %.3f for edges',
                 len(all_nodes), nx.is_connected(state.undirected_graph.subgraph(all_nodes)), node_size_red, edge_size_red)
    relevant_nodes = node_data.loc[node_data['index'].isin(all_nodes)].copy()
    relevant_nodes['visible_degree'] = relevant_nodes['index'].map(lambda node: sub_graph.degree[node])
    relevant_nodes.index = pd.RangeIndex(0, len(relevant_nodes), step=1)
    relevant_nodes['line_width'] = 2.0 * (relevant_nodes['degree'] != relevant_nodes['visible_degree'])
    #relevant_nodes['alpha'] = (relevant_nodes['date'] <= t) * 0.5 + 0.5
    relevant_nodes['select_line_width'] = 4.0 / node_size_red
    relevant_nodes['hover_line_width'] = 4.0 / node_size_red # relevant_nodes['index'].map(lambda node: (4.0 if node in sel_set else 2.0))
    relevant_nodes['markers'] = relevant_nodes['index'].map(lambda node: 'hex_dot' if node in state.all_interesting_nodes else 'circle')
    relevant_edges = edge_data.loc[np.logical_and(edge_data['start'].isin(all_nodes),
                                                  edge_data['end'].isin(all_nodes))].copy()
    def map_edge_style(row: pd.Series) -> str:
        start_date = row[EDGE_COLUMN_START_DATE]
        end_date = row[EDGE_COLUMN_END_DATE]

        if start_date > t:
            style= 'dotted'
        elif end_date < t:
            style = 'dashed'
        else:
            style = 'solid'
        return style

    if len(relevant_edges) >= 1:
        relevant_edges['line_dash'] = relevant_edges[[EDGE_COLUMN_START_DATE, EDGE_COLUMN_END_DATE]].apply(map_edge_style, axis='columns')
    else:
        relevant_edges['line_dash'] = 'solid'
    # Idea for alpha calculation: use that fact that booleans evaluate to 1 and 0 respectively and add a base alpha of 0.5 to ensure inactive edges are visible
    #relevant_edges['alpha'] = np.minimum(((relevant_edges[EDGE_COLUMN_START_DATE] <= t) & (relevant_edges[EDGE_COLUMN_END_DATE] >= t)) + 0.5, 1.0)


    relevant_node_data = {c: [('' if is_na else v) for v, is_na in zip(relevant_nodes[c], relevant_nodes[c].isna())]
                          for c in relevant_nodes.columns}
    relevant_edge_data = {c: [('' if is_na else v) for v, is_na in zip(relevant_edges[c], relevant_edges[c].isna())]
                          for c in relevant_edges.columns}
    #LOGGER.debug('Dumping node_data %s', relevant_node_data)
    #LOGGER.debug('Dumping edge_data %s', relevant_edge_data)
    relevant_node_data = ColumnDataSource(relevant_node_data)
    relevant_node_data.selected.indices = relevant_nodes.index[relevant_nodes['index'].isin(sel_set)]

    def on_node_selected(attr, old: Iterable[int], new: Iterable[int]):
        old = set(old)
        new = set(new)
        if old == new:
            LOGGER.error('Selection callback called, but selection %s = %s did not change', old, new)
            return
        if old.issubset(new):
            new_nodes = list(new - old)
            new_nodes = relevant_nodes['index'].iloc[new_nodes].to_list()
            LOGGER.debug('User selected node %s in graph-view.', new_nodes)
            state.on_select_node(new_nodes)
        else:
            deselected_nodes = list(old - new)
            deselected_nodes = relevant_nodes['index'].iloc[deselected_nodes].to_list()
            LOGGER.debug('Unselected node %s ', deselected_nodes)
            state.on_deselect_node(deselected_nodes)

    # links node selection to our master selection list that will trigger re-drawing the graph
    relevant_node_data.selected.on_change('indices', on_node_selected)

    graph_renderer = GraphRenderer()
    graph_renderer.node_renderer.data_source = relevant_node_data

    LOGGER.debug('Slicing completed, creating layout.')
    # TODO push non-overlapping graphs nicer together...
    key_set = set(last_coords.keys())
    if last_coords and all_nodes == key_set:
        LOGGER.debug('Graph did not change - keeping old layout')
        needs_comp = False
    elif last_coords:
        LOGGER.debug('Graph changed - computing new spring layout')
        needs_comp = True
    else:
        LOGGER.debug('Graph not computed yet - computing spring layout')
        needs_comp = True

    if needs_comp:
        last_coords.clear()
        computed = compute_spring_layout(sub_graph, last_coords)#nx.spring_layout(sub_graph, center=(0, 0),  pos=pos, iterations=min(len(all_nodes) ** 2, 1000), k=0.1)
        last_coords.update({k: tuple(v) for k, v in computed.items()})
        LOGGER.debug('Spring layout computed with size %d', len(last_coords))
        on_kde_change(set(last_coords.keys()))
        LOGGER.debug('Updated kde-plot.')

    #print(all_layout_coords)
    layout = StaticLayoutProvider(graph_layout=last_coords)
    graph_renderer.layout_provider = layout
    graph_renderer.node_renderer.glyph = Scatter(size=20 / node_size_red, fill_color='color', line_width='line_width', line_color='grey', marker='markers')#, line_alpha='alpha', fill_alpha='alpha')
    graph_renderer.node_renderer.hover_glyph = Scatter(size=25 / node_size_red, fill_color='color', line_width='hover_line_width', line_color='grey', marker='markers')#, line_alpha='alpha', fill_alpha='alpha')
    graph_renderer.node_renderer.selection_glyph = Scatter(size=25 / node_size_red, fill_color='color', line_width='select_line_width', line_color='black', marker='markers')#, line_alpha='alpha', fill_alpha='alpha')
    graph_renderer.node_renderer.nonselection_glyph = graph_renderer.node_renderer.glyph

    # now setup Bézier curves...
    # start with creating an array of shape (2, |V|, d) where d=2 since where are in 2D space

    node_coords = np.array([
        [list(last_coords[k]) for k in relevant_edge_data[key]] for key in ['start', 'end']
    ])
    c0, c1 = compute_bezier_control_points(node_coords)

    bezier_data = dict(
        x0 = node_coords[0, :, 0],
        cx0 = c0[:, 0],
        cx1 = c1[:, 0],
        x1 = node_coords[1, :, 0],

        y0=node_coords[0, :, 1],
        cy0=c0[:, 1],
        cy1=c1[:, 1],
        y1=node_coords[1, :, 1],
    )
    relevant_edge_data.update(bezier_data)
    graph_renderer.edge_renderer = GlyphRenderer(
        glyph=Bezier(line_dash='line_dash', line_color='color', line_width=4 / edge_size_red),#, line_alpha='alpha'),
        data_source=ColumnDataSource(relevant_edge_data)
    )
    #graph_renderer.edge_renderer.glyph = MultiLine(line_dash='line_dash', line_alpha='alpha', line_color='color', line_width=4)
    graph_renderer.selection_policy = NodesOnly()
    graph_renderer.inspection_policy = NodesOnly()
    # bokeh display tooltip on graph edge see also https://gist.github.com/canavandl/8cb5ecece6ba720d09c0d1aef1666642
    #graph_renderer.inspection_policy = EdgesAndLinkedNodes()

    LOGGER.debug('Layout created. Updating figure.')
    # remember zoom...
    x_min = node_coords[..., 0].min()
    x_max = node_coords[..., 0].max()
    y_min = node_coords[..., 1].min()
    y_max = node_coords[..., 1].max()
    state.cur_bounds = (
        x_min - 0.05 * (x_max - x_min),
        x_max + 0.05 * (x_max - x_min),
        y_min - 0.05 * (y_max - y_min),
        y_max + 0.05 * (y_max - y_min),
    )
    if any(map(lambda val: val != -1, state.cur_range)) and (all_nodes.issubset(key_set) or key_set.issubset(all_nodes)):
        state.cur_range = (
            graph_layout.children[-1].x_range.start,
            graph_layout.children[-1].x_range.end,
            graph_layout.children[-1].y_range.start,
            graph_layout.children[-1].y_range.end
        )
    else:
        state.cur_range = state.cur_bounds
    #state.cur_bounds = state.cur_range
    graph_layout.children[-1] = new_graph_plot(graph_renderer)
    LOGGER.debug('Figure updated - graph update complete.')

DEFAULT_BANDWIDTH = 15
DEFAULT_KERNEL = 'gaussian'
neighborhood_size_input.on_change('value', lambda attr, old, new: state.notify_update())
#bandwidth_selector = NumericInput(value=DEFAULT_BANDWIDTH, low=1, high=365, title='Select Bandwidth in days')
#kernel_selector = Select(title='Select Kernel', value=DEFAULT_KERNEL, options=VALID_KERNELS)
auto_complete_input = AutocompleteInput(title="Name to select", completions=state.node_data['index'].to_list(), search_strategy="includes", case_sensitive=False)
reset_button = Button(label="Reset", button_type="default")
set_to_last_button = Button(label="Focus on Last", button_type="default")
LOGGER.debug('Wilson Grant is in completions: %s', 'Wilson Grant' in state.node_data['index'].to_list())
def autocomplete_callback(attr, old, new):
    LOGGER.debug('Autocomplete callback triggered with old = %s and new = %s', old, new)
    # if np.any(state.node_data['index'] == old):
    #     index_to_remove = state.node_data.loc[state.node_data['index'].isin([old])].index.to_numpy()[0]
    #     LOGGER.debug('Removing old with index value %d', index_to_remove)
    #     selection_indices = [i for i in state.selection_data_source.selected.indices if i != index_to_remove]
    # else:
    #     selection_indices = list(state.selection_data_source.selected.indices)
    #
    # if np.any(state.node_data['index'] == new):
    #     index_to_add = state.node_data.loc[state.node_data['index'].isin([new])].index.to_numpy()[0]
    #     LOGGER.debug('Adding new with index value %d', index_to_add)
    #     selection_indices.append(index_to_add)
    # state.selection_data_source.selected.indices = selection_indices
    state.on_select_node(new)



auto_complete_input.on_change('value', autocomplete_callback)

control_layout = column(neighborhood_size_input, auto_complete_input, date_picker, row(reset_button, set_to_last_button)) # bandwidth_selector, kernel_selector,
#bandwidth_selector.on_change('value', lambda attr, old, new: on_kde_change(bandwidth_selector.value, kernel_selector.value))
#kernel_selector.on_change('value', lambda attr, old, new: on_kde_change(bandwidth_selector.value, kernel_selector.value))

def create_table() -> Tuple[DataTable, Callable[[List[str]], None]]:
    def get_component() -> int:
        unique_indices: np.ndarray = state.get_selection_data(NODE_COLUMN_CONNECTED_COMPONENT).unique()
        if unique_indices.shape[0] == 0:
            return 0
        return int(unique_indices[0])

    def get_component_companies(component: Optional[int] = None) -> pd.DataFrame:
        component = get_component() if component is None else component
        component_data = state.node_data.loc[state.node_data[NODE_COLUMN_CONNECTED_COMPONENT] == component]
        component_data = component_data.loc[component_data[NODE_COLUMN_TYPE].isin(set(NODE_TYPE_MAP.values()) - {'Person', 'Ceo'})]
        LOGGER.debug('Component is %d with %d companies', component, len(component_data))
        return component_data

    columns = [
        # see https://stackoverflow.com/questions/50996875/how-to-color-rows-and-or-cells-in-a-bokeh-datatable for inspiration
        TableColumn(field='index', title='Name', width=150),
        TableColumn(field=NODE_COLUMN_TYPE, title='Type', width=75),
        TableColumn(field=NODE_COLUMN_GENERIC_DATE, title='Founding Date', width=75, formatter=DateFormatter()),
        TableColumn(field=NODE_COLUMN_COUNTRY, title='Country', width=75),
        TableColumn(field=NODE_COLUMN_POINT_OF_CONTACT, title='Contact', width=75),
        TableColumn(field=NODE_COLUMN_HEAD_OF_ORG, title='Head of Org.', width=75),
        TableColumn(field='degree', title='#Neighbours', width=50, formatter=NumberFormatter()),
        TableColumn(field=NODE_COLUMN_REVENUE, title='Revenue', width=50, formatter=NumberFormatter()),
        TableColumn(field=NODE_COLUMN_PRODUCT_SERVICE, title='Product', width=500),
        TableColumn(field=NODE_COLUMN_TRADE_DESC, title='Trade', width=1000)
    ]

    visible_companies_source = ColumnDataSource(get_component_companies())
    cur_component = [get_component()]

    def on_company_selection_changed(attr, old, new):
        old = [visible_companies_source.data['index'][index] for index in old if 0 <= index < len(visible_companies_source.data['index'])]
        new = [visible_companies_source.data['index'][index] for index in new if 0 <= index < len(visible_companies_source.data['index'])]
        on_interesting_node_selected(old, new)

    def on_external_selection_changed(new_nodes: List[str]):
        new_nodes = set(new_nodes)
        component = get_component()
        if component not in cur_component:
            visible_companies_source.data = get_component_companies(component) #.to_dict(orient='list')
            cur_component[0] = component
        new_index_list = [i for i, node in enumerate(visible_companies_source.data['index']) if node in new_nodes]
        if set(visible_companies_source.selected.indices) != set(new_index_list):
            visible_companies_source.selected.indices = new_index_list

    on_external_selection_changed(state.get_selection_data('index').to_list())
    visible_companies_source.selected.on_change('indices', on_company_selection_changed)

    node_table = DataTable(source=visible_companies_source, columns=columns, frozen_columns=1, height=150, width=2000, sizing_mode = 'stretch_width', selectable="checkbox")


    return node_table, on_external_selection_changed

reset_button.on_click(default_selection)
set_to_last_button.on_click(last_selection)

#node_table, on_external_table_update = create_table()

LOGGER.debug('Added renderer.')
def on_interesting_node_selected(old: List[str], new: List[str]):
    old_set = set(old)
    new_set = set(new)
    if old_set == new_set:
        LOGGER.debug('Discarding interesting node update since old == new')
        return
    deselection_set = old_set - new_set
    selection_set = new_set - old_set
    if deselection_set:
        LOGGER.debug('Removing nodes based on interesting_node_selection interaction since old - new = %s', deselection_set)
        state.on_deselect_node([node for node in old if node in deselection_set])
    if selection_set:
        LOGGER.debug('Adding nodes based on interesting_node_selection interaction since new - old  = %s', selection_set)
        state.on_select_node([node for node in new if node in selection_set])

tabs, interesting_nodes_callback, to_label, interesting_nodes = create_interesting_nodes_tabs(g, node_df, edge_df, on_interesting_node_selected, state.interesting_nodes)
def on_propagate_node_clicked(attr, old, new):
    LOGGER.debug('on_propagate_node_clicked("%s", %s, %s)', attr, old, new)
    if old != new:
        nodes = state.get_selection_data('index').to_list()
        interesting_nodes_callback(nodes, state.cur_time, False)
        #on_external_table_update(nodes)

state.labeled_nodes.update(to_label)
for key, values in interesting_nodes.items():
    state.interesting_nodes.setdefault(key, set()).update(values)
state.selection_data_source.selected.on_change('indices', on_propagate_node_clicked)
state.add_time_update_listener(lambda s: update_graph())
state.all_interesting_nodes = set(node for node_set in state.interesting_nodes.values() for node in node_set)
interesting_nodes_reverted = {
    node: {key for key, value_set in state.interesting_nodes.items() if node in value_set} for node in state.all_interesting_nodes
}
state.node_data['interesting_node_reason'] = state.node_data['index'].map(lambda node: '-' if node not in interesting_nodes_reverted else '<br>'.join(interesting_nodes_reverted[node]))
default_selection()
curdoc().add_root(column(Spacer(width=10, height=60, sizing_mode='stretch_width'), row(control_layout, time_select_layout, sizing_mode='stretch_width'), row(graph_layout, tabs, sizing_mode='stretch_width'), sizing_mode='stretch_width')) # , node_table
# bokeh serve bokeh_dashboard.py --show