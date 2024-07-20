from itertools import product

import pandas as pd
from bokeh import palettes
from bokeh.layouts import gridplot, layout, row, grid
from bokeh.models import (ColumnDataSource, DataRange1d, Quad, HoverTool, LayoutDOM)
from bokeh.models import (NumeralTickFormatter, Scatter)
from bokeh.plotting import figure, show

from data_handling.network import *

SIZING_MODE = 'stretch_both'
LOGGER = get_logger(__name__)

update_in_progress_flag = False
def create_scatterplot_matrix(graph, node_df, edge_df, on_notify_selection_changed: Callable[[List[str], List[str]], None], known_interesting_nodes: Dict[str, Set[str]], company_filter: bool) -> Tuple[LayoutDOM, Callable[[Set[str]], None]]:
    LOGGER.info('Creating Scatterplot-Matrix with company_filter = %s', company_filter)

    def aggregate_targets(group):
        return pd.Series({
            'target_works_for_counter': (group[EDGE_COLUMN_TYPE] == 'Works for').sum(),
            'target_shareholder_counter': (group[EDGE_COLUMN_TYPE] == 'Shareholder').sum(),
            'target_beneficial_ownership_counter': (group[EDGE_COLUMN_TYPE] == 'Beneficial Ownership').sum(),
            'target_family_relation_counter': (group[EDGE_COLUMN_TYPE] == 'Family Relation').sum()
        })

    def aggregate_sources(group):
        return pd.Series({
            'source_works_for_counter': pd.to_numeric((group[EDGE_COLUMN_TYPE] == 'Works for').sum()),
            'source_shareholder_counter': pd.to_numeric((group[EDGE_COLUMN_TYPE] == 'Shareholder').sum()),
            'source_beneficial_ownership_counter': pd.to_numeric(
                (group[EDGE_COLUMN_TYPE] == 'Beneficial Ownership').sum()),
            'source_family_relation_counter': pd.to_numeric((group[EDGE_COLUMN_TYPE] == 'Family Relation').sum())
        })

    target_counter_series = edge_df.groupby(
        [EDGE_INDEX_TARGET, EDGE_COLUMN_TYPE]).size()  #.apply(aggregate_targets, include_groups=False)
    source_counter_series = edge_df.groupby(
        [EDGE_INDEX_SOURCE, EDGE_COLUMN_TYPE]).size()  #.apply(aggregate_sources, include_groups=False)

    target_counter_df = target_counter_series.unstack().rename(columns={
        'Works for': 'target_works_for_counter',
        'Shareholder': 'target_shareholder_counter',
        'Beneficial Ownership': 'target_beneficial_ownership_counter',
        'Family Relation': 'target_family_relation_counter'
    })
    source_counter_df = source_counter_series.unstack().rename(columns={
        'Works for': 'source_works_for_counter',
        'Shareholder': 'source_shareholder_counter',
        'Beneficial Ownership': 'source_beneficial_ownership_counter',
        'Family Relation': 'source_family_relation_counter'
    })

    LOGGER.debug('Merging computed attributes.')
    #print(target_counter_df.describe())
    #print(target_counter_df.head(10))
    #print(source_counter_df.describe())
    #print(source_counter_df.head(10))
    edge_counter_df = node_df.merge(target_counter_df, how='left', right_index=True, left_index=True)
    edge_counter_df = edge_counter_df.merge(source_counter_df, how='left', right_index=True, left_index=True)

    fill_values = {
        'target_works_for_counter': 0,
        'target_shareholder_counter': 0,
        'target_beneficial_ownership_counter': 0,
        'target_family_relation_counter': 0,
        'source_works_for_counter': 0,
        'source_shareholder_counter': 0,
        'source_beneficial_ownership_counter': 0,
        'source_family_relation_counter': 0
    }

    edge_counter_df.fillna(value=fill_values, inplace=True)

    persons = ['Person', 'Ceo']
    companies = [company_type for company_type in NODE_TYPE_MAP.values() if company_type not in persons]

    edge_color_map = {
        t: c for t, c in zip(EDGE_TYPE_MAP.values(), palettes.RdBu4)
    }


    if company_filter:
        company_df = edge_counter_df.loc[edge_counter_df[NODE_COLUMN_TYPE].isin(companies)]
        df = company_df.copy()

        attributes = (
            NODE_COLUMN_IN_DEGREE,
            NODE_COLUMN_REVENUE,
            'target_works_for_counter',
            'target_shareholder_counter',
            'target_beneficial_ownership_counter',
            'source_shareholder_counter')

        attribute_labels = {
            NODE_COLUMN_IN_DEGREE: {"label": "In-Relation", "color": "#555555"},
            NODE_COLUMN_REVENUE: {"label": "Revenue", "color": "#777777"},
            'target_works_for_counter': {"label": "Worker", "color": edge_color_map["Works for"]},
            'target_shareholder_counter': {"label": "has Shareholder", "color": edge_color_map["Shareholder"]},
            'target_beneficial_ownership_counter': {"label": "Owner", "color": edge_color_map["Beneficial Ownership"]},
            'source_shareholder_counter': {"label": "is Shareholder", "color": edge_color_map["Shareholder"]}
        }
    else:
        person_df = edge_counter_df.loc[edge_counter_df[NODE_COLUMN_TYPE].isin(persons)]
        df = person_df.copy()

        attributes = (
            NODE_COLUMN_OUT_DEGREE,
            'source_works_for_counter',
            'source_shareholder_counter',
            'source_beneficial_ownership_counter',
            'source_family_relation_counter')

        attribute_labels = {
            NODE_COLUMN_OUT_DEGREE: {"label": "Out-Relation", "color": "#555555"},
            'source_works_for_counter': {"label": "Works For", "color": edge_color_map["Works for"]},
            'source_shareholder_counter': {"label": "Shareholder", "color": edge_color_map["Shareholder"]},
            'source_beneficial_ownership_counter': {"label": "Owner", "color": edge_color_map["Beneficial Ownership"]},
            'source_family_relation_counter': {"label": "Family Relation", "color": edge_color_map["Family Relation"]}
        }

    N = len(attributes)
    LOGGER.debug('Computed Dataframe with %d attributes and columns %s', N, list(df.columns))
    # print(df.index.to_list()[:100])
    intrest_nodes = {node for node_set in known_interesting_nodes.values() for node in node_set}
    df['marker'] = ['hex_dot' if node in intrest_nodes else 'circle' for node in df.index]
    source = ColumnDataSource(data=df)

    # Calculate histogram data for each attribute
    hist_data = {}
    for attr in attributes:
        hist, edges = np.histogram(df[attr].dropna(), bins=20)
        hist_data[attr] = ColumnDataSource(
            data={'top': hist, 'left': edges[:-1], 'right': edges[1:], 'bottom': np.zeros_like(hist)})

    LOGGER.debug('Computed histograms for %d attributes', N)

    xdrs = [DataRange1d(bounds=None) for _ in range(N)]
    ydrs = [DataRange1d(bounds=None) for _ in range(N)]
    hover = HoverTool(tooltips=[(('Company' if company_filter else 'Person'), '@id')])

    def create_scatter_plot(x, y, i, N, source):

        p = figure(width=200, height=200, x_range=xdrs[i % N], y_range=ydrs[i // N],
                   background_fill_color="#fafafa", border_fill_color="white", min_border=5, sizing_mode='inherit',
                   tools="tap", toolbar_location=None
                   )

        if (i % N == 0) or (i == N * (N - 1) + 1):  # first column
            p.min_border_left = p.min_border + 4
            p.width += 40
            if i % N == 0:
                p.yaxis.axis_label = attribute_labels[y]["label"]
            # p.yaxis.major_label_orientation = "vertical"
            p.yaxis.visible = True
        else:
            p.yaxis.visible = False

        if i >= N * (N - 1):  # last row
            p.min_border_bottom = p.min_border + 40
            p.height += 40
            p.xaxis.axis_label = attribute_labels[x]["label"]
            p.xaxis.visible = True
        else:
            p.xaxis.visible = False

        scatter = Scatter(x=x, y=y, fill_alpha=0.6, size=5, line_color=None,
                          fill_color='color', marker='marker'
                          )
        r = p.add_glyph(source, scatter)
        r.selection_glyph = Scatter(x=x, y=y, fill_alpha=1.0, size=7.5, line_color='black', line_width=0.75,  fill_color='color')
        r.nonselection_glyph = r.glyph
        p.x_range.renderers.append(r)
        p.y_range.renderers.append(r)

        if x.startswith("revenue"):
            p.xaxis[0].formatter = NumeralTickFormatter(format="0a")
        if y.startswith("revenue"):
            p.yaxis[0].formatter = NumeralTickFormatter(format="0a")

        p.add_tools(hover)

        source.selected.on_change('indices', on_selected)

        return p

    def on_selected(attr, old, new):
        global update_in_progress_flag # for efficiency...
        old_set = set(old)
        new_set = set(new)
        if old_set != new_set:
            if len(new_set) > 1 and not update_in_progress_flag:
                LOGGER.debug('more than one node is selected')
                selected_components = [source.data[NODE_COLUMN_CONNECTED_COMPONENT][i] for i in new]
                if len(new_set) == 2 and selected_components[-2] != selected_components[-1]:
                    # most likely we are switching to a new component...
                    selected_values = [source.data['id'][new[-1]]]
                else:
                    component_counts = pd.Series(selected_components).value_counts()
                    most_selected_component = component_counts.idxmax()
                    selected_values = [source.data['id'][i] for i in new if source.data[NODE_COLUMN_CONNECTED_COMPONENT][i] == most_selected_component]
            else:
                selected_values = [source.data['id'][i] for i in new]
            old_selected_values = [source.data['id'][i] for i in old]
            LOGGER.debug('For scatterplot on_selected(%s, %s) = %s', old, new, selected_values)
            update_in_progress_flag = True
            on_notify_selection_changed(old_selected_values, selected_values)
            update_in_progress_flag = False


    def create_diagonal_plot(attr, i, N):
        p = figure(width=200, height=200, background_fill_color="#fafafa", border_fill_color="white", min_border=5,
                   tools="", toolbar_location=None,
                   sizing_mode='inherit')
        if i % N == 0:  # first column
            p.min_border_left = p.min_border + 4
            p.width += 40
            p.yaxis.major_label_orientation = "vertical"
            p.yaxis.visible = True
            p.yaxis.axis_label = attribute_labels[attr]["label"]
            p.yaxis.major_label_text_color = 'white'
        else:
            p.yaxis.visible = False

        if i >= N * (N - 1):  # last row
            p.min_border_bottom = p.min_border + 40
            p.height += 40
            p.xaxis.axis_label = attribute_labels[attr]["label"]
            p.xaxis.visible = True
        else:
            p.xaxis.visible = False

        source = hist_data[attr]
        quad = Quad(top='top', bottom='bottom', left='left', right='right', line_color="white", fill_alpha=0.6,
                    fill_color=attribute_labels[attr]["color"])
        p.add_glyph(source, quad)


        return p

    plots = []

    for i, (y, x) in enumerate(product(attributes, reversed(attributes))):
        if x == y:
            p = create_diagonal_plot(x, i, N)
        else:
            p = create_scatter_plot(x, y, i, N, source)
        plots.append(p)

    # Create the gridplot
    #grid_layout = layout(children=[[plots[j * N + i] for i in range(N)] for j in range(len(plots)//N)], sizing_mode='inherit')
    #grid_layout = gridplot(plots, ncols=N, sizing_mode='inherit', width=800, height=800)
    grid_layout = grid(plots, ncols=N, sizing_mode='inherit')

    # TODO Click Tool
    # TODO Color for outlier?

    def on_external_selection(selected_nodes: Set[str]):
        global update_in_progress_flag
        new_indices = {i for i, node in enumerate(source.data['id']) if node in selected_nodes}
        if new_indices != set(source.selected.indices):
            update_in_progress_flag = True
            source.selected.indices = list(new_indices)
            update_in_progress_flag = False

    # Display the grid plot
    LOGGER.info('Created SPLOM-gridplot - returning.')
    return grid_layout, on_external_selection


if __name__ == '__main__':
    from bokeh.models import ColumnDataSource, DatetimeTickFormatter, Range1d, BoxZoomTool, LabelSet, \
        NumeralTickFormatter, \
        FactorRange, Tabs, TabPanel, LayoutDOM, HoverTool, WheelZoomTool, MultiLine, TapTool
    graph, node_df, edge_df = get_graph_attributes(merge_node_dates=True)
    p5, _ = create_scatterplot_matrix(graph, node_df, edge_df, lambda x1, x2: None,True)
    p6, _ = create_scatterplot_matrix(graph, node_df, edge_df, lambda x1, x2: None,False)
    tab5 = TabPanel(child=p5, title='Company Overview')
    tab6 = TabPanel(child=p6, title='Person Overview')
    show(row(Tabs(tabs=[tab5, tab6], sizing_mode='stretch_both', width=800, height=800), sizing_mode='stretch_both'))
