import datetime
import itertools

import numpy as np
import pandas as pd
import bokeh.palettes as palettes
from bokeh.transform import factor_cmap, factor_mark
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, Range1d, BoxZoomTool, LabelSet, NumeralTickFormatter, \
    FactorRange, Tabs, TabPanel, LayoutDOM, HoverTool, WheelZoomTool, MultiLine, TapTool
from data_handling.network import *
from bokeh.models import Select, Slider, CheckboxGroup, ColumnDataSource
from bokeh.layouts import column, row
from bokeh.io import curdoc

from interactive_plots.vis_heatmap_company_merges import create_heatmap_view
from interactive_plots.vis_multi_edge import create_multi_edge_view
from interactive_plots.vis_scatterplot import create_scatterplot_matrix
from utils import get_logger
LOGGER = get_logger(__name__)

DEFAULT_TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save"
def create_figure_companies(graph, node_df, edge_df):
    company_df = node_df.loc[node_df[NODE_COLUMN_TYPE].apply(lambda x: 'Company' in x)]

    edge_types = [edge_type.lower().replace(' ', '_') for edge_type in edge_df['type'].unique()]
    # Count edges where the company is the target
    target_counts = edge_df.groupby([EDGE_INDEX_TARGET, EDGE_COLUMN_TYPE]).size().unstack(fill_value=0)
    target_counts.columns = target_counts.columns.str.lower().str.replace(' ', '_')
    # Add counts to companies_df
    for edge_type in edge_types:
       # if edge_type in source_counts.columns:
       #     company_df = company_df.merge(source_counts[edge_type], left_on='id', right_index=True, how='left').rename(columns={edge_type: f'{edge_type}_source'})
        if edge_type in target_counts.columns:
            company_df = company_df.merge(target_counts[edge_type], left_on='id', right_index=True, how='left').rename(columns={edge_type: f'{edge_type}_counter'})
    WORKS_FOR_COUNTER = 'works_for_counter'
    SHAREHOLDER_COUNTER = 'shareholder_counter'
    OWNERSHIP_COUNTER = 'beneficial_ownership_counter'
    sort_by_selection = Select(title="Sort by", value=NODE_COLUMN_IN_DEGREE,
                               options=[NODE_COLUMN_GENERIC_DATE, NODE_COLUMN_IN_DEGREE, NODE_COLUMN_REVENUE,
                                        NODE_COLUMN_OUT_DEGREE, WORKS_FOR_COUNTER, SHAREHOLDER_COUNTER,
                                        OWNERSHIP_COUNTER])
    top_i_slider = Slider(title="Amount", start=0, end=50, value=25, step=5)
    ascending_checkbox = CheckboxGroup(labels=["Sort ascending"], active=[])

    columns = [NODE_COLUMN_GENERIC_DATE, NODE_COLUMN_REVENUE, NODE_COLUMN_COUNTRY, NODE_COLUMN_IN_DEGREE,
               NODE_COLUMN_OUT_DEGREE, WORKS_FOR_COUNTER, SHAREHOLDER_COUNTER, OWNERSHIP_COUNTER]
    continuous = [NODE_COLUMN_GENERIC_DATE, NODE_COLUMN_REVENUE, NODE_COLUMN_IN_DEGREE, NODE_COLUMN_OUT_DEGREE,
                  WORKS_FOR_COUNTER, SHAREHOLDER_COUNTER, OWNERSHIP_COUNTER]
    select_x = Select(title='X-Axis', value=NODE_COLUMN_REVENUE, options=continuous)

    select_y = Select(title='Y-Axis', value=NODE_COLUMN_IN_DEGREE, options=columns)

    ranked_companies_controls = column(sort_by_selection, select_x, select_y, top_i_slider, ascending_checkbox,
                                       width=200)

    is_ascending = len(ascending_checkbox.active) > 0
    top_companies_df = ColumnDataSource(
        company_df.sort_values(by=sort_by_selection.value, ascending=is_ascending).head(top_i_slider.value))

    COMPANYTYPES = sorted(top_companies_df.data[NODE_COLUMN_TYPE].unique())

    if is_ascending:
        title = 'Last ' + str(top_i_slider.value) + ' companies by ' + sort_by_selection.value
    else:
        title = 'Top ' + str(top_i_slider.value) + ' companies by ' + sort_by_selection.value

    if select_y.value == NODE_COLUMN_COUNTRY:
        p = figure(title=title, background_fill_color="#fafafa", tools=DEFAULT_TOOLS, y_range=FactorRange(), width=800, height=800, sizing_mode='inherit')
        p.y_range = FactorRange(*list(top_companies_df.data[NODE_COLUMN_COUNTRY].unique()))
    else:
        p = figure(title=title, background_fill_color="#fafafa", tools=DEFAULT_TOOLS, width=800, height=800, sizing_mode='inherit')
    p.yaxis.axis_label = select_y.value
    p.xaxis.axis_label = select_x.value
    p.scatter(select_x.value, select_y.value, source=top_companies_df,
              legend_group=NODE_COLUMN_TYPE, fill_alpha=0.4, size=12,
              color=factor_cmap(NODE_COLUMN_TYPE, 'Category10_6', COMPANYTYPES))

    p.hover.tooltips = [
        ('name', '@id'),
        ('type', '@type'),
        ('date', '@date{%F}'),
        ('connected component', '@connected_component'),
        #('products', '@ProductServices'),
        ('country', '@country'),
        ('revenue', '@revenue{$0.00 a}'),
        ('head of org', '@HeadOfOrg')
    ]
    p.hover.formatters = {
        '@date': 'datetime'
    }
    if select_x.value == NODE_COLUMN_REVENUE:
        p.xaxis[0].formatter = NumeralTickFormatter(format="$0.00 a")
    if select_y.value == NODE_COLUMN_REVENUE:
        p.yaxis[0].formatter = NumeralTickFormatter(format="$0.00 a")
    if select_x.value == NODE_COLUMN_GENERIC_DATE:
        p.xaxis[0].formatter = DatetimeTickFormatter(years=['%Y'])
    if select_y.value == NODE_COLUMN_GENERIC_DATE:
        p.yaxis[0].formatter = DatetimeTickFormatter(years=['%Y'])
    p.legend.location = "top_left"
    p.legend.title = "Company Type"

    ranked_companies_layout = row(p, ranked_companies_controls)

    def update(attr, old, new):
        LOGGER.debug('Selection changed to %s', new)
        ranked_companies_layout.children[0] = create_figure_companies(graph, node_df, edge_df)

    select_x.on_change('value', update)
    select_y.on_change('value', update)
    sort_by_selection.on_change('value', update)
    top_i_slider.on_change('value', update)
    ascending_checkbox.on_change('active', update)
    sort_by_selection.on_change('value', update)
    top_i_slider.on_change('value', update)
    ascending_checkbox.on_change('active', update)

    return ranked_companies_layout



# ------------------------- works for last --------------------------------------
def works_for_last(graph, node_df, edge_df) -> LayoutDOM:
    grouped_sorted = edge_df.sort_values(by=EDGE_COLUMN_START_DATE).groupby(EDGE_COLUMN_MULTI_EDGE_ID)[
        EDGE_COLUMN_TYPE].apply(list).reset_index(name='multi_edges_types_sorted')

    # Filter cases where 'Works for' is the last entry in the list
    works_for_last = grouped_sorted[grouped_sorted['multi_edges_types_sorted'].apply(lambda x: x[-1] == 'Works for')]
    works_for_last_df = pd.merge(works_for_last, edge_df, on=[EDGE_COLUMN_MULTI_EDGE_ID], how='left')
    works_for_last_df = pd.merge(works_for_last_df, node_df[NODE_COLUMN_CONNECTED_COMPONENT], how='left', left_on=EDGE_INDEX_SOURCE, right_index=True)
    works_for_last_df[EDGE_INDEX_TARGET] = works_for_last_df[EDGE_INDEX_TARGET].astype(str)
    works_for_last_df[EDGE_COLUMN_START_DATE] = pd.to_datetime(works_for_last_df[EDGE_COLUMN_START_DATE])
    works_for_last_df = works_for_last_df.sort_values(by='multi_edges_types_sorted')
    # print(works_for_last_df.data)
    EDGETYPES = sorted(np.unique(works_for_last_df[EDGE_COLUMN_TYPE]))
    p2 = figure(title='Companies with atypical multi edges (works-for last)', background_fill_color="#fafafa", tools=DEFAULT_TOOLS,
                y_range=FactorRange(), width=800, height=800)
    p2.yaxis.axis_label = 'Company'
    p2.xaxis.axis_label = 'Edge Start Date'
    p2.y_range = FactorRange(*list(works_for_last_df[EDGE_INDEX_TARGET].unique()))
    p2.scatter(EDGE_COLUMN_START_DATE, EDGE_INDEX_TARGET, source=works_for_last_df,
               legend_group=NODE_COLUMN_TYPE, fill_alpha=0.4, size=12,
               color=factor_cmap(EDGE_COLUMN_TYPE, 'Category10_6', EDGETYPES))
    p2.hover.tooltips = [
        ('source', '@source'),
        ('target', '@target'),
        ('start date', '@start_date{%F}'),
        #    ('end date', '@end_date{%F}'),
        ('target type', '@target_type'),
        ('connected component', '@connected_component')
    ]
    p2.hover.formatters = {
        '@start_date': 'datetime',
        '@end_date': 'datetime'
    }
    p2.xaxis[0].formatter = DatetimeTickFormatter(years=['%Y'])
    p2.legend.location = "top_left"
    p2.legend.title = "Edge Type"
    return p2

# ----------- works for multiple companies ------------------

def _get_column_list_from_grouping(df: pd.DataFrame, grouping: pd.api.typing.DataFrameGroupBy, column: str, uniquify: bool = False) -> List[pd.Series]:
    ls = [df.loc[indices, column] for indices in grouping.groups.values()]
    if uniquify:
        ls = [series.unique()[0] for series in ls]
    return ls


OUTLIER_WORKS_FOR_MULTIPLE = ('Person Works for Multiple Companies')
OUTLIER_EMPLOYEE_WORKS_FOR_MULTIPLE = ('Employee works for multiple companies')
def works_for_multiple_companies(graph, node_df, edge_df, on_notify_selection_changed: Callable[[List[str], List[str]], None]) -> Tuple[LayoutDOM, Callable[[Set[str]], None], Set[str], Dict[str, Set[str]]]:
    works_for_multiple_df = edge_df.loc[edge_df[EDGE_COLUMN_TYPE].apply(lambda x: 'Works' in x)].groupby(
        EDGE_INDEX_SOURCE).size().reset_index(name='works_for_count')
    works_for_multiple_df = works_for_multiple_df.loc[works_for_multiple_df['works_for_count'] > 1]
    works_for_multiple_df = pd.merge(works_for_multiple_df, edge_df, how='left', on=EDGE_INDEX_SOURCE)
    works_for_multiple_df = works_for_multiple_df.loc[works_for_multiple_df[EDGE_COLUMN_TYPE] == 'Works for']
    works_for_multiple_df = pd.merge(works_for_multiple_df, node_df[[NODE_COLUMN_CONNECTED_COMPONENT, NODE_COLUMN_COUNTRY, NODE_COLUMN_DATE_OF_BIRTH]], how='left', left_on=EDGE_INDEX_SOURCE, right_index=True)
    works_for_multiple_df[NODE_COLUMN_CONNECTED_COMPONENT] = works_for_multiple_df[NODE_COLUMN_CONNECTED_COMPONENT].astype(str)

    #py_time = works_for_multiple_df[EDGE_COLUMN_START_DATE].dt.to_pydatetime()
    min_time = datetime.datetime.fromisoformat('2009-06-01')
    max_time = datetime.datetime.fromisoformat('2035-01-01')
    p = figure(title='Person works for multiple companies', background_fill_color="#fafafa", tools=['pan', 'reset'], width=800, height=800,
               x_range=Range1d(-0.05, 1.05, bounds=(-0.05, 1.05)), y_range=Range1d(min_time, max_time, bounds=(min_time, max_time)), sizing_mode='inherit')
    MARKERS = {'Generic Company': 'circle', 'Fishing Company':'square', 'Logistics Company': 'diamond'}


    works_for_multiple_df = works_for_multiple_df.sort_values(by=[EDGE_INDEX_SOURCE, EDGE_COLUMN_START_DATE])
    # now we create an alternating ordering by ranking within groups and then reducing this to Z/2
    # note that we add a +1 in order to enforce the slopes to start on the left
    grouping = works_for_multiple_df.groupby(EDGE_INDEX_SOURCE, sort=False)
    works_for_multiple_df['side'] = (grouping[EDGE_COLUMN_START_DATE].rank(method='first') + 1) % 2
    works_for_multiple_df['marker'] = works_for_multiple_df[EDGE_COLUMN_TARGET_TYPE].map(MARKERS)
    works_for_multiple_df['size'] = 20 + 10 * (works_for_multiple_df[EDGE_COLUMN_TARGET_TYPE] == 'Logistics Company')

    #print(works_for_multiple_df[['side', EDGE_COLUMN_START_DATE, EDGE_INDEX_SOURCE]].head(10))
    PALETTES = {
        '4780': palettes.BrBG4,
        '4661': palettes.BrBG4[::-1],
        '0': palettes.Purples5
    }
    color_dict = {
       key: PALETTES[connected_component][i]
       for connected_component in works_for_multiple_df[NODE_COLUMN_CONNECTED_COMPONENT].unique()
       for i, key in enumerate(works_for_multiple_df.loc[works_for_multiple_df[NODE_COLUMN_CONNECTED_COMPONENT] == connected_component, EDGE_INDEX_SOURCE].unique())
    }
    #LOGGER.debug('Connected componentes have size: %s', works_for_multiple_df.drop_duplicates(subset=[EDGE_INDEX_SOURCE]).groupby([NODE_COLUMN_CONNECTED_COMPONENT]).size())

    #print(dob_series)
    data_dict = {
        'xs': _get_column_list_from_grouping(works_for_multiple_df, grouping, 'side'),
        'ys':_get_column_list_from_grouping(works_for_multiple_df, grouping, EDGE_COLUMN_START_DATE),
        'color': [color_dict[name] for name in grouping.groups.keys()],
        'source': list(grouping.groups.keys()),
        'dob': [np.unique(series.dt.to_pydatetime())[0] for series in _get_column_list_from_grouping(works_for_multiple_df, grouping, NODE_COLUMN_DATE_OF_BIRTH)],
        'person_country': _get_column_list_from_grouping(works_for_multiple_df, grouping, NODE_COLUMN_COUNTRY, uniquify=True),
        'connected_component': _get_column_list_from_grouping(works_for_multiple_df, grouping, NODE_COLUMN_CONNECTED_COMPONENT, uniquify=True),
        'companies': ['<br>'.join(series) for series in _get_column_list_from_grouping(works_for_multiple_df, grouping, EDGE_INDEX_TARGET)]
    }
    multi_line_ds = ColumnDataSource(data_dict)
    multi_line_renderer = p.multi_line(xs='xs', ys='ys',source=multi_line_ds, line_color='color',
                                       line_width=6)
    multi_line_renderer.hover_glyph = MultiLine(line_width=8, line_color='color')
    multi_line_renderer.selection_glyph = MultiLine(line_width=8, line_color='color')
    multi_line_renderer.nonselection_glyph = MultiLine(line_width=6, line_color='color', line_alpha=0.5)
    p.vspan(x=0, line_color='black', line_width=10)
    p.vspan(x=1, line_color='black', line_width=10)
    scatter_renderer = p.scatter('side', EDGE_COLUMN_START_DATE, source=works_for_multiple_df, size='size',
                legend_group=EDGE_COLUMN_TARGET_TYPE,
                #color=factor_cmap(EDGE_COLUMN_TARGET_TYPE, 'Category10_6',
                #                 sorted(np.unique(works_for_multiple_df[EDGE_COLUMN_TARGET_TYPE]))),
                color='#AAAAAA',
                alpha=0.9,
                line_color='black',
                line_width=2,
                marker='marker')#factor_mark(NODE_COLUMN_CONNECTED_COMPONENT, MARKERS, sorted(set(works_for_multiple_df[[NODE_COLUMN_CONNECTED_COMPONENT, EDGE_COLUMN_TARGET_TYPE]].itertuples(index=False)))))

    p.legend.location = "bottom_right"
    p.yaxis[0].formatter = DatetimeTickFormatter(years=['%Y'])
    hover_person = HoverTool(tooltips=[('Person', '@source'),
                                       ('Date of Birth', '@dob{%F}'),
                                       ('Country of Origin', '@person_country'),
                                       ('Employers', '@companies{safe}')],
                             formatters={'@dob': 'datetime'},
                             renderers=[multi_line_renderer],
                             line_policy='nearest')
    zoom_tool = WheelZoomTool(dimensions='height')
    tap_tool = TapTool(renderers=[multi_line_renderer])
    p.add_tools(hover_person, zoom_tool, tap_tool)
    p.xaxis.visible = False

    def on_selected(attr, old, new):
        old_set = set(old)
        new_set = set(new)
        if old_set != new_set:
            old_selected_values = [multi_line_ds.data['source'][i] for i in old]
            selected_values = [multi_line_ds.data['source'][i] for i in new]
            LOGGER.debug('on_selected(%s, %s) = %s', old, new, selected_values)
            # if selected_values:
            #     selected_components = [multi_line_ds.data['connected_component'][i] for i in new]
            #     if np.unique(selected_components).shape[0] > 1:
            #         LOGGER.info('Clearing parts of the  selection since multiple connected components were selected. Selected = %s, retaining = %s', selected_components, selected_components[-1])
            #         multi_line_ds.selected.indices = [i for i, component in zip(new, selected_components) if component == selected_components[-1]]
            #         return # only propagate the follow-up update
            #     else:
            #         LOGGER.debug('Selecting nodes in the graph.')
            on_notify_selection_changed(old_selected_values, selected_values)

    def on_external_selection(new_values: Set[str]):
        new_selected_indices = [i for i, value in enumerate(multi_line_ds.data['source']) if value in new_values]
        if set(new_selected_indices) != set(multi_line_ds.selected.indices):
            multi_line_ds.selected.indices = new_selected_indices

    multi_line_ds.selected.on_change('indices', on_selected)

    persons = cast(Set[str],set(data_dict['source']))
    res_dict = {
        OUTLIER_WORKS_FOR_MULTIPLE: persons,
        OUTLIER_EMPLOYEE_WORKS_FOR_MULTIPLE: cast(Set[str], set(works_for_multiple_df[EDGE_INDEX_TARGET].to_list())),
    }
    return p, on_external_selection, persons, res_dict

def create_interesting_nodes_tabs(graph, node_df, edge_df, on_notify_external_selection: Callable[[List[str], List[str]], None], known_interesting_nodes: Dict[str, Set[str]]) -> Tuple[LayoutDOM, Callable[[List[str], datetime.datetime, bool], None], Set[str], Dict[str, Set[str]]]:
    #p1 = create_figure_companies(graph, node_df, edge_df)
    p2, multi_edge_callback, multi_edge_labeled, multi_edge_interesting = create_multi_edge_view(graph, node_df, edge_df, on_notify_external_selection)
    p3, company_callback, company_labeled, company_interesting = works_for_multiple_companies(graph, node_df, edge_df, on_notify_external_selection)
    p4, heatmap_callback, heatmap_labeled, heatmap_interesting = create_heatmap_view(graph, node_df, edge_df, on_notify_external_selection)

    to_label = company_labeled | heatmap_labeled | multi_edge_labeled
    interesting_nodes = known_interesting_nodes.copy()
    for key, values in itertools.chain(multi_edge_interesting.items(), company_interesting.items(), heatmap_interesting.items()):
        interesting_nodes.setdefault(key, set()).update(values)

    p5, scatterplot1_callback = create_scatterplot_matrix(graph, node_df, edge_df, on_notify_external_selection, interesting_nodes, True)
    p6, scatterplot2_callback = create_scatterplot_matrix(graph, node_df, edge_df, on_notify_external_selection, interesting_nodes, False)

    def on_external_selection(new_values: List[str], cur_time: datetime.datetime, is_time_only_update):
        LOGGER.debug('on_external_selection(%s, %s)', new_values, cur_time.isoformat())
        new_values = set(new_values)
        heatmap_callback(new_values, cur_time)
        if not is_time_only_update:
            multi_edge_callback(new_values)
            company_callback(new_values)
            scatterplot1_callback(new_values)
            scatterplot2_callback(new_values)

    #tab1 = TabPanel(child=p1, title='Companies ranked by features')
    tab2 = TabPanel(child=p2, title='Largest Companies with Atypical Multi Relations')
    tab3 = TabPanel(child=p3, title='Person Works for Multiple Companies')
    tab4 = TabPanel(child=p4, title='Company Merges')
    tab5 = TabPanel(child=p5, title='Company Overview')
    tab6 = TabPanel(child=p6, title='Person Overview')
    tabs = Tabs(tabs=[tab5, tab6, tab4, tab2, tab3], sizing_mode='stretch_both', width=1400, height=800)
    return tabs, on_external_selection, to_label, interesting_nodes

if __name__ == '__main__':
    graph, node_df, edge_df = get_graph_attributes(merge_node_dates=True)
    curdoc().add_root(create_interesting_nodes_tabs(graph, node_df, edge_df, lambda x,y: None, {})[0])
