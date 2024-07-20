from math import pi

from bokeh import palettes
from bokeh.plotting import figure
from bokeh.models import DatetimeTickFormatter, FactorRange, NumeralTickFormatter, HoverTool, GlyphRenderer, LayoutDOM
from bokeh.models import ColumnDataSource, Wedge
from data_handling.network import *


OUTLIER_WORKS_FOR_LAST = 'Ownership before Work-Relation'
def create_multi_edge_view(graph, node_df, edge_df, on_notify_selection_changed: Callable[[List[str], List[str]], None]) -> Tuple[LayoutDOM, Callable[[Set[str]], None], Set[str], Dict[str, Set[str]]]:
    TOOLS = "pan,wheel_zoom,box_zoom,tap,reset,save"

    # ----------- WEGDE -----------------------------------
    edge_color_map = {
        t: c for t, c in zip(EDGE_TYPE_MAP.values(), palettes.RdBu4)
    }
    edge_df['color'] = edge_df[EDGE_COLUMN_TYPE].map(edge_color_map)


    def aggregate_group(group):
        group = group.sort_values(by=EDGE_COLUMN_START_DATE)

        start_dates = group[EDGE_COLUMN_START_DATE].tolist()
        end_dates = group[EDGE_COLUMN_END_DATE].tolist()
        types = group[EDGE_COLUMN_TYPE].tolist()
        colors = group['color'].tolist()

        return pd.Series({
            'target': group[EDGE_INDEX_TARGET].tolist()[0],
            'source': group[EDGE_INDEX_SOURCE].tolist()[0],
            'start_first': start_dates[0] if len(start_dates) > 0 else None,
            'start_second': start_dates[1] if len(start_dates) > 1 else None,
            'start_third': start_dates[-1] if len(start_dates) > 0 else None,
            'end': end_dates[-1] if len(end_dates) > 0 else None,
            'type_first': colors[0] if len(types) > 0 else None,
            'type_second': colors[1] if len(types) > 1 else None,
            'type_third': colors[-1] if len(types) > 0 else None,
            'multi_edges_types_sorted': types
        })


    shift = pi / 2
    top_angle = 0 + shift

    # 1 = works for start date
    # 2 = in degree
    # 3 = revenue

    x_choice = 2
    y = EDGE_INDEX_TARGET

    if x_choice == 1:
        x = 'start_third'
        radius = 20 * 60 * 60 * 24 * 365

    elif x_choice == 2:
        x = NODE_COLUMN_IN_DEGREE
        radius = 0.2

    else:
        x = NODE_COLUMN_REVENUE
        radius = 4500

    result = edge_df.groupby(EDGE_COLUMN_MULTI_EDGE_ID).apply(aggregate_group).reset_index()

    # TODO what to do with the once where both edges start at the same time? remove them?
    result = result[result['start_first'] != result['start_second']]

    #wedge_df = result[result['multi_edges_types_sorted'].apply(lambda x: x[-1] == 'Works for')]
    wedge_df = result[result['multi_edges_types_sorted'].apply(lambda x: 'Works for' in x and x[0] != 'Works for')]
    wedge_df = pd.merge(wedge_df, node_df, how='left',
                        left_on='target', right_index=True)

    wedge_df['length_of_multi_edges'] = wedge_df['multi_edges_types_sorted'].apply(len)
    wedge_df['str_multi_edges'] = wedge_df['multi_edges_types_sorted'].apply(str)

    wedge_df['date'] = pd.to_datetime(wedge_df['date'])

    def combine_dates(row):
        if len(row['multi_edges_types_sorted']) == 2:
            start_first = pd.to_datetime(row['start_first']).strftime('%Y-%m-%d')
            start_second = pd.to_datetime(row['start_second']).strftime('%Y-%m-%d')
            return f"{start_first}<br>{start_second}"#f"1st edge: {start_first}<br>2nd edge: {start_second}<br>"
        elif len(row['multi_edges_types_sorted']) == 3:
            start_first = pd.to_datetime(row['start_first']).strftime('%Y-%m-%d')
            start_second = pd.to_datetime(row['start_second']).strftime('%Y-%m-%d')
            start_third = pd.to_datetime(row['start_third']).strftime('%Y-%m-%d')
            return f"{start_first}<br>{start_second}<br>{start_third}"#f"1st edge: {start_first}<br>2nd edge: {start_second}<br>3rd edge: {start_third}<br>"
        else:
            return None


    wedge_df['combined_dates'] = wedge_df.apply(combine_dates, axis=1)

    # wedge_df = wedge_df#[wedge_df['multi_edges_types_sorted'].apply(lambda x: len(x) == 3)]
    wedge_df = wedge_df.sort_values(by=NODE_COLUMN_IN_DEGREE, ascending=False).head(60)

    wedge_df = wedge_df.sort_values(
        by=['length_of_multi_edges', 'str_multi_edges', 'start_third'],
        ascending=[False, True, False]
    )


    wedge_df['middle1'] = top_angle - (wedge_df['start_second'] - wedge_df['start_first']) / (
            wedge_df['end'] - wedge_df['start_first']) * 2 * np.pi

    wedge_df['middle2'] = top_angle - (wedge_df['start_third'] - wedge_df['start_first']) / (
            wedge_df['end'] - wedge_df['start_first']) * 2 * np.pi

    #wedge_df_2['start_third'] = pd.to_datetime(wedge_df_2['start_third'])
    wedge_df['start_third'] = pd.to_datetime(wedge_df['start_third'])

    # source1 = ColumnDataSource(wedge_df_2)
    source2 = ColumnDataSource(wedge_df)

    #print(wedge_df_2.head(50))
    #print(wedge_df.head(25))

    plot = figure(title='Largest Companies with Atypical Multi Relations', background_fill_color="#fafafa",
                  tools=TOOLS,
                  y_range=FactorRange(),
                  width=1000, height=800, sizing_mode='inherit')

    # for the once with only 3 edges:
    glyph21 = Wedge(x=x, y=y,
                    radius=radius,
                    start_angle='middle1',
                    end_angle=top_angle,
                    fill_color='type_first')

    glyph22 = Wedge(x=x, y=y,
                    radius=radius,
                    start_angle='middle2',
                    end_angle='middle1',
                    fill_color='type_second')

    glyph23 = Wedge(x=x, y=y,
                    radius=radius,
                    start_angle=top_angle,
                    end_angle='middle2',
                    fill_color='type_third')

    renderer21 = plot.add_glyph(source2, glyph21)
    renderer22 = plot.add_glyph(source2, glyph22)
    renderer23 = plot.add_glyph(source2, glyph23)


    plot.y_range = FactorRange(*list(wedge_df[EDGE_INDEX_TARGET].unique()))
    # plot.yaxis[0].formatter = DatetimeTickFormatter(years=['%Y'])

    hover2 = HoverTool(tooltips=[
        ('Person', '@source'),
        ('Company', '@target'),
        ('Company Founding Date', '@date{%F}'),
        ('Company Type', '@type'),
        ('Company Country', '@country'),
        ('Company Revenue', '@revenue{$0.00 a}'),
        ('Relation Created at', '@combined_dates{safe}'),
        # ('2nd edge', '@start_second{%F}'),
        # ('3rd edge', '@start_third{%F}')
        # ('connected component', '@connected_component')
    ], renderers=[renderer21, renderer22, renderer23])

    hover2.formatters = {
        '@date': 'datetime',
        '@start_first': 'datetime',
        '@start_second': 'datetime',
        '@start_third': 'datetime',

    }
    plot.add_tools(hover2)

    plot.yaxis.axis_label = 'Company'

    if x_choice == 1:
        plot.xaxis.axis_label = 'Works-for Start Date'
        plot.xaxis[0].formatter = DatetimeTickFormatter(years=['%Y'])

    elif x_choice == 2:
        plot.xaxis.axis_label = 'Company In-Relation'

    else:
        plot.xaxis.axis_label = 'Revenue'
        plot.xaxis[0].formatter = NumeralTickFormatter(format="$0.00 a")


    def on_selection_changed(attr, old, new):
        new_persons = [source2.data[EDGE_INDEX_SOURCE][i] for i in new]
        old_persons = [source2.data[EDGE_INDEX_SOURCE][i] for i in old]
        on_notify_selection_changed(old_persons, new_persons)

    def on_external_selection_changed(new_persons: Set[str]):
        source2.selected.indices = [i for i, person in enumerate(source2.data[EDGE_INDEX_SOURCE]) if person in new_persons]

    named_entities = set(source2.data[EDGE_INDEX_SOURCE])
    all_entities = named_entities.copy()
    all_entities.update(source2.data[EDGE_INDEX_TARGET])
    source2.selected.on_change('indices', on_selection_changed)
    return plot, on_external_selection_changed, named_entities, {OUTLIER_WORKS_FOR_LAST: all_entities}
