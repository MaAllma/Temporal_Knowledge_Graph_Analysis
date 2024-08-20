import datetime
import itertools
from math import pi
from typing import Container

import pandas as pd
from bokeh.palettes import OrRd9, OrRd3
from bokeh.transform import linear_cmap
from bokeh.plotting import figure
from bokeh.models import ColorBar, ColumnDataSource, LayoutDOM
from bokeh.layouts import column
from data_handling.network import *

OUTLIER_COMPANY_MERGE = 'Unusual Company Merge/Acquisition'
def create_company_merges_heatmap(graph, node_df, edge_df, top_i, title, filter=False) -> Tuple[figure, ColumnDataSource]:
    OWNERSHIP_COUNTER = 'ownership_count'
    works_for_multiple_df = edge_df.loc[edge_df[EDGE_COLUMN_TYPE].apply(lambda x: 'Beneficial Ownership' in x)].groupby(
        EDGE_INDEX_SOURCE).size().reset_index(name=OWNERSHIP_COUNTER)

    works_for_multiple_df = works_for_multiple_df.loc[works_for_multiple_df[OWNERSHIP_COUNTER] > top_i]

    works_for_multiple_df = pd.merge(works_for_multiple_df, edge_df, how='left', on=EDGE_INDEX_SOURCE)
    works_for_multiple_df = works_for_multiple_df.loc[works_for_multiple_df[EDGE_COLUMN_TYPE] == 'Beneficial Ownership']
    works_for_multiple_df = pd.merge(works_for_multiple_df, node_df[NODE_COLUMN_CONNECTED_COMPONENT], how='left',
                                     left_on=EDGE_INDEX_SOURCE, right_index=True)
    works_for_multiple_df[NODE_COLUMN_CONNECTED_COMPONENT] = works_for_multiple_df[
        NODE_COLUMN_CONNECTED_COMPONENT].astype(str)

    if filter:
        works_for_multiple_df = works_for_multiple_df.loc[works_for_multiple_df['connected_component'] != '0']

    works_for_multiple_df['year'] = works_for_multiple_df[EDGE_COLUMN_START_DATE].dt.year

    heatmap_data: pd.DataFrame = works_for_multiple_df.groupby([EDGE_INDEX_SOURCE, 'year']).agg(
        count=(EDGE_COLUMN_START_DATE, 'size'),
        targets=(EDGE_INDEX_TARGET, lambda x: '<br>'.join(x))
    ).reset_index()

    heatmap_data[EDGE_INDEX_SOURCE] = heatmap_data[EDGE_INDEX_SOURCE].astype(str)
    heatmap_data['year'] = heatmap_data['year'].astype(str)

    # colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    if filter:
        colors = OrRd3[::-1]
    else:
        colors = OrRd9[::-1]
    sorted_years = sorted(heatmap_data['year'].unique())
    tools = "hover,tap,save"

    p = figure(title=title,
               x_range=sorted_years, y_range=list(heatmap_data[EDGE_INDEX_SOURCE].unique()[::-1]),
               x_axis_location="above", width=1000, height=400,
               tools=tools, toolbar_location='below',
               tooltips=[('Owner', '@source'), ('Companies', '@targets{safe}')], sizing_mode='inherit'
               )

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "7px"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3

    min_count = heatmap_data['count'].min()
    max_count = heatmap_data['count'].max()
    #heatmap_data = heatmap_data.sort_values(by=['year', EDGE_INDEX_SOURCE]).reset_index(drop=True)
    heatmap_ds = ColumnDataSource(heatmap_data)
    color_mapper = linear_cmap("count", colors, low=min_count, high=max_count)
    renderer = p.rect(x="year", y=EDGE_INDEX_SOURCE, width=1, height=1, source=heatmap_ds,
                       fill_color=color_mapper,
                       line_color=None)
    # disable nonselection_glyph to prevent inconsistencies with the second heatmap...
    renderer.nonselection_glyph = renderer.glyph
    # TODO set selection_glyph to something that is better visible


    color_bar = ColorBar(color_mapper=color_mapper['transform'], width=8, location=(0, 0))
    p.add_layout(color_bar, 'right')



    return p, heatmap_ds


def create_heatmap_view(graph, node_df, edge_df, on_notify_external_selection: Callable[[List[str], List[str]], None]) -> Tuple[LayoutDOM, Callable[[Set[str], datetime.datetime], None], Set[str], Dict[str, Set[str]]]:
    c1, c1_ds = create_company_merges_heatmap(graph, node_df, edge_df, 50, 'New Ownerships Per Year (In the largest Connected Component)')
    c2, c2_ds = create_company_merges_heatmap(graph, node_df, edge_df, 3,
                                       'New Ownerships Per Year (Except entities the largest Connected Component)',True)

    def indicies_for_person_and_year(ds: ColumnDataSource, persons: Container[str], cur_year: str):
        return [i for i, (person, year) in enumerate(zip(ds.data[EDGE_INDEX_SOURCE], ds.data['year'])) if person in persons and year == cur_year]

    def heatmap_callback_factory(ds: ColumnDataSource, ds_name: str) -> Callable[[str, List[int], List[int]], None]:
        def on_selected(attr: str, old: List[int], new: List[int]) -> None:

            LOGGER.debug('heatmap_on_selected(%s, %s, %s) for %s', attr, old, new, ds_name)
            decoded_years = [ds.data['year'][i] for i in new]
            decoded_old_person = [ds.data[EDGE_INDEX_SOURCE][i] for i in old]
            decoded_person = [ds.data[EDGE_INDEX_SOURCE][i] for i in new]
            unique_years = set(decoded_years)
            if len(unique_years) > 1:
                last_year = decoded_years[-1]
                LOGGER.debug('Selected multiple years: %s. Changing selection for %s to retain the persons selected, but switching over to the most recently clicked year %s.', unique_years, ds_name, last_year)
                #person_index_map = {person: index for index, (person, year) in enumerate(zip(ds.data[EDGE_INDEX_SOURCE], ds.data['year'])) if year == last_year}
                new_indices = indicies_for_person_and_year(ds, set(decoded_person), last_year)#[person_index_map[person] for person in decoded_person if person in person_index_map]
                ds.selected.indices = new_indices
                return
            elif len(unique_years) == 0: # everything deselected
                LOGGER.debug('Deselected everything in heatmap view for %s.', ds_name)
                on_notify_external_selection(decoded_old_person, [])
            else:
                LOGGER.debug('Selected year = %s, person = %s in %s', decoded_years[0], decoded_person, ds_name)
                on_notify_external_selection(decoded_old_person, decoded_person)
                # TODO select year

        return on_selected

    on_comp_0_selected = heatmap_callback_factory(c1_ds, 'Component-0 Datasource')
    on_comp_non0_selected = heatmap_callback_factory(c2_ds,'Component-non0 Datasource')

    c1_ds.selected.on_change('indices', on_comp_0_selected)
    c2_ds.selected.on_change('indices', on_comp_non0_selected)

    def on_external_selection_changed(new_values: Set[str], cur_time: datetime.datetime) -> None:
        cur_year = str(cur_time.year)
        LOGGER.debug('on_external_selection_changed for heatmap-view changed to %s and year %s', new_values, cur_year)
        c1_ds.selected.indices = indicies_for_person_and_year(c1_ds, new_values, cur_year)
        c2_ds.selected.indices = indicies_for_person_and_year(c2_ds, new_values, cur_year)

    merge_persons = {person for person in itertools.chain(c1_ds.data[EDGE_INDEX_SOURCE], c2_ds.data[EDGE_INDEX_SOURCE])}
    mentioned_entities = merge_persons.copy()
    mentioned_entities.update(
        company for ds in [c1_ds, c2_ds] for company_str in ds.data['targets'] for company in company_str.split('<br>')
    )

    view = column(c1, c2)
    return view, on_external_selection_changed, merge_persons, {OUTLIER_COMPANY_MERGE: mentioned_entities}
