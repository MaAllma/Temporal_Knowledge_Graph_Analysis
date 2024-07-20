import datetime
from typing import Union, Tuple, Callable

import numpy as np
import pandas as pd
from bokeh.events import Tap, Event
from bokeh.layouts import row, column
from bokeh.models import LayoutDOM, Range1d, ColumnDataSource, DatetimeTickFormatter, BoxZoomTool, DatetimePicker, \
    CustomJS, Span, VSpan, WheelZoomTool
from bokeh.plotting import figure
from sklearn.neighbors import KDTree, KernelDensity
from data_handling.network import *
from utils import get_logger
import bokeh.palettes as palettes
LOGGER = get_logger(__name__)

def time_based_kde(title: str,
                   time_data: Dict[str, Tuple[pd.DataFrame, str, Dict[str, Any]]],
                   start_datetime: datetime.datetime,
                   end_datetime: datetime.datetime,
                   default_bandwidth: Union[int, float],
                   default_kernel: str,
                   span_source: ColumnDataSource,
                   height: int = 300) -> Tuple[figure, Callable[[str, Set[str]], None]]:

    #LOGGER.debug('Creating kde-plot for time-series %s', time_points)

    epoch_data = {
        key: np.array([dt.timestamp() for dt in time_points[col].dt.to_pydatetime()]).reshape(-1, 1) for key, (time_points, col, _) in time_data.items()
    }
    x_min = {key: max(time_since_unix_epoch.min(), start_datetime.timestamp()) for key, time_since_unix_epoch in epoch_data.items()}
    x_max = {key: min(time_since_unix_epoch.max(), end_datetime.timestamp()) for key, time_since_unix_epoch in epoch_data.items()}
    x_range = Range1d(start_datetime, end_datetime, bounds=(start_datetime, end_datetime))
    def kernel_density_estimate(key: str, valid_nodes: Optional[Set[str]] = None, bandwidth_days: Union[int, float] = default_bandwidth, kernel = default_kernel):
        bandwidth = bandwidth_days * 24 * 60 * 60
        x_min_adj = x_min[key] - bandwidth
        x_max_adj = x_max[key] + bandwidth
        days_in_range = int((x_max_adj - x_min_adj) // (24 * 60 * 60))
        x = np.linspace(x_min_adj, x_max_adj, days_in_range).reshape(-1, 1)

        if kernel == 'gaussian' or kernel == 'exponential':
            bandwidth /= 2
        # see https://github.com/scikit-learn/scikit-learn/blob/2621573e6/sklearn/neighbors/_kde.py#L36
        # we use the tree directly to avoid rebuilding the same tree over and over again...
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        data = epoch_data[key]
        if valid_nodes is not None:
            source_nodes = time_data[key][0][EDGE_INDEX_SOURCE]
            target_nodes = time_data[key][0][EDGE_INDEX_TARGET]
            data = data[source_nodes.isin(valid_nodes) | target_nodes.isin(valid_nodes)]
        if data.shape[0] > 0:
            kde.fit(data)
            y = np.exp(kde.score_samples(x))
            y /= (y.max() - y.min()) * 1.05
        else:
            y = np.zeros_like(x)
        X = pd.Series([datetime.datetime.fromtimestamp(ts) for ts in x.reshape(-1)])
        return X, y

    sources: Dict[str, ColumnDataSource] = {}
    p = figure(title=title, background_fill_color="#fafafa", tools="pan,reset", height=height, sizing_mode='stretch_width',
               x_axis_type='datetime', x_range=x_range, y_range=Range1d(0, 1, bounds=(0, 1)))
    for key, (_, col, kwargs) in time_data.items():
        X, y = kernel_density_estimate(key)

        source = ColumnDataSource(data={
            'X': X,
            'y': y,
        })
        if 'fill_color' not in kwargs:
            kwargs['fill_color'] = 'grey'
        if 'fill_alpha' not in kwargs:
            kwargs['fill_alpha'] = 0.3
        p.varea(x='X', y1='y', y2=0, source=source, **kwargs)
        p.line(x='X', y='y', source=source, alpha=1.0, color='black')
        sources[key] = source
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Changes'
    p.yaxis.visible = False
    p.grid.grid_line_color = "white"
    # p.xaxis.major_label_orientation = 3.14 / 4  # rotate labels
    p.xaxis[0].formatter = DatetimeTickFormatter(years="%Y")
    zoom_tool = WheelZoomTool(dimensions='width')
    p.add_tools(zoom_tool)

    def on_density_params_change(key: str, valid_nodes: Set[str], bandwidth: Union[int, float] = default_bandwidth, kernel = default_kernel):
        X, y = kernel_density_estimate(key, valid_nodes, bandwidth_days=bandwidth, kernel=kernel)
        sources[key].data.update({
            'X': X,
            'y' : y
        })
        p.y_range = Range1d(0, min(p.y_range.end, 1), bounds=(0, 1))

    p.vspan(source=span_source, x='x', line_color='black', line_width=2, line_alpha=0.75)

    return p, on_density_params_change

VALID_KERNELS = ['gaussian', 'exponential', 'epanechnikov', 'cosine', 'linear', 'tophat']
def frequency_time_selection(node_df: pd.DataFrame, edge_df: pd.DataFrame,
                             on_time_selected: Callable[[datetime.datetime], None],
                             min_time: Union[str, datetime.datetime] = '2005-01-01',
                             initial_time: Union[str, datetime.datetime] = '2035-01-01',
                             default_bandwidth = 30,
                             default_kernel = VALID_KERNELS[0]) -> Tuple[LayoutDOM, LayoutDOM, Callable[[Set[str]], None]]:
    initial_time = datetime.datetime.fromisoformat(initial_time) if isinstance(initial_time, str) else initial_time
    start_datetime = datetime.datetime.fromisoformat(min_time) if isinstance(min_time, str) else min_time
    end_datetime = edge_df[EDGE_COLUMN_END_DATE].dt.to_pydatetime().max()
    end_date = end_datetime.date()
    #edge_df = edge_df.loc[edge_df[EDGE_COLUMN_END_DATE] < edge_df[EDGE_COLUMN_END_DATE].max()]
    def valid_timeseries(df: pd.DataFrame, column: str, include_end_filter: bool = False) -> pd.DataFrame:
        series = df[column]
        dt_series = series.dt.to_pydatetime()
        filter_ar = (series.notna() & (dt_series >= start_datetime))
        if include_end_filter:
            filter_ar = filter_ar & (edge_df[EDGE_COLUMN_END_DATE] < edge_df[EDGE_COLUMN_END_DATE].max())
        return df.loc[filter_ar, [column, EDGE_INDEX_SOURCE, EDGE_INDEX_TARGET]]

    date_picker = DatetimePicker(
        title="Select Date of interest",
        value=initial_time,
        min_date=start_datetime,
        max_date=end_date
    )
    LOGGER.debug('Default value for date picker is %s', date_picker.value)
    end_datetime = datetime.datetime.combine(end_datetime, datetime.datetime.max.time())
    span_source = ColumnDataSource(data={
        'x': [initial_time]
    })

    time_data = {
        'Disconnect Frequency': (valid_timeseries(edge_df, EDGE_COLUMN_END_DATE, True), EDGE_COLUMN_END_DATE, {'fill_color': palettes.Dark2_3[1], 'legend_label': 'Removed Relations'}),
        'Connect Frequency': (valid_timeseries(edge_df, EDGE_COLUMN_START_DATE), EDGE_COLUMN_START_DATE, {'fill_color': palettes.Dark2_3[0], 'legend_label': 'Created Relations'}),
        #'Founding Frequency': (valid_timeseries(node_df, NODE_COLUMN_FOUNDING_DATE), {'fill_color': palettes.Dark2_3[2]})
    }

    kde, params_on_change = time_based_kde('Event Frequency',  time_data, start_datetime, end_datetime, default_bandwidth, default_kernel, span_source)
    kde.legend.location = 'top_left'

    def on_time_changed(new_time: datetime.datetime):
        LOGGER.debug('Recognizing that time changed to %s', new_time.isoformat())
        span_source.data.update({
            'x': [new_time]
        })
        if datetime.datetime.fromtimestamp(date_picker.value / 1000) != new_time:
            date_picker.value = new_time

    def on_kde_tapped(event: Event):
        event: Tap = cast(Tap, event)
        LOGGER.debug('Kernel Density plot selected a new time %s, %s', str(type(event.x)), str(event.x))
        on_time_changed(datetime.datetime.fromtimestamp(event.x / 1000))

    def on_date_selected(attr, old, new):
        if old != new:
            dt = datetime.datetime.fromtimestamp(date_picker.value / 1000)
            LOGGER.debug('Date picker selected a new time %s = %s', new, dt.isoformat())
            on_time_selected(dt)

    def on_valid_nodes_changed(valid_nodes: Set[str]):
        for key in time_data.keys():
            params_on_change(key, valid_nodes)

    date_picker.on_change('value', on_date_selected)
    kde.on_event(Tap, on_kde_tapped)

    return kde, date_picker, on_valid_nodes_changed

if __name__ == '__main__':
    from bokeh.io import show
    graph, node_df, edge_df = get_graph_attributes()
    edge_df_mod = edge_df.loc[(edge_df[EDGE_COLUMN_START_DATE] >= '2025-01-01')]
    layout, l2, _ = frequency_time_selection(node_df, edge_df_mod, print, min_time='2025-01-01')
    show(column(layout, l2, sizing_mode='stretch_width'))