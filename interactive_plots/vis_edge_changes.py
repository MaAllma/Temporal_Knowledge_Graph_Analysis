import numpy as np
import pandas as pd
from bokeh.layouts import column
from matplotlib import pyplot as plt
import seaborn as sns

from data_handling.network import get_graph_attributes
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, Range1d, BoxZoomTool
from data_handling.network import *
from sklearn.neighbors import KernelDensity



if __name__ == '__main__':
    graph, node_df, edge_df = get_graph_attributes()

    # Convert the datetime data to numerical values (timestamps)
    #edge_df_mod = edge_df
    edge_df_mod = edge_df.loc[(edge_df[EDGE_COLUMN_START_DATE] >= '2000-01-01') & (edge_df[EDGE_COLUMN_END_DATE] < edge_df[EDGE_COLUMN_END_DATE].max())]
    start_time_data: pd.Series = edge_df_mod[EDGE_COLUMN_START_DATE].astype(np.int64) / 1e9
    end_time_data: pd.Series = edge_df_mod[EDGE_COLUMN_END_DATE].astype(np.int64) / 1e9

    def kernel_density_estimate(data: np.ndarray, x: np.ndarray, bandwidth_days: Union[int, float], kernel = 'tophat'):
        bandwidth = bandwidth_days * 24 * 60 * 60
        if kernel == 'gaussian':
            bandwidth /= 2
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(data)
        y = np.exp(kde.score_samples(x))
        return y

    x_min = min(start_time_data.min(), end_time_data.min())
    x_max = max(start_time_data.max(), end_time_data.max())
    x = np.linspace(x_min, x_max, 1000)
    y_start = kernel_density_estimate(start_time_data.values, x, 30)
    y_end = kernel_density_estimate(end_time_data.values, x, 30)
    y_min = min(y_start.min(), y_end.min())
    y_max = max(y_start.max(), y_end.max()) * 1.1
    # don't ask me why, but pd.to_timestamp does not work, whilst this does...
    X = pd.Series([datetime.datetime.fromtimestamp(ts) for ts in x])
    source = ColumnDataSource(data={
        'X': X,
        'Created Edges': y_start,
        'Deleted Edges': y_end
    })
    p1 = figure(title="KDE of New and Deleted Edges", background_fill_color="#fafafa", tools="pan,wheel_zoom,reset", sizing_mode='stretch_width',
               x_axis_type='datetime', x_range=Range1d(X.min(), X.max(), bounds=(X.min(), X.max())), y_range=Range1d(y_min, y_max, bounds=(y_min, y_max)))
    p1.varea(x='X', y1='Created Edges', y2=0, source=source, fill_alpha=0.4, fill_color='green', legend_label='Created Edges')
    p1.line(x='X', y='Created Edges', source=source, alpha=1.0, color='black')
    p1.varea(x='X', y1='Deleted Edges', y2=0, source=source, fill_alpha=0.4, fill_color='red', legend_label='Deleted Edges')
    p1.line(x='X', y='Deleted Edges',  source=source, alpha=1.0, color='black')
    p1.xaxis.axis_label = 'Time'
    p1.yaxis.axis_label = 'Amount'
    p1.grid.grid_line_color = "white"
    p1.legend.location = "top_left"
    # p.xaxis.major_label_orientation = 3.14 / 4  # rotate labels

    p1.xaxis[0].formatter = DatetimeTickFormatter(years="%Y")
    p1.add_tools(BoxZoomTool(match_aspect=True))

    # Compute the histogram data with numpy (because bokeh has no histogram :/ )
    hist_start, edges_start = np.histogram(start_time_data, bins=24)
    edges_start_datetime = pd.to_datetime(edges_start, unit='s')
    hist_end, edges_end = np.histogram(end_time_data, bins=12)
    edges_end_datetime = pd.to_datetime(edges_end, unit='s')

    # ColumnDataSource
    source_start = ColumnDataSource(data=dict(
        top=hist_start,
        left=edges_start_datetime[:-1],
        right=edges_start_datetime[1:]
    ))
    source_end = ColumnDataSource(data=dict(
        top=hist_end,
        left=edges_end_datetime[:-1],
        right=edges_end_datetime[1:]
    ))
    x_range = (np.min(np.concatenate([edges_start_datetime, edges_end_datetime])),
               np.max(np.concatenate([edges_start_datetime, edges_end_datetime])))
    y_range = (0, np.max(np.concatenate([hist_start, hist_end])))
    p = figure(title="Histogram of New and Deleted Edges", background_fill_color="#fafafa", tools="pan,wheel_zoom,reset",
               x_axis_type='datetime', x_range=Range1d(*x_range, bounds=x_range), y_range=Range1d(*y_range, bounds=y_range))

    # quad glyph (from the histogram from np)
    p.quad(top='top', bottom=0, left='left', right='right', source=source_start,
           fill_color="green", line_color="white", alpha=0.5, legend_label="New Edges")

    p.quad(top='top', bottom=0, left='left', right='right', source=source_end,
           fill_color="red", line_color="white", alpha=0.5, legend_label="Deleted Edges")

    p.y_range.start = 0
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Amount'
    p.grid.grid_line_color = "white"
    p.legend.location = "top_left"
    # p.xaxis.major_label_orientation = 3.14 / 4  # rotate labels

    p.xaxis[0].formatter = DatetimeTickFormatter(years="%Y")

    show(column(p, p1, sizing_mode='stretch_width'))

    # ----------------------- Hist Plot: deleted edges over time with type -----------------------
    edge_df = edge_df.loc[np.logical_and(edge_df[EDGE_COLUMN_START_DATE] != edge_df[EDGE_COLUMN_START_DATE].min(),
                                         edge_df[EDGE_COLUMN_END_DATE] != edge_df[EDGE_COLUMN_END_DATE].max())]

    EDGE_COLUMN_SOURCE_TYPE = 'source_type'
    EDGE_COLUMN_TARGET_TYPE = 'target_type'

    edge_df = edge_df.merge(node_df[[NODE_COLUMN_TYPE]].rename(columns={NODE_COLUMN_TYPE: EDGE_COLUMN_SOURCE_TYPE}),
                            how='left',
                            left_on=EDGE_INDEX_SOURCE,
                            right_index=True)

    edge_df = edge_df.merge(node_df[[NODE_COLUMN_TYPE]].rename(columns={NODE_COLUMN_TYPE: EDGE_COLUMN_TARGET_TYPE}),
                            how='left',
                            left_on=EDGE_INDEX_TARGET,
                            right_index=True)

    # sns.histplot(data=edge_df, x=EDGE_COLUMN_END_DATE, bins=50, kde=True, hue=EDGE_COLUMN_TYPE, multiple='stack')
    g = sns.catplot(data=edge_df, x=EDGE_COLUMN_END_DATE, y=EDGE_COLUMN_TYPE, hue=EDGE_COLUMN_TARGET_TYPE)
    g.fig.suptitle('Deleted edges over time sorted by type')
    plt.xticks(rotation=45)
    g.set_axis_labels("", "Edge Type")
    # plt.legend(title='Target Node Type')
    plt.show()
