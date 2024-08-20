import seaborn as sns
import matplotlib.pyplot as plt
from data_handling.network import *
from utils import get_logger

if __name__ == '__main__':
    graph, node_df, edge_df = get_graph_attributes(merge_node_dates=True)
    company_df = node_df.loc[node_df[NODE_COLUMN_TYPE].apply(lambda x: 'Company' in x)]

    edge_types = [edge_type.lower().replace(' ', '_') for edge_type in edge_df[EDGE_COLUMN_TYPE].unique()]
    target_counts = edge_df.groupby([EDGE_INDEX_TARGET, EDGE_COLUMN_TYPE]).size().unstack(fill_value=0)

    target_counts.columns = target_counts.columns.str.lower().str.replace(' ', '_')

    for edge_type in edge_types:
       company_df = company_df.merge(target_counts[edge_type], left_on='id', right_index=True, how='left').rename(columns={edge_type: f'{edge_type}_counter'})
    # print(company_df.describe(include='all'))

    company_df = company_df.drop(columns=['family_relation_counter', 'dob', 'TradeDescription', 'HeadOfOrg', 'PointOfContact', 'ProductServices', 'country'])
    print(company_df.loc[company_df['revenue'] > 0].describe(include='all'))
    g= sns.pairplot(company_df, diag_kind="kde", hue=NODE_COLUMN_TYPE)
    g.map_lower(sns.kdeplot, levels=3, color=".2")
    plt.show()

