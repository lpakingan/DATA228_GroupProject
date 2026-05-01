#!/usr/bin/env python3
"""
Build a user similarity graph where nodes = users and edges = connections between them. Run a community detection algorithm to find clusters of densely connected users. Filter out small clusters below cluster size 3.

Reads:
  - outputs/user_similarity_edges.parquet
Writes:
  - outputs/user_clusters.parquet
"""

from pathlib import Path
import pandas as pd
import networkx as nx
from networkx.algorithms import community

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# set paths
INPUT_EDGES = PROJECT_ROOT / "outputs" / "user_similarity_edges.parquet"
OUTPUT_CLUSTERS = PROJECT_ROOT / "outputs" / "user_clusters.parquet"

# set minimum cluster size = 3
MIN_CLUSTER_SIZE = 3

def main():
    input_edges = INPUT_EDGES
    
    # read input parquet to dataframe
    edge_df = pd.read_parquet(input_edges)

    # create graph object
    edge_graph = nx.from_pandas_edgelist(edge_df, source='user_id_a', target='user_id_b')

    # define communities using networkx
    communities = community.louvain_communities(edge_graph)

    # create list of clusters from communities
    clusters = []
    for cluster_id, members in enumerate(communities):
        if len(members) >= MIN_CLUSTER_SIZE:
            for user_id in members:
                clusters.append([user_id, cluster_id, len(members)])

    # convert to parquet and save output
    clusters_df = pd.DataFrame(clusters, columns=['user_id', 'cluster_id', 'cluster_size'])
    OUTPUT_CLUSTERS.parent.mkdir(parents=True, exist_ok=True)
    clusters_df.to_parquet(OUTPUT_CLUSTERS, index=False)
    print(f"Output saved to: {OUTPUT_CLUSTERS}")


if __name__ == "__main__":
    main()