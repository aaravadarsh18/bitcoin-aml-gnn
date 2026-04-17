import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


def load_elliptic_data(base_path="data/elliptic_bitcoin_dataset/"):
    print("Loading dataset...")

    # Load files
    features = pd.read_csv(base_path + "elliptic_txs_features.csv", header=None)
    edges = pd.read_csv(base_path + "elliptic_txs_edgelist.csv")
    classes = pd.read_csv(base_path + "elliptic_txs_classes.csv")

    classes.columns = ["txId", "class"]

    # Merge
    df = features.merge(classes, left_on=0, right_on="txId")

    # Encode labels
    df["class"] = df["class"].map({"unknown": -1, "1": 1, "2": 0})

    # Remove unknown
    df = df[df["class"] != -1]

    print(f"Total labeled transactions: {len(df)}")

    # Sort
    df = df.sort_values(by=0).reset_index(drop=True)

    # Mapping
    id_map = {tx_id: idx for idx, tx_id in enumerate(df[0].values)}

    # Features
    X = df.drop(columns=["class", "txId"]).values
    y = df["class"].values

    # FIX: Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    # Build edges
    edge_list = []

    src_nodes = edges["txId1"].values
    dst_nodes = edges["txId2"].values

    for src, dst in zip(src_nodes, dst_nodes):
        if src in id_map and dst in id_map:
            edge_list.append([id_map[src], id_map[dst]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    print(f"Total nodes: {x.shape[0]}")
    print(f"Total edges: {edge_index.shape[1]}")

    data = Data(x=x, edge_index=edge_index, y=y)

    return data