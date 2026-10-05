"""Experimental: converting lift states to graphs for a GNN policy. Needs the `gnn` extra."""

import time

import torch
from torch.nn import Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Sequential

from elevator_rl.sim import LiftSim, LiftState


def convert(state: LiftState) -> Data:

    node_data = torch.tensor([[0], [1]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.tensor([[2], [3]], dtype=torch.float)
    return Data(x=node_data, edge_index=edge_index, edge_attr=edge_attr)


def simple_model(in_channels, out_channels):
    model = Sequential(
        "x, edge_index",
        [
            (GCNConv(in_channels, 64), "x, edge_index -> x"),
            ReLU(inplace=True),
            (GCNConv(64, 64), "x, edge_index -> x"),
            ReLU(inplace=True),
            Linear(64, out_channels),
        ],
    )
    return model


def simple_graph():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    node_data = torch.tensor([[0], [1]], dtype=torch.float)
    edge_attr = torch.tensor([[2], [3]], dtype=torch.float)
    return Data(x=node_data, edge_index=edge_index, edge_attr=edge_attr)


if __name__ == "__main__":
    sim = LiftSim()
    for _ in range(100):
        sim.sample_passengers()

    t0 = time.perf_counter()
    graph = convert(sim.state())
    dt = time.perf_counter() - t0
    print(dt)

    t0 = time.perf_counter()
    graph = convert(sim.state())
    dt = time.perf_counter() - t0
    print(dt)
    print(graph.validate())
    print(graph)
