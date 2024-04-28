import json
import os
from collections import defaultdict

import pandas as pd
import numpy as np


class Graph:
    def __init__(self, vertices, metaPath='../KSS/meta_data/'):
        self.graph = defaultdict(list)
        self.V = vertices
        self.metaPath = metaPath

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def addEdges(self, edges):
        for (u, v) in edges:
            self.addEdge(u, v)

    def topologicalSortUtil(self, v, visited, stack: list):
        visited[v] = True
        for i in self.graph[v]:
            if not visited[i]:
                self.topologicalSortUtil(i, visited, stack)
        stack.insert(0, v)

    def topologicalSort(self):
        visited = [False] * self.V
        stack = []
        for i in range(self.V):
            if not visited[i]:
                self.topologicalSortUtil(i, visited, stack)
        return stack

    def writeMetaData(self, stack, edges):
        items = {_: {'knowledge': _} for _ in range(self.V)}
        with open(os.path.join(self.metaPath, 'items.json'), 'w') as f:
            json.dump(items, f)
        with open(os.path.join(self.metaPath, 'learning_order.json'), 'w') as f:
            json.dump(stack, f)
        df = pd.DataFrame(edges)
        df.to_csv(os.path.join(self.metaPath, 'knowledge_structure.csv'), header=False, index=False)


if __name__ == '__main__':
    edges_path = '../../../data_tmp/junyi/data/prerequisite.json'
    with open(edges_path) as f:
        edges = json.load(f)
    nodes = np.max(edges) - np.min(edges) + 1
    nodesNotInGraph = np.setdiff1d(np.arange(nodes), edges)
    print(nodesNotInGraph)
    preAndSuc = np.random.choice(nodes, len(nodesNotInGraph))
    newEdges = np.stack([nodesNotInGraph, preAndSuc[:len(nodesNotInGraph)]], axis=-1)
    randomShuffle = np.random.randint(0, 2, len(newEdges))
    randomShuffle = np.stack([randomShuffle, 1 - randomShuffle], -1)
    newEdges = np.take_along_axis(newEdges, randomShuffle, -1)
    edges = np.concatenate([edges, newEdges], 0)
    # edges = [[0, 1], [0, 2], [1, 3], [2, 4], [2, 8], [3, 4], [4, 8], [5, 4], [5, 9], [6, 7], [7, 8], [8, 9]]

    g = Graph(np.max(edges) - np.min(edges) + 1)
    g.addEdges(edges.tolist())
    stack = g.topologicalSort()
    g.writeMetaData(stack, edges)
