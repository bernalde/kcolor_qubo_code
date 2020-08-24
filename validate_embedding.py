import networkx as nx
from itertools import combinations


def validate_embedding(X, Y, embed):
    """
    validate_embedding(X, Y, embed):
    Returns a boolean variable validating if embed is a feasible embedding of Y in X

    Args:

        X: an iterable of label pairs representing the edges in the hardware graph, or a NetworkX Graph

        Y: an iterable of label pairs representing the edges in the problem graph, or a NetworkX Graph

        embed: dict. Dictionary with keys being nodes in Y and values being list of nodes of X in each vertex model

    Returns:

        flag: boolean flag  representing if embed is a feasible embedding of Y in X
    """

    flag = True
    nodes_X = set(map(int, X.nodes()))
    nodes_Y = set(map(int, Y.nodes()))
    edges_X = set(tuple(int(x) for x in edge) for edge in X.edges)
    edges_Y = set(tuple(int(y) for y in edge) for edge in Y.edges)
    embed = {int(variable): [int(qubit) for qubit in vertex_model] for variable, vertex_model in embed.items()}

    # Validate representability condition (all variables in Y should be in X)
    for node in nodes_Y:
        if node not in embed.keys():
            flag = False
            print('Node ' + str(node) + ' of Y not found in embedding')
            return flag
        qubits = embed[node]
        for i in range(len(qubits)):
            if not any((qubits[i], j) in edges_X or (j, qubits[i]) in edges_X for j in qubits) and len(qubits) != 1:
                flag = False
                print('Node ' + str(node) + ', qubit ' + str(qubits[i]) + ' disconnected from vertex model')
                return flag

    # Validate connectivity condition (all edges in Y should have an edge on X)
    if not all((any((j, k) in edges_X or (k, j) in edges_X for j in embed[n1] for k in embed[n2])) for (n1, n2) in
               edges_Y):
        flag = False
        print('Not all edges in Y are present in X')
        for edge in edges_Y:
            n1 = edge[0]
            n2 = edge[1]
            if not any((j, k) in edges_X or (k, j) in edges_X for j in embed[n1] for k in embed[n2]):
                print('Edge ' + str(edge) + ' of Y not found in embedding')

    # Validate not duplicate representation (every node in X can only be in a single vertex model phi(y), y in Y)
    for node in nodes_X:
        if any(node in embed[n1] and node in embed[n2] for (n1, n2) in combinations(nodes_Y, 2)):
            flag = False
            print('Node ' + str(node) + ' of X in two different vertex models')
    return flag


# Y = nx.read_edgelist('tests/433.edgelist')
# Hembedding = {0: [9, 17, 21, 25], 1: [10, 26], 2: [8, 24, 30], 3: [1, 5, 13], 4: [6, 14], 5: [11, 27], 6: [19],
              # 7: [4, 12], 8: [3, 7, 15], 9: [23, 31], 10: [29], 11: [0], 12: [2, 18, 20, 28], 13: [16], 14: [22]}
# Hembedding2 = {0: [9, 17, 21, 25], 1: [10, 26], 2: [8, 24, 30], 3: [1, 5, 13], 4: [6, 14], 5: [11, 27], 6: [19],
               # 7: [4, 12], 8: [3, 7, 15], 9: [23, 31], 10: [2], 11: [0], 12: [2, 18, 20, 28], 13: [16], 14: [22]}
# Hembedding3 = {0: [9, 17, 21, 25], 1: [10, 26], 2: [8, 24, 30], 3: [1, 5, 13], 4: [6, 14], 5: [11, 27], 6: [19],
               # 8: [3, 7, 15], 9: [23, 31], 10: [29], 11: [0], 12: [2, 18, 20, 28], 13: [16], 14: [22]}
# Hembedding4 = {0: [9, 17, 21, 25], 1: [10, 26], 2: [8, 30], 3: [1, 5, 13], 4: [6, 14], 5: [11, 27], 6: [19],
               # 8: [3, 7, 15], 9: [23, 31], 10: [29], 11: [0], 12: [2, 18, 20, 28], 13: [16], 14: [22]}
# X2 = nx.read_edgelist('tests/chimera_2_2.edgelist')

# print(validate_embedding(X2, Y, Hembedding4))
