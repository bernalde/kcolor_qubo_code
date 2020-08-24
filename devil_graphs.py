import networkx as nx
import numpy as np


def devil_graphs(n):
    """
    devil_graphs(n):
    Returns a graph defined in paper [ADD REFERENCE] known as devil graphs.
    This graphs are interesting for th e stable set problem since the stable set number can be computed analytically as
    n+1, where n is the rank of the adjacency matrix A.
    The adjacency matrix is defined as
    A =[0  1  en'   zn'   zn';
        1  0  zn'   en'   zn';
        en zn Zn    Jn-In In;
        zn en Jn-In Zn    In;
        zn zn In    In    Zn];
    Where the En and zn vector are ones and zero vectors of size n, and the Jn, In, and Zn matrices are ones, identity
    and zero matrices of size nxn.


    Args:

        n: rank of the resulting adjacency matrix

    Returns:

        Z: NetworkX undirected graph defined by the adjacency matrix A of the devil graph of rank n

        alpha: stable set number computed as n + 1
    """

    Jn = np.ones((n,n))
    In = np.eye(n)
    en = np.ones((n,1))
    zn = np.zeros((n,1))
    Zn = np.zeros((n,n))

    A = np.block([[0, 1, np.transpose(en), np.transpose(zn), np.transpose(zn)],
              [1, 0, np.transpose(zn), np.transpose(en), np.transpose(zn)],
              [en, zn, Zn, Jn-In, In],
              [zn, en, Jn-In, Zn, In],
              [zn, zn, In, In, Zn]]);

    # We duplicate the upper-triangular part of the matrix since the graph is undirected
    Z = nx.from_numpy_matrix(A)

    alpha = n + 1

    return Z, alpha


if __name__ == "__main__":
    G, alpha = devil_graphs(3)
    print(G.edges())
    print(G.nodes())
    print(nx.adjacency_matrix(G).todense())
    print(alpha)