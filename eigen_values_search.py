import itertools
import math
import os
import sys
from time import time  # timing package

import dimod
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy import sparse
from scipy.sparse.linalg import eigs

from devil_graphs import devil_graphs
from stable_set_formulations import laserre, proposed


def eigen_values_search(formulation, prob, nodes, K0, K, c1, overwrite_files, draw_plots, generate_plots):

    four_node_edges = [[],
                    [(0, 1)],
                    [(0, 1), (0, 3)], [(0, 1), (2, 3)],
                    [(0, 1), (1, 2), (2, 3)], [(0, 1), (0, 2), (0, 3)], [(0, 1), (1, 2), (2, 0)],
                    [(0, 1), (1, 2), (2, 3), (3, 0)], [(0, 1), (1, 2), (2, 0), (3, 0)],
                    [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
                    [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
                    ]


    dir_path = os.path.dirname(os.path.abspath(__file__))
    mingap_path = os.path.join(dir_path, "results/mingap/")
    if formulation == 'nonlinear':
        results_path = os.path.join(mingap_path, "eigenvalues")
    elif formulation == 'linear':
        results_path = os.path.join(mingap_path, "eigenvalues_l")


    solfile = os.path.join(mingap_path, "erdos_" + str(
        formulation) + "_" + str(int(100*prob)) + "_" + str(c1))

    # Pauli matrices

    Id = [[1, 0], [0, 1]]
    s_x = [[0, 1], [1, 0]]
    s_y = [[0, -1j], [1j, 0]]
    s_z = [[1, 0], [0, -1]]

    # A and B functions such that H = A(s)Hd + B(s)Hp. These are specific to D-Wave 2000Q at NASA.
    # These are old fits by Zoe we will instead use the data provided by DWave directly

    def a_function(s):
        return (2 / math.pi) * (
            math.exp(3.173265 * s) * (207.253 * (1 - s) ** 9 + 203.843 * (1 - s) ** 7 - 380.659 * (1 - s) ** 8))

    def b_function(s):
        return (2 / math.pi) * (0.341734 + 6.713285 * s + 32.9702 * s * s)

    # s_list = np.arange(0, 1.0001, 0.001)

    # List of s values you want to use, between 0 and 1


    df = pd.read_excel(os.path.join(dir_path, "09-1192A-C_DW_2000Q_2_1_processor-annealing-schedule.xlsx"),
                    sheet_name='DW_2000Q_2_processor-annealing-')

    s_list = df['s']

    # Nested Kronecker that we need for the individual Pauli matrices
    # Performs a Kronecker product on a series of matrices all stored in the variable a

    def nested_kronecker(a):
        if len(a) == 2:
            return np.kron(a[0], a[1])
        else:
            return np.kron(a[0], nested_kronecker(a[1:]))

    # Builds a Pauli matrix acting on qubit j (variable mat is the Pauli matrix we want applied on j, s_x, s_y or s_z)

    def individual_pauli(mat, j, num_of_spins):
        ops = []
        for _ in range(j):
            ops.append(Id)
        ops.append(mat)
        for _ in range(num_of_spins - j - 1):
            ops.append(Id)
        return nested_kronecker(ops)


    columns = ['id', 'eigenval', 'mingap']
    solutions = pd.DataFrame(columns=columns)

    for k in range(K0, K):
        G = nx.Graph()
        # G = nx.cycle_graph(k)
        # G.add_nodes_from([0, 1, 2, 3])
        # G.add_edges_from(four_node_edges[k])
        # G, alpha = devil_graphs(k)
        G = nx.erdos_renyi_graph(n=nodes, p=prob, seed=k)
        temp = dict()

        tolerance = 1 #GHz
        idx_start = 0
        idx_end = len(s_list)
        idx_interval = 100
        decrease = 10

        eigenvals = []
        s_plot = []
        mingap_ss = []
        index_ss = []

        eigenfilename = "erdos_" + str(prob) + "_" + str(k) + "_M" + str(c1)
        idxfilename = "erdos_idx_" + str(prob) + "_" + str(k) + "_M" + str(c1)

        print(eigenfilename)

        eigenfile = os.path.join(results_path, eigenfilename + ".npy")
        idxfile = os.path.join(results_path, idxfilename + ".npy")

        ti = time()  # start timer

        # If file exists, the load them
        if os.path.exists(eigenfile) and os.path.exists(idxfile) and not overwrite_files:
            eigenvals = np.load(eigenfile, allow_pickle=True)
            s_plot = np.load(idxfile, allow_pickle=True)

            for i in range(1, eigenvals.shape[1]):
                if min(abs(eigenvals[:, i] - eigenvals[:, 0])) < tolerance:
                    pass
                else:
                    break
            gap = eigenvals[:, i] - eigenvals[:, 0]
            mingap = min(gap)
            mingap_idx = np.argmin(gap)
            mingap_s = s_plot[mingap_idx]
        else:            
        # Calculate full Hamiltonian for the s values in your list and get the eigenvalues and gap
            print('Running eigenvalues')
            break

            # Find the maximum independent set, which is known in this case to be of length 3
            if formulation == 'nonlinear':
                Q, offset = proposed(G, M=c1)
            elif formulation == 'linear':
                Q, offset = laserre(G, M=c1)
            bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset=offset)
            h, J, offset = bqm.to_ising()
            indices = dict((key, idx)
                        for (idx, key) in enumerate(bqm.linear.keys()))

            # Number of qubits
            n = len(list(bqm.linear))

            # Build driver and problem Hamiltonians
            Hd = 0
            for i in range(n):
                Hd = np.add(Hd, individual_pauli(s_x, i, n))

            Hp = 0
            for pair in bqm.quadratic:
                Hp = np.add(Hp,
                            J[pair] * individual_pauli(s_z, indices[pair[0]], n) * individual_pauli(s_z, indices[
                                pair[1]], n))

            for qubit in bqm.linear:
                Hp = np.add(Hp, h[qubit] *
                            individual_pauli(s_z, indices[qubit], n))
            # If your Ising Hamiltonian also has an external field you can add it with terms of the form individual_pauli(s_z, i, n)
            while idx_interval >= 1:
                for idx in range(idx_start, idx_end, idx_interval):
                    if df.loc[idx, 's'] in mingap_ss or idx in index_ss:
                        pass
                    else:
                        H = df.loc[idx, 'A(s) (GHz)'] * Hd + df.loc[idx, 'B(s) (GHz)'] * Hp
                        if n >= 16:
                            sH = sparse.csc_matrix(H)
                            eig = eigs(sH, 10, which='SR', tol=1e-3, return_eigenvectors=False)
                        else:
                            eig = la.eigvalsh(H)
                        eig = np.sort(eig.real)
                        eigenvals.append(eig)
                        # np.append(eig, eigenvals, axis=0)
                        s_plot.append(df.loc[idx, 's'])

                s_plot, eigenvals = (list(t) for t in zip(*sorted(zip(s_plot, eigenvals))))

                eigenvalues = np.array(eigenvals)

                for i in range(1, eigenvalues.shape[1]):
                    if min(abs(eigenvalues[:, i] - eigenvalues[:, 0])) < tolerance:
                        pass
                    else:
                        break
                print('Minimup gap computed with ' + str(i) + 'th eigenvalue')
                gap = eigenvalues[:, i] - eigenvalues[:, 0]

                mingap = min(gap)
                mingap_idx = np.argmin(gap)
                mingap_s = s_plot[mingap_idx]
                mingap_ss.append(mingap_s)
                mingap_index = df.index[df['s'] == mingap_s].tolist()

                print('Minimup gap: ' + str(mingap) + ' GHz')
                print('At s= ' + str(mingap_s))

                idx_end = mingap_index[0] + idx_interval
                idx_start = mingap_index[0] - idx_interval
                index_ss.append(idx_start)
                index_ss.append(idx_end)
                index_ss.append(mingap_index[0])
                idx_interval = int(idx_interval / decrease)

            tf = time()  # start timer
            print("simulation took {0:.4f} sec".format(tf - ti))
            temp['time'] = tf - ti
            eigenvals = np.array(eigenvals)
            np.save(eigenfile, eigenvals)
            np.save(idxfile, s_plot)

        temp['id'] = eigenfilename
        temp['eigenval'] = i
        temp['mingap'] = mingap

        if generate_plots:

            # plt.figure(0)
            # df.plot(x='s', y=['A(s) (GHz)', 'B(s) (GHz)'])
            # plt.xlabel('Adimensional time $s$')
            # plt.ylabel('Energy (GHz)')
            # plt.legend()
            # plt.xlim(0, 1)
            #
            plt.figure(1)
            plt.plot(s_plot, eigenvals)
            plt.xlabel('Adimensional time $s$')
            plt.ylabel('Hamiltonian eigenvalues')
            plt.xlim(0, 1)
            plt.savefig(os.path.join(results_path, "erdos_" + str(prob) +
                                    "_" + str(k) + "_M" + str(c1) + "all_eigs.png"))
            #
            plt.figure(2)
            plt.plot(s_plot, eigenvals, '0.75')
            plt.plot(s_plot, eigenvals[:, i])
            plt.plot(s_plot, eigenvals[:, 0])
            plt.axvline(mingap_s, color='k', linestyle="dashed")
            plt.xlabel('Adimensional time $s$')
            plt.ylabel('Hamiltonian eigenvalues')
            plt.xlim(0, 1)
            plt.savefig(os.path.join(results_path, "erdos_" + str(prob) +
                                    "_" + str(k) + "_M" + str(c1) + "eigs.png"))

            plt.figure(3)
            plt.plot(s_plot, gap, '*')
            plt.vlines(mingap_s, 0, mingap, linestyle="dashed")
            plt.hlines(mingap, 0, mingap_s, linestyle="dashed")
            plt.xlabel('Adimensional time $s$')
            plt.ylabel('Eigenvalues gap $\Delta$')
            plt.xlim(0, 1)
            plt.ylim(0, None)
            plt.savefig(os.path.join(results_path, "erdos_" + str(prob) +
                                    "_" + str(k) + "_M" + str(c1) + "gap.png"))

            if draw_plots:
                plt.show()

            plt.figure(1).clf()
            plt.figure(2).clf()
            plt.figure(3).clf()

        solutions = solutions.append(temp, ignore_index=True)
        solutions.to_csv(solfile + ".csv")
    sol_total = pd.DataFrame.from_dict(solutions)
    sol_total.to_excel(solfile + ".xlsx")


if __name__ == "__main__":
    # formulation = 'nonlinear'
    # formulation = 'linear'
    # prob = 0.25
    nodes = 5
    K0 = 0
    K = 100
    # c1 = 1
    overwrite_files = False
    draw_plots = False
    generate_plots = False
    c1s = [1,2,5]
    probs = [0.25, 0.50, 0.75]
    formulations = ['nonlinear', 'linear']

    for c1 in c1s:
        for prob in probs:
            for formulation in formulations:
                eigen_values_search(formulation, prob, nodes, K0, K,
                                    c1, overwrite_files, draw_plots, generate_plots)
