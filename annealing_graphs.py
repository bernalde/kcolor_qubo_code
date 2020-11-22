import matplotlib.pyplot as plt
import random
import os

from stable_set_formulations import laserre, proposed
from devil_graphs import devil_graphs

import dimod
from dwave.embedding import chimera, pegasus
from dwave.system import DWaveSampler
import dwave_networkx as dnx

import subprocess
import time
import statistics
# import find_embedding
import minorminer as mm
import networkx as nx
import numpy as np
import pandas as pd
import itertools
import ast


from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
import pickle
import dimod
import neal
from collections import Counter


DRY_RUN = False


def annealing(instance, TEST, prob=0.25, seed=42,
              K=1, overwrite_pickles=False, draw_figures=False,
              annealing_time=[2, 20, 200],  # Microseconds
              chain_strenghts=[0.1, 0.2, 0.5, 1, 2, 5, 10],
              samples=100):

    

    # Store the results in csv files
    dir_path = os.path.dirname(os.path.abspath(__file__))
    embedding_path = os.path.join(dir_path, "results/embedding/")
    results_path = os.path.join(dir_path, "results/dwave_chimera/")

    if instance == 'spreadsheet':
        # spreadsheet_name = "erdos_" + \
        #     str(int(100*prob)) + '_graphs_with_opt.xlsx'
        # spreadsheet_name = "erdos_" + \
        #     str(int(100*prob)) + '_graphs.xlsx'
        # spreadsheet_name = "erdos_" + \
        #     str(int(100*prob)) + '_graphs_' + str(K) + '.xlsx'
        spreadsheet_name = instance + \
            str(int(100*prob)) + '_embedding_pegasus_' + str(K) + '.xlsx'
        spreadsheet_name = os.path.join(embedding_path, spreadsheet_name)
        input_data = pd.read_excel(spreadsheet_name)
        n0 = 0
        N = input_data.shape[0]
        file_name = instance + str(int(100*prob)) + '_annealing_pegasus_' + str(K)
        K = 1
    elif instance == 'erdos':
        # Parameters for random graphs
        N = 51
        n0 = 3
        K = 5  # number of graphs
        seed = 1
        instance = "erdos_" + str(int(100*prob))
        file_name = instance + '_embedding'
    elif instance == 'cycle':
        # Parameters for cycles
        N = 200
        n0 = 100
        file_name = instance + '_embedding'
    elif instance == 'devil':
        # Parameters for devil graphs
        N = 10
        n0 = 8
        file_name = instance + '_embedding'
    else:
        print("Graph type not implemented")
        return()

    # If results directory does not exist, we create it
    if not(os.path.exists(results_path)):
        print('Results directory ' + results_path +
              ' does not exist. We will create it.')
        os.makedirs(results_path)

    file_name = os.path.join(results_path, file_name)

    # time horizons and time limit in seconds
    if TEST:
        random.seed(seed)
        time_limit = 120
        # Number of times the heuristic is run
        n_heur = 100
    else:
        time_limit = 3600
        # Number of times the heuristic is run
        n_heur = 1000

    # Graph corresponding to D-Wave 2000Q or Pegasus
    qpu = DWaveSampler(solver='DW_2000Q_6')
    # qpu = DWaveSampler(solver='Advantage_system1.1')
    qpu_edges = qpu.edgelist
    qpu_nodes = qpu.nodelist
    # X = dnx.chimera_graph(16, node_list=qpu_nodes, edge_list=qpu_edges)
    # X = dnx.pegasus_graph(16, node_list=qpu_nodes, edge_list=qpu_edges)
    X = nx.Graph()
    X.add_edges_from(qpu_edges)
    # nx.write_edgelist(X, os.path.join(results_path,"X.edgelist"))
    


    for k in range(K):
        for n in range(n0, N):
            print(n)

            # Graph generation
            if instance == "spreadsheet":
                # Generate graph from spreadsheet
                Input = nx.Graph()
                edges = ast.literal_eval(input_data.edges[n])
                Input.add_edges_from(edges)
                alpha = 0

                # Import embeddings
                best_embedding = dict()
                best_embedding['n'] = ast.literal_eval(input_data.heur_best_embed_n[n]
                                                )
                best_embedding['l'] = ast.literal_eval(input_data.heur_best_embed_l[n]
                                                )




            elif instance == "cycle":
                # Cycle graphs
                Input = nx.cycle_graph(n)
                temp['id'] = "cycle_" + str(n)
                alpha = np.floor(n / 2)
            elif instance == "devil":
                # Devil graphs
                Input, alpha = devil_graphs(n)
                temp['id'] = "devil_rank_" + str(n)
            else:
                # Random graphs
                Input = nx.erdos_renyi_graph(n, prob)
                temp['id'] = "random_" + str(n) + "_" + str(prob) + "_" + str(k)
                alpha = 0

            # Input graph parameters

            # Define samplers: exact, simulated annealing and Dwave (with automatic embedding)
            samplers = dict()
            samplers['simann'] = neal.SimulatedAnnealingSampler()
            samplers['dwave_embed'] = EmbeddingComposite(DWaveSampler())
            best_embed = False

            if DRY_RUN:
                pass
            else:

                # Problem reformulations
                # Nonlinear and Linear (Laserre) reformulation
                reforms = ['n', 'l']

                # Set up experiments in the sense of which embedding to use
                experiments = ['', '_fixed']

                columns = ['formulation', 'chain_strength',
                           'embedding', 'chain_breaks', 'chain_breaks_err', 'min_fraction',
                           'embedding_fixed', 'chain_breaks_fixed', 'chain_breaks_err_fixed', 'min_fraction_fixed'
                           ]
                results = pd.DataFrame(columns=columns)


                for reform in reforms:

                    min_matrix = np.zeros(
                        [len(annealing_time), len(chain_strenghts), len(experiments)])
                    if reform == 'n':
                        Q, offset = proposed(Input, M=1, draw=False)
                    elif reform == 'l':
                        Q, offset = laserre(Input, draw=False)

                    bqm = dimod.BinaryQuadraticModel.from_qubo(
                        Q, offset=offset)
                    edges = list(itertools.chain(
                        bqm.quadratic, ((v, v) for v in bqm.linear)))

                    G = nx.Graph()
                    G.add_edges_from(edges)

                    # Add fixed embedding sampler if allowed
                    if best_embed:
                        fail = 0
                        min_length = X.number_of_nodes()
                        for n_h in range(n_heur):
                            h_embed = mm.find_embedding(
                                G, X, timeout=time_limit, random_seed=n_h)
                            count = 0
                            for _, value in h_embed.items():
                                count += len(value)
                            if count == 0:
                                fail += 1
                            elif count < min_length:
                                best_embed = h_embed
                                min_length = count
                            samplers['dwave'] = FixedEmbeddingComposite(
                                DWaveSampler(), embedding=best_embed)
                        else:
                            samplers.pop('dwave', None)

                    pickle_path = os.path.join(results_path, file_name, str(n), reform)
                    if not(os.path.exists(pickle_path)):
                        print('Pickled results directory ' + pickle_path +
                            ' does not exist. We will create it.')
                        os.makedirs(pickle_path)
                    
                    idx_i = 0
                    for ann_time in annealing_time:
                        results_name = instance + '_chains_' + \
                            reform + '_' + str(ann_time) + '.csv'
                        results_name = os.path.join(results_path, results_name)


                        temp = dict()

                        idx_j = 0
                        for c in chain_strenghts:

                            # Identification
                            temp['formulation'] = reform
                            temp['chain_strength'] = c

                            idx_k = 0
                            for kind in experiments:
                                pickle_name = "chain_str_" + \
                                    str(c) + "_" + str(ann_time) + kind + ".p"
                                pickle_name = os.path.join(pickle_path, pickle_name)
                                if os.path.exists(pickle_name) and not overwrite_pickles:
                                    response = pickle.load(open(pickle_name, "rb"))
                                else:
                                    # chain_strength = max(Q.min(), Q.max(), key=abs)*c
                                    chain_strength = max(
                                        abs(min(Q.values())), abs(max(Q.values())))*c
                                    # Here is where the D-Wave run happens
                                    if kind == '':
                                        response = samplers['dwave_embed'].sample(
                                            bqm, num_reads=samples, return_embedding=True, chain_strength=chain_strength, annealing_time=ann_time)
                                    elif kind == '_fixed' and best_embed:
                                        response = samplers['dwave'].sample(
                                            bqm, num_reads=samples, return_embedding=True, chain_strength=chain_strength, annealing_time=ann_time)
                                    
                                    pickle.dump(response, open(pickle_name, "wb"))

                                temp['embedding' +
                                    kind] = response.info['embedding_context']['embedding']

                                # plot_energies(response, title=pickle_name)

                                if 'chain_break_fraction' in response.record.dtype.names:
                                    temp['chain_breaks' +
                                        kind] = np.mean(response.record.chain_break_fraction)
                                    temp['chain_breaks_err' +
                                        kind] = np.std(response.record.chain_break_fraction)

                                energies = response.data_vectors['energy']
                                occurrences = response.data_vectors['num_occurrences']

                                counts = {}
                                for index, energy in enumerate(energies):
                                    if energy in counts.keys():
                                        counts[energy] += occurrences[index]
                                    else:
                                        counts[energy] = occurrences[index]

                                total_counts = sum(occurrences)
                                temp['min_fraction' + kind] = sum(counts[key] for key in [min(energies)] if key in counts.keys())/total_counts
                                # temp['opt_fraction' + kind] = sum(counts[key] for key in opt_energies[
                                #     name] if key in counts.keys())/total_counts
                                # temp['feas_fraction' + kind] = sum(
                                #     counts[key] for key in feas_energies[name] if key in counts.keys())/total_counts

                                min_matrix[idx_i, idx_j, idx_k] = sum(counts[key] for key in [min(energies)] if key in counts.keys())/total_counts
                                # opt_matrix[idx_i, idx_j, idx_k] = sum(counts[key] for key in opt_energies[
                                #     name] if key in counts.keys())/total_counts
                                # feas_matrix[idx_i, idx_j, idx_k] = sum(
                                #     counts[key] for key in feas_energies[name] if key in counts.keys())/total_counts

                                idx_k += 1

                            results = results.append(temp, ignore_index=True)

                            results.to_csv(file_name + ".csv")

                            idx_j += 1

                        if draw_figures:
                            plt.figure()

                            plt.errorbar(results.chain_strength, results.chain_breaks,
                                        yerr=results.chain_breaks_err, fmt='o', capsize=3)
                            plt.errorbar(results.chain_strength, results.chain_breaks_fixed,
                                        yerr=results.chain_breaks_err_fixed, fmt='o', capsize=3)
                            plt.legend(
                                ['Random Embedding', 'Best Embedding'])
                            plt.xscale('log')
                            plt.ylabel('Chain break fraction')
                            plt.xlabel(
                                'Chain strength (factor of maximum coefficient in Q)')
                            plt.title(
                                'Chain break fraction vs. chain strength (t_ann = ' + str(ann_time) + ')')

                            plt.figure()

                            plt.plot(results.chain_strength,
                                    results.min_fraction, 'o', linestyle='--')
                            # plt.plot(results.chain_strength,
                            #         results.opt_fraction, 'o', linestyle='--')
                            # plt.plot(results.chain_strength,
                            #         results.feas_fraction, 's', linestyle='--')

                            plt.plot(results.chain_strength,
                                     results.min_fraction_fixed, 'o', linestyle='-')
                            # plt.plot(results.chain_strength,
                            #         results.opt_fraction_fixed, 'o', linestyle='-')
                            # plt.plot(results.chain_strength,
                            #         results.feas_fraction_fixed, 's', linestyle='-')

                            plt.ylim([0, 1])
                            plt.xscale('log')
                            plt.ylabel('Feasible solutions found')
                            plt.xlabel(
                                'Chain strength (factor of maximum coefficient in Q)')
                            plt.title(
                                'Solutions found fraction vs. chain strength (t_ann = ' + str(ann_time) + ')')
                            plt.legend(['Random Embedding optimal',
                                        'Random Embedding feasible',
                                        'Best Embedding optimal',
                                        'Best Embedding feasible'])


                            for k in range(len(experiments)):
                                fig, ax = plt.subplots()

                                colormesh = ax.imshow(
                                    min_matrix[:, 4:, k], vmin=0, vmax=1
                                )
                                ax.set_title(
                                    'Minimum solutions probability embedding' + experiments[k])
                                plt.xlabel(
                                    'Chain strength (factor of maximum coefficient in Q)')
                                plt.ylabel('Annealing time [microseconds]')
                                fig.colorbar(colormesh, ax=ax)
                                plt.yticks([0, 1, 2])
                                fig.canvas.draw()

                                labels_x = [item.get_text() for item in ax.get_xticklabels()]
                                labels_x = chain_strenghts[3:]
                                labels_y = [item.get_text() for item in ax.get_yticklabels()]
                                labels_y = annealing_time

                                ax.set_xticklabels(labels_x)
                                ax.set_yticklabels(labels_y)

                            # for k in range(len(experiments)):
                            #     fig, ax = plt.subplots()

                            #     colormesh = ax.imshow(
                            #         opt_matrix[:, 4:, k], vmin=0, vmax=1
                            #     )
                            #     ax.set_title(
                            #         'Minimal solutions probability embedding' + experiments[k])
                            #     plt.xlabel(
                            #         'Chain strength (factor of maximum coefficient in Q)')
                            #     plt.ylabel('Annealing time [microseconds]')
                            #     fig.colorbar(colormesh, ax=ax)
                            #     plt.yticks([0, 1, 2])
                            #     fig.canvas.draw()

                            #     labels_x = [item.get_text() for item in ax.get_xticklabels()]
                            #     labels_x = chain_strenghts[3:]
                            #     labels_y = [item.get_text() for item in ax.get_yticklabels()]
                            #     labels_y = annealing_time

                            #     ax.set_xticklabels(labels_x)
                            #     ax.set_yticklabels(labels_y)

                            plt.show()

                        idx_i += 1
                   

                    

    sol_total = pd.DataFrame.from_dict(results)

    sol_total.to_excel(os.path.join(results_path,file_name + ".xlsx"))


# Figure functions
def plot_enumerate(results, title=None):
    samples = [''.join(c for c in str(datum.sample.values()).strip(
        ', ') if c.isdigit()) for datum in results.data(['sample'], sorted_by=None)]

    energies = [datum.energy for datum in results.data(
        ['energy'], sorted_by=None)]

    plt.figure()
    plt.bar(samples, energies)
    plt.xticks(rotation=90)
    plt.xlabel('bitstring for solution')
    plt.ylabel('Energy')
    plt.title(str(title))
    plt.show()


def plot_energies(results, title=None):
    energies = results.data_vectors['energy']
    occurrences = results.data_vectors['num_occurrences']
    counts = Counter(energies)
    total = sum(occurrences)
    counts = {}
    for index, energy in enumerate(energies):
        if energy in counts.keys():
            counts[energy] += occurrences[index]
        else:
            counts[energy] = occurrences[index]
    for key in counts:
        counts[key] /= total
    df = pd.DataFrame.from_dict(counts, orient='index').sort_index()
    df.plot(kind='bar', legend=None)

    plt.xlabel('Objective function')
    plt.ylabel('Probabilities')
    plt.title(str(title))
    # plt.show()


def plot_samples(results, title=None):
    samples = [''.join(c for c in str(datum.sample.values()).strip(
        ', ') if c.isdigit()) for datum in results.data(['sample'], sorted_by=None)]

    counts = Counter(samples)
    total = len(samples)
    for key in counts:
        counts[key] /= total
    df = pd.DataFrame.from_dict(counts, orient='index').sort_index()
    plt.figure()
    df.plot(kind='bar', legend=None)

    plt.xticks(rotation=80)
    plt.xlabel('bitstring for solution')
    plt.ylabel('Probabilities')
    plt.title(str(title))
    plt.show()

if __name__ == "__main__":

    # graph_type = 'erdos'
    # graph_type = 'cycle'
    # graph_type = 'devil'
    graph_type = 'spreadsheet'
    TEST = True
    prob = 0.25  # graph probability
    K = 1
    overwrite_pickles = False
    draw_figures = False
    # annealing_time=[2, 20, 200]  # Microseconds
    # chain_strenghts=[0.1, 0.2, 0.5, 1, 2, 5, 10]
    annealing_time=[20]  # Microseconds
    chain_strenghts=[1]
    
    samples=1000
    
    annealing(instance=graph_type, TEST=TEST, prob=prob,
              K=K, overwrite_pickles=overwrite_pickles, draw_figures=draw_figures,
              annealing_time=annealing_time,  # Microseconds
              chain_strenghts=chain_strenghts,
              samples=samples)
