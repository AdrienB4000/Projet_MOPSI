import json
import numpy as np
import networkx as nx


def priority_sort(to_sort, order_list):
    """Return a sorted to_sort, with elements sorted as in order iterable"""
    priority = {to_sort[index]: order_list[index] for index in range(len(order_list))}
    return sorted(to_sort, key=lambda e: priority[e])


instance_file = open('taillardInstances/tai4_4_1.txt', 'r')  # the instance we want to use
lines = instance_file.readlines()
instance_file.close()

# completing the parameters given by the text file
nb_jobs, nb_machines = [int(x) for x in (lines[1].split())[:2]]
int_lines = []
execution_times = np.zeros((nb_machines, nb_jobs), dtype=int)
for i in range(nb_jobs):
    times_line = lines[3+i]
    order_line = lines[3+nb_jobs+1+i]
    job_times = [int(x) for x in times_line.split()]  # convert string into list of int
    print(job_times)
    order = [int(x) for x in order_line.split()]
    print(order)
    execution_times[i] = np.array(priority_sort(job_times, order))
    print(execution_times[i])

execution_times = (np.transpose(execution_times)).tolist()

# completing the remaining parameters
with open('parameters.json') as parameters:
    param = json.load(parameters)

edge_probability = [0.2, 0.5, 0.8]
graph_names = ["LD", "MD", "HD"]
for i in range(len(graph_names)):
    # the complete instance we store in a json file
    adjacency_matrix = nx.convert_matrix.to_numpy_array(nx.erdos_renyi_graph(nb_jobs, edge_probability[i]), dtype=int)
    instance = {
        "nb_schedule": param["nb_schedule"],
        "nb_machine": nb_machines,
        "nb_job": nb_jobs,
        "processing_times": execution_times,
        "mutation_probability": param["mutation_probability"],
        "Graph": {"rand": True, "edge_probability": edge_probability[i], "Adj_matrix": adjacency_matrix.tolist()}
    }
    with open('taillard_instance_'+graph_names[i]+'.json', 'w') as outfile:
        json.dump(instance, outfile)
