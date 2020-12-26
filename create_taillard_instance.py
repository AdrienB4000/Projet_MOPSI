import json
import os
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph


# nb_jobs, nb_machines, string=LOW,MEDIUM,HARD

def priority_sort(to_sort, order_list):
    """Return a sorted to_sort, with elements sorted as in order iterable"""
    priority = {to_sort[index]: order_list[index]
                for index in range(len(order_list))}
    return sorted(to_sort, key=lambda e: priority[e])


def create_taillard_instance(NbMachines=4, NbJobs=4, density="LD", instance_num=1, num_graph=1, insert_sorted=True):
    file_name = 'taillardInstances/tai' + \
        f'{NbMachines}_{NbJobs}_{instance_num}.txt'
    instance_file = open(file_name, 'r')  # the instance we want to use
    lines = instance_file.readlines()
    instance_file.close()
    # completing the parameters given by the text file
    nb_jobs, nb_machines = [int(x) for x in (lines[1].split())[:2]]

    execution_times = np.zeros((nb_machines, nb_jobs), dtype=int)
    for i in range(nb_jobs):
        times_line = lines[3 + i]
        order_line = lines[3 + nb_jobs + 1 + i]
        # convert string into list of int
        job_times = [int(x) for x in times_line.split()]
        order = [int(x) for x in order_line.split()]
        execution_times[i] = np.array(priority_sort(job_times, order))
    execution_times = (np.transpose(execution_times)).tolist()

    # completing the remaining parameters
    with open('parameters.json') as parameters:
        param = json.load(parameters)
    erdos_proba = {'LD': 0.2, 'MD': 0.5, 'HD': 0.8}
    graph = nx.erdos_renyi_graph(nb_jobs, erdos_proba[density])
    adjacency_matrix = nx.to_numpy_matrix(graph).tolist()

    instance = {
        "nb_schedule": param["nb_schedule"],
        "nb_machine": nb_machines,
        "nb_job": nb_jobs,
        "processing_times": execution_times,
        "mutation_probability": param["mutation_probability"],
        "insert_sorted": insert_sorted,
        "graph": {"rand": True, "edge_probability": erdos_proba[density], "adjacency_matrix": adjacency_matrix}
    }
    with open(os.path.join(this_dir, "instances", f"{nb_machine}_{nb_job}", f"{density}", f"{instance_num}", f"{NbMachines}OSC{NbJobs}_{density}_{instance_num}_{num_graph}.json"), 'w') as outfile:  # 4OSC4_1_HD2
        json.dump(instance, outfile)
    return 0


instance_size = [(4, 4), (5, 5), (7, 7), (10, 10), (15, 15), (20, 20)]
densities = ['LD', 'MD', 'HD']
nb_instance = 10
nb_graph = 10

if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    for (nb_machine, nb_job) in instance_size:
        for density in densities:
            for instance_num in range(nb_instance):
                os.makedirs(os.path.join(this_dir, "instances",
                                         f"{nb_machine}_{nb_job}", f"{density}", f"{instance_num+1}"))
                for num_graph in range(nb_graph):
                    create_taillard_instance(
                        nb_machine, nb_job, density, instance_num+1, num_graph+1, True)
