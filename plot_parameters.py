from script import principal_loop
import numpy as np
from graph import max_clique
from create_taillard_instance import create_taillard_instance
import time
import csv
import os
import json
import matplotlib.pyplot as plt


def merge_two_dicts(d1, d2):
    """adds d2 to d1"""
    d1_keys = d1.keys()
    d2_keys = d2.keys()
    for key in d2_keys:
        if key in d1_keys:
            d1[key] += d2[key]
        else:
            d1[key] = d2[key]


def plot_parameters(parameters_file, col, ax1, ax2):
    # , (15, 15)]  # , (20, 20)]
    # , (15, 15)]  # , (20, 20)]
    instance_size = [(4, 4), (5, 5), (7, 7), (10, 10)]
    densities = ['LD', 'MD', 'HD']
    nb_instance = 5
    nb_graph = 5
    engine = 'ND'
    results = []
    times = []
    this_dir = os.path.dirname(os.path.abspath(__file__))
    with open(parameters_file) as param:
        parameters = json.load(param)
    mutation_probability = parameters["mutation_probability"]
    for (nb_machine, nb_job) in instance_size:
        lb_deviation = []
        execution_time = []
        for density in densities:
            for num_instance in range(nb_instance):
                for num_graph in range(nb_graph):
                    execution_time.append(time.time())
                    instance_path = os.path.join(this_dir, "instances", f"{nb_machine}_{nb_job}", f"{density}",
                                                 f"{num_instance+1}", f"{nb_machine}OSC{nb_job}_{density}_{num_instance+1}_{num_graph+1}.json")
                    final_pop, lower_bound, function_times = principal_loop(
                        instance_path, parameters, engine)
                    execution_time[-1] = time.time() - \
                        execution_time[-1]
                    optimal_schedule = final_pop.population[len(
                        final_pop.population) - 1]
                    lb_deviation.append(
                        (optimal_schedule.Cmax-lower_bound)/lower_bound)
        results.append(np.mean(lb_deviation))
        times.append(np.mean(execution_time))
    ax1.scatter([i[0] for i in instance_size], results, color=col)
    if parameters["insert_sorted"]:
        ax2.scatter([i[0] for i in instance_size], times,
                    color=col, label="inserted sorted")
    else:
        ax2.scatter([i[0] for i in instance_size], times,
                    color=col, label="no inserted sorted")


if __name__ == "__main__":
    ax1 = plt.subplot(121)
    ax1.set_title("La déviation à la borne inférieure")
    ax2 = plt.subplot(122)
    ax2.set_title("Temps d'éxecution")
    couleurs = ['b', 'g', 'r', 'y', 'b', 'c', 'm']
    for i in range(1, 3):
        print("On passe au traitement du fichier ", i, "\n \n")
        parameters_file = 'parameters'+str(i)+'.json'
        plot_parameters(parameters_file, couleurs[i], ax1, ax2)
    plt.legend()
    plt.show()
