from script import principal_loop
import numpy as np
from graph import max_clique
from create_taillard_instance import create_taillard_instance
import time
import csv
import os
import json


def merge_two_dicts(d1, d2):
    """adds d2 to d1"""
    d1_keys = d1.keys()
    d2_keys = d2.keys()
    for key in d2_keys:
        if key in d1_keys:
            d1[key] += d2[key]
        else:
            d1[key] = d2[key]


#instance_size = [(4, 4), (5, 5), (7, 7), (10, 10), (15, 15)]
instance_size = [(15, 15)]
densities = ['LD', 'MD', 'HD']
nb_instance = 10
nb_graph = 10
engine = 'ND'
results = []
this_dir = os.path.dirname(os.path.abspath(__file__))
with open("parameters.json") as param:
    parameters = json.load(param)

for (nb_machine, nb_job) in instance_size:
    for density in densities:
        d = {}
        results.append(
            {"instance_size": (nb_machine, nb_job), "density": density})
        lb_deviation = []
        lb_hit = 0
        execution_time = []
        cumulated_function_times = {}
        for num_instance in range(nb_instance):
            for num_graph in range(nb_graph):
                instance_result = {}
                instance_path = os.path.join(this_dir, "instances", f"{nb_machine}_{nb_job}", f"{density}",
                                             f"{num_instance+1}", f"{nb_machine}OSC{nb_job}_{density}_{num_instance+1}_{num_graph+1}.json")
                execution_time.append(time.time())
                final_pop, lower_bound, function_times = principal_loop(
                    instance_path, parameters, engine)
                execution_time[-1] = time.time() - \
                    execution_time[-1]
                merge_two_dicts(cumulated_function_times, function_times)
                instance_result["execution_time"] = execution_time[-1]
                optimal_schedule = final_pop.population[len(
                    final_pop.population) - 1]
                lb_deviation.append(
                    (optimal_schedule.Cmax-lower_bound)/lower_bound)
                instance_result["lb_deviation"] = lb_deviation[-1]
                if optimal_schedule.Cmax == lower_bound:
                    instance_result["lb_hit"] = True
                    lb_hit += instance_result["lb_hit"]
                else:
                    instance_result["lb_hit"] = False
                instance_result["function_times"] = function_times
                with open(os.path.join(this_dir, "instances", f"{nb_machine}_{nb_job}", f"{density}", f"{num_instance+1}", f"result_{engine}_{nb_machine}OSC{nb_job}_{density}_{num_instance+1}_{num_graph+1}.txt"), 'w') as outfile:
                    json.dump(instance_result, outfile)
        results[-1]["lb_deviation"] = np.mean(lb_deviation)
        results[-1]["lb_hit"] = lb_hit
        results[-1]["execution_time"] = np.mean(execution_time)
        print("Valeurs moyennes", results[-1])
        cumulated_function_times = {k: v / (nb_graph*nb_instance)
                                    for k, v in cumulated_function_times.items()}
        print("Temps des différentes parties en moyenne :",
              cumulated_function_times)

print(results)
with open('results'+engine+'_15.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, results[0].keys(), delimiter=";")
    w.writeheader()
    w.writerows(results)
