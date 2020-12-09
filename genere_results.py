from script import *
import random as rd
import json
import numpy as np
import networkx as nx
from networkx.algorithms import approximation
from pylab import *
from graph import max_clique
from create_taillard_instance import *
import time


instance_size = [(4, 4), (5, 5), (7, 7), (10, 10), (15, 15), (20, 20)]
densities = ['LD', 'MD', 'HD']
nb_instance = 10
results = []
for (nb_machine, nb_job) in instance_size:
    for density in densities:
        results.append(
            {"instance_size": (nb_machine, nb_job), "density": density})
        lb_deviation = []
        lb_hit = 0
        execution_time = []
        for num_instance in range(nb_instance):
            create_taillard_instance(
                nb_machine, nb_job, num_instance+1, density)
            execution_time.append(time.time())
            final_pop, lower_bound = principal_loop('taillard_instance.json')
            execution_time[num_instance] = time.time() - \
                execution_time[num_instance]
            optimal_schedule = final_pop.population[len(
                final_pop.population) - 1]
            lb_deviation.append(
                (optimal_schedule.Cmax-lower_bound)/lower_bound)
            if optimal_schedule.Cmax == lower_bound:
                lb_hit += 1
            print("La valeur optimale trouvée est l'emploi du temps : ")
            print(optimal_schedule)
            print("Dont le cmax vaut : " +
                  str(optimal_schedule.Cmax))
            print("Avec une borne inférieure de : " + str(lower_bound))
            print("Dont la déviation de LB vaut : " +
                  str((optimal_schedule.Cmax-lower_bound)/lower_bound))
            print("en un temps d'exécution de " +
                  str(execution_time[num_instance]))
            if num_instance == nb_instance-1:
                results[-1]["lb_deviation"] = np.mean(lb_deviation)
                results[-1]["lb_hit"] = lb_hit
                results[-1]["execution_time"] = np.mean(
                    execution_time)


print(results)
