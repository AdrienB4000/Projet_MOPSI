from script import principal_loop
import numpy as np
from graph import max_clique
from create_taillard_instance import create_taillard_instance
import time
import csv

# terminer tester test sur  giff, fifo
# tester 10 fois la même instance de taillard pour une taille et densité donnée
# avoir des instances fixes donc créer le 1500 instances avant (4OSC4_1_HD2)
# déterminer les parties qui prennent du temps
# voir si on peut run en release au lieu de debug
# stocker un fichier de résultat par instance en dehors du temps d'exécution
# refléchir à l'organisation du poster
#
# voir partie 5

instance_size = [(4, 4), (7, 7), (10, 10), (15, 15), (20, 20)]
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
            if num_instance == nb_instance-1:
                results[-1]["lb_deviation"] = np.mean(lb_deviation)
                results[-1]["lb_hit"] = lb_hit
                results[-1]["execution_time"] = np.mean(execution_time)
                print("Valeurs moyennes", results[-1])

print(results)
with open('results.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, results[0].keys())
    w.writeheader()
    w.writerows(results)
