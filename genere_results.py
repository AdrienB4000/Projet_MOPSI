from script import principal_loop
import numpy as np
from graph import max_clique
from create_taillard_instance import create_taillard_instance
import time
import csv
import os
import json

# Remi :
# bien écrire dans le csv
# tester 10 fois la même instance de taillard pour une taille et densité donnée
# stocker un fichier de résultat par instance en dehors du temps d'exécution

# Adrien :
# terminer tester   giff, fifo
# déterminer les parties qui prennent du temps

# Ensemble:
# voir si on peut run en release au lieu de debug
# refléchir à l'organisation du poster
#
# voir partie 5

# A étudier:
# On ne voit pas comment améliorer la complexité de notre algo qui ne dépend que du calcul de C et donc de l'heuristique
# par conséquent nous ne pouvons pas générer nos résultats en 10 h (il faut donc diminuer voir pour diminuer la quantité d'instances)
# Quel format pour le poster, sur quel outil ?

# Poster : introduction sur pq ce pb se pose (pb d'ordonancement ultra classique dans l'industrie ou ailleurs,
# schéma d'un pb pour que le lecteur comprenne (trouver un ex qui se schématise bien, qu'on utilise tout au long ou pas)
# Spécificité de l'open shop avec les conflits (schéma avec les autre pb d'ordonnancement)
# Présentation des algorithmes génétiques en général ou directement notre algo ?
# Est ce qu'on présente la génération de la population initiale,les mutations,
#  les crossovers, les 3 heuristiques, les conditions d'arrêts ?
# Présentation du rendu graphique
# Amélioration de notre algorithme
# Présentation des résultats (tableau)

# A retenir
# Modifier la prise en compte des paramètres pour les lire 1 fois et pas 1500 fois
# Tenter de modifier la fonction ND engine
# faire des stats sur les heuristiques


def merge_two_dicts(d1, d2):
    """adds d2 to d1"""
    d1_keys = d1.keys()
    d2_keys = d2.keys()
    for key in d2_keys:
        if key in d1_keys:
            d1[key] += d2[key]
        else:
            d1[key] = d2[key]


instance_size = [(10, 10)]  # , (20, 20)]
densities = ['LD', 'MD', 'HD']
nb_instance = 10
nb_graph = 10
engine = 'GIFFLER'
results = []
this_dir = os.path.dirname(os.path.abspath(__file__))
for (nb_machine, nb_job) in instance_size:
    for density in densities:
        d = {}
        results.append(
            {"instance_size": f"({nb_machine},{nb_job})", "density": density})
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
                    instance_path, engine)
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
with open('results'+engine+'test.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, results[0].keys(), delimiter=";")
    w.writeheader()
    w.writerows(results)
