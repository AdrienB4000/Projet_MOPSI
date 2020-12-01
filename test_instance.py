from script import *
import random as rd
import json
import numpy as np
import networkx as nx
from networkx.algorithms import approximation
from pylab import *
from graph import max_clique

# Initialization of the instance read in the json file

with open('instance.json') as instance:
    # convert the json file into a dictionnary
    parameters = json.load(instance)

NbSchedules = parameters["nb_schedule"]
NbJobs = parameters["nb_job"]
NbMachines = parameters["nb_machine"]
ExecutionTimes = np.array(parameters["processing_times"])
MutationProbability = parameters["mutation_probability"]

if parameters["Graph"]["rand"]:
    ConflictGraph = nx.erdos_renyi_graph(
        NbJobs, parameters["Graph"]["edge_probability"])
else:
    Adj_matrix = np.array(parameters["Graph"]["Adj_matrix"])
    ConflictGraph = nx.from_numpy_matrix(Adj_matrix)

# DEFINITION OF SOME USEFUL TOOLS
clique_max = list(nx.algorithms.approximation.max_clique(ConflictGraph))
lower_bound = lower_bound_calculus(ExecutionTimes, clique_max)
nb_test = 5
Cmax = []
LB_deviation = []
nb_reach_LB = 0

for i in range(nb_test):
    final_pop, lower_bound = principal_loop(NbJobs, NbMachines, ExecutionTimes,
                                            MutationProbability, NbSchedules, ConflictGraph)
    optimal_schedule = final_pop.population[len(final_pop.population) - 1]
    clique_max = list(nx.algorithms.approximation.max_clique(ConflictGraph))
    clique_max2 = max_clique(ConflictGraph)
    if len(clique_max2) > len(clique_max):
        clique_max = clique_max2
    print("La valeur optimale trouvée est l'emploi du temps :")
    print(optimal_schedule)
    print("Dont le temps d'éxecution vaut : " + str(optimal_schedule.Cmax))
    print("Avec une borne inférieure de : " + str(lower_bound))
    print(clique_max)
    Cmax.append(optimal_schedule.Cmax)
    print("Dont la déviation de LB vaut : " +
          str((optimal_schedule.Cmax-lower_bound)/lower_bound))
    LB_deviation.append((optimal_schedule.Cmax-lower_bound)/lower_bound)
    if optimal_schedule.Cmax == lower_bound:
        nb_reach_LB += 1

result = {"Cmax": np.mean(Cmax), "LB_deviation": np.mean(
    LB_deviation), "reach_LB": nb_reach_LB}
print("les résulats : ", result)
with open('result.json', 'w') as outfile:
    json.dump(result, outfile)
