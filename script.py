import random as rd
import json
import networkx as nx
import time
import os
from networkx.algorithms import approximation
from pylab import *
from graph import *
from math import inf
from create_taillard_instance import create_taillard_instance
from tools import *
from schedule import Schedule
from population import Population


def principal_loop(instance_file, parameters, engine='ND'):
    """runs the principal loop of our algorithm"""
# Initialize parameters of the instance
    function_times = {}
    last_time = time.time()
    mutation_probability = parameters["mutation_probability"]
    nb_schedule = parameters["nb_schedule"]
    insert_sorted = parameters["insert_sorted"]
    proba_first_parent = parameters["proba_first_parent"]

    with open(instance_file) as inst:
        instance = json.load(inst)

    nb_jobs = instance["nb_job"]
    nb_machines = instance["nb_machine"]
    execution_times = np.array(instance["processing_times"])

    conflict_graph = nx.from_numpy_matrix(
        np.array(instance["graph"]["adjacency_matrix"]))
    density = instance["graph"]["edge_probability"]
    adjacency_list = [list(conflict_graph.adj[i]) for i in range(nb_jobs)]
    function_times['instance'] = time.time()-last_time
    last_time = time.time()

# Compute the lower bound
    job_times = execution_times.sum(axis=0)
    clique_max = max_weight_clique(conflict_graph, list(job_times))
    lower_bound = lower_bound_calculus(execution_times, clique_max)
    function_times['lower_bound'] = time.time() - last_time
    last_time = time.time()

# Initialize population

    initial_population = Population(nb_jobs, nb_machines, execution_times, conflict_graph, adjacency_list,
                                    lower_bound, engine, nb_schedule, insert_sorted)
    function_times['init_population'] = time.time() - last_time
    last_time = time.time()
    nb_schedule = len(initial_population.population)
    iteration_number = 10*(1-density)*nb_schedule*max(nb_machines, nb_jobs)
    max_constant_iterations = iteration_number/10
    iteration = 0
    compteur = 0
    Cmax = initial_population.population[-1]
    while (iteration < iteration_number and
           initial_population.population[nb_schedule-1].Cmax > lower_bound and
           compteur < max_constant_iterations):
        iteration += 1
        index_p1 = fitness_rank_distribution(nb_schedule)
        index_p2 = uniform_distribution(nb_schedule-1)
        if index_p2 >= index_p1:
            # On décale pour atteindre tout les éléments mais pas index_p1
            index_p2 += 1
        first_parent = initial_population.population[index_p1]
        second_parent = initial_population.population[index_p2]
        proba = rd.random()
        if proba <= proba_first_parent:
            child = first_parent.crossover_lox(
                nb_jobs, nb_machines, execution_times, adjacency_list, engine, second_parent)
            initial_population.completion_time += child.completion_time
        else:
            child = second_parent.crossover_lox(
                nb_jobs, nb_machines, execution_times, adjacency_list, engine, first_parent)
            initial_population.completion_time += child.completion_time
        proba = rd.random()
        if proba < mutation_probability:
            mutated_child = child.move(
                nb_jobs, nb_machines, execution_times, adjacency_list, engine)
            initial_population.completion_time += mutated_child.completion_time
            if mutated_child.Cmax not in initial_population.Used:
                child = mutated_child
        rank_to_replace = uniform_distribution(nb_schedule//2)
        if child.Cmax not in initial_population.Used:
            initial_population.Used.pop(
                initial_population.population[rank_to_replace].Cmax)
            initial_population.Used[child.Cmax] = 1
            initial_population.population.pop(rank_to_replace)
            replacement_index = insert_sorted_list(
                child, initial_population.population, lambda sch: -sch.Cmax)
            if replacement_index > 3*nb_schedule/4:
                compteur = 0
            else:
                compteur += 1
        else:
            compteur += 1
    function_times['principal_loop'] = time.time() - last_time
    print('\n Compteur : ', compteur, " Iteration : ",
          iteration, " Nombre d'itération maximal : ", iteration_number)
    print(function_times)
    print("Temps de calcul total de C : ", initial_population.completion_time)
    return (initial_population, lower_bound, function_times)


# test pour une instance aléatoire
if __name__ == "__main__":
    with open("parameters.json") as param:
        parameters = json.load(param)
    instance_path = os.path.join(
        "instances", "10_10", "MD", "7", "10OSC10_MD_7_6.json")
    (final_pop, lower_bound, function_times) = principal_loop(
        instance_path, parameters, "ND")
    optimal_schedule = final_pop.population[len(final_pop.population) - 1]
    print("Dont le temps d'éxecution vaut : " + str(optimal_schedule.Cmax))
    print("Avec une borne inférieure de : " + str(lower_bound))
    with open(instance_path) as inst:
        instance = json.load(inst)
    conflict_graph = nx.from_numpy_matrix(
        np.array(instance["graph"]["adjacency_matrix"]))
    execution_times = np.array(instance["processing_times"])
    optimal_schedule.visualize(execution_times, conflict_graph, lower_bound)
