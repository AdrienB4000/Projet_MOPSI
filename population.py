import random as rd
import networkx as nx
import time
from pylab import *
from graph import *
from math import inf
from tools import *
from schedule import Schedule


class Population:

    def __init__(self, nb_jobs, nb_machines, execution_times, conflict_graph, adjacency_list, lower_bound, engine, nb_schedule=30, insert_sorted=False):
        """initiates a population of nb_schedule elements (if possible) with sorted schedules if insert_sorted"""
        self.population = []
        k = 0
        self.Used = {}
        self.completion_time = 0
        sort_functions = [lambda task: execution_times[task[0], task[1]],
                          lambda task: -execution_times[task[0], task[1]],
                          lambda task: conflict_graph.degree(task[1]),
                          lambda task: -conflict_graph.degree(task[1]),
                          lambda task: execution_times[task[0], task[1]
                                                       ]/(nb_jobs-conflict_graph.degree(task[1])),
                          lambda task: -execution_times[task[0], task[1]]/(nb_jobs-conflict_graph.degree(task[1]))]
        if insert_sorted:
            for function in sort_functions:
                s = Schedule(nb_jobs, nb_machines, execution_times,
                             adjacency_list, engine, sort_function=function)
                if s.Cmax not in self.Used.keys():
                    k += 1
                    self.Used[s.Cmax] = 1
                    self.population.append(s)
                else:
                    self.Used[s.Cmax] += 1
                if lower_bound in self.Used.keys():
                    self.population.sort(key=lambda sch: -sch.Cmax)
                    return
        n_tries = 0
        n_total = 0
        while k != nb_schedule:
            if n_tries == 50:
                k += 1
                n_tries = 0
            s = Schedule(nb_jobs, nb_machines,
                         execution_times, adjacency_list, engine)
            n_total += 1
            n_tries += 1
            if n_total > 1600:
                print(
                    "Alerte générale : boucle infinie dans la création de population, alors qu'on est à k= ", k)
                print("On a d'ailleurs n_tries =", n_tries)
            if s.Cmax not in self.Used.keys():
                k += 1
                n_tries = 0
                self.population.append(s)
                self.Used[s.Cmax] = 1
            else:
                self.Used[s.Cmax] += 1
            if lower_bound in self.Used.keys():
                self.population.sort(key=lambda sch: -sch.Cmax)
                return
        # all_cmax = [s.Cmax for s in self.population]
        # print(all_cmax)
        # print(self.Used)
        #print(" nb essai pop initiale : ", n_total)
        self.population.sort(key=lambda sch: -sch.Cmax)

    def __str__(self):
        return "[" + ", ".join([str(sch) for sch in self.population]) + "]"

    def __repr__(self):
        str_to_return = "Le meilleur emploi du temps est : \n"
        str_to_return = str_to_return + \
            str(self.population[len(self.population)-1])+"\n"
        str_to_return = str_to_return+"Le Cmax de cet emploi du temps est "
        str_to_return = str_to_return + \
            str((self.population[len(self.population)-1]).Cmax) + "\n"
        return str_to_return
