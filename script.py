import random as rd
import json
import numpy as np
import networkx as nx
from pylab import *

# A faire : comparaisons tableaux/listes, afficher LB et matrice de completion
# Créer nos propres instances, voir pour celles de Taillard, interface graphique??
# Optimiser le code : une seule clique max, stocker uniquement C, insertion/suppression plus efficaces?


# Initialization of the instance read in the json file

with open('instance.json') as instance:
    parameters = json.load(instance)  # convert the json file into a dictionnary

NbSchedules = parameters["nb_schedule"]
NbJobs = parameters["nb_job"]
NbMachines = parameters["nb_machine"]
ExecutionTimes = np.array(parameters["processing_times"])
MutationProbability = parameters["mutation_probability"]
if parameters["Graph"]["rand"]:
    ConflictGraph = nx.erdos_renyi_graph(NbJobs, parameters["Graph"]["edge_probability"])
else:
    Adj_matrix = np.array(parameters["Graph"]["Adj_matrix"])
    ConflictGraph = nx.from_numpy_matrix(Adj_matrix)

# DEFINITION OF SOME USEFUL TOOLS


def fitness_rank_distribution(number):
    """simulates a random variable on [0,number-1] whose
    probability is P(k)=2*(k+1)/(number*(number+1))"""
    value = rd.random()
    proba_sum = 0
    for k in range(1, number+1):
        proba_sum += 2*k/(number*(number+1))
        if value < proba_sum:
            return k-1


def uniform_distribution(number):
    """simulates a uniform distribution on [0,number-1]"""
    value = rd.random()
    proba_sum = 0
    for k in range(number):
        proba_sum += 1/number
        if value < proba_sum:
            return k


def tuple_to_int(tup, nb_machines):
    """converts any tuple into a number by a bijection"""
    return tup[1]*nb_machines+tup[0]


def int_to_tuple(integer, nb_machines):
    """the reverse conversion of tuple_to_int"""
    return integer % nb_machines, integer//nb_machines


def insert_sorted_list(element, sorted_list, f):
    """inserts an element in the sorted_list where f is the sorting function"""
    for i in range(len(sorted_list)):
        if f(sorted_list[i]) > f(element):
            sorted_list.insert(i, element)
            return sorted_list
    sorted_list.append(element)
    return sorted_list


def lower_bound_calculus(execution_times, maximal_clique):
    """calculates the lower bound of our problem"""
    lb_machine = max(execution_times.sum(axis=1))
    jobs_times = execution_times.sum(axis=0)
    lb_jobs = max(execution_times.sum(axis=0))
    lb_clique = 0
    for j in maximal_clique:
        lb_clique += jobs_times[j]
    return max(lb_machine, lb_jobs, lb_clique)


# DEFINITION OF THE CENTRAL CLASSES

class Schedule:
    """This class will we the class of the schedules which are lists
    of tuples (i,j) in [1,m]x[1,n]"""
    def __init__(self, nb_jobs, nb_machines, executions_times,
                 graph, copying_list=None, sort_function=None):
        """We create a schedule which is copied if a copying_list is given"""
        """If not this schedule is sorted if a sort_function is given"""
        """else this schedule is randomly sorted"""
        task_list = [(i, j) for i in range(nb_machines) for j in range(nb_jobs)]
        if not (copying_list is None):
            task_list = copying_list
        else:
            if sort_function is None:
                rd.shuffle(task_list)
            else:
                task_list.sort(key=sort_function)
        self.schedule = task_list
        self.completion_matrix = np.zeros((nb_machines, nb_jobs), dtype=int)
        self.completion_matrix_computation(nb_jobs, nb_machines, executions_times, graph)
        self.Cmax = np.max(self.completion_matrix)

    def __str__(self):
        return "[" + ", ".join([str(task) for task in self.schedule]) + "]"
    #à améliorer, afficher le graphe à côté
    def diagram_repr(self, best=True):
        nb_machines=len(self.completion_matrix)
        nb_jobs = len(self.completion_matrix[0])
        xlabel("time")
        ylabel("Machine m")
        axes = gca()
        axes.set_ylim(0, NbMachines)
        axes.set_xlim(0, self.Cmax)
        machine_labels = [f'm={i}' for i in range(1, nb_machines + 1)]
        machine_ticks = [i + 0.5 for i in range(nb_machines)]
        yticks(machine_ticks, machine_labels)
        if best:
            title('The best solution found')
        else:
            title('The real time representation of the schedule')
        for i in range(NbMachines):
            hlines(i, 0, self.Cmax, colors='black')
            for j in range(nb_jobs):
                axes.add_patch(Rectangle((self.completion_matrix[i, j] - ExecutionTimes[i, j], i), ExecutionTimes[i, j], 1,
                                         edgecolor='black', facecolor=f'{j / nb_jobs}', fill=True))
                text(self.completion_matrix[i, j] - ExecutionTimes[i, j] / 2, i + 0.5, f'{j + 1}', color='red',
                     size=5 + 60/nb_machines, ha='center', va='center')
        show()

    def completion_matrix_computation(self, nb_jobs, nb_machines, execution_times, graph):
        """calculates the completion matrix of a schedule"""
        # Ceci est la représentation en liste d'adjacence du graphe de conflit
        eom = [0]*nb_machines
        eoj = [0]*nb_jobs
        adjacency_list = [list(graph.adj[i]) for i in range(nb_jobs)]
        untackled_tasks = self.schedule.copy()
        while untackled_tasks:
            earliest_times = [max(eom[i], eoj[j]) for (i, j) in untackled_tasks]
            # On parcourt la matrice C
            # Pour toutes les taches non 0 sur C
            t = min(earliest_times)
            index = np.argmin(earliest_times)
            (u, v) = untackled_tasks[index]
            completion_time = t+execution_times[u, v]
            self.completion_matrix[u, v] = completion_time
            eom[u] = completion_time
            eoj[v] = completion_time
            for j in adjacency_list[v]:
                eoj[j] = max(eoj[j], eoj[v])
            untackled_tasks.pop(index)

    def crossover_lox(self, nb_jobs, nb_machines, execution_times, graph, second_parent):
        """returns the child which is the crossover between self and second parent by the lox method"""
        nb_tasks = nb_machines*nb_jobs
        taken = [False]*nb_tasks
        p = uniform_distribution(nb_tasks)
        child_schedule = [(0, 0)]*nb_tasks
        if p == 0:
            q = uniform_distribution(nb_tasks-1)
        else:
            q = p+uniform_distribution(nb_tasks-p)
        for i in range(p, q+1):
            child_schedule[i] = self.schedule[i]
            taken[tuple_to_int(self.schedule[i], nb_machines)] = True
        j = 0
        for i in range(nb_tasks):
            if not taken[tuple_to_int(second_parent.schedule[i], nb_machines)]:
                if j == p:
                    j = q+1
                child_schedule[j] = second_parent.schedule[i]
                j += 1
        child = Schedule(nb_jobs, nb_machines, execution_times, graph, copying_list=child_schedule)
        return child

    def crossover_ox(self, nb_jobs, nb_machines, execution_times, graph, second_parent):
        """returns the child which is the crossover between self and second parent by the ox method"""
        nb_tasks = nb_machines*nb_jobs
        taken = [False]*nb_tasks
        p = uniform_distribution(nb_tasks)
        child_schedule = [(0, 0)]*nb_tasks
        if p == 0:
            q = uniform_distribution(nb_tasks-1)
        else:
            q = p+uniform_distribution(nb_tasks-p)
        for i in range(p, q+1):
            child_schedule[i] = self.schedule[i]
            taken[tuple_to_int(self.schedule[i], nb_machines)] = True
        j = (q+1) % nb_tasks
        for i in range(j, j+nb_tasks):
            if not taken[tuple_to_int(second_parent.schedule[i % nb_tasks], nb_machines)]:
                child_schedule[j] = second_parent.schedule[i % nb_tasks]
                j = (j+1) % nb_tasks
        children = Schedule(nb_jobs, nb_machines, execution_times, graph, copying_list=child_schedule)
        return children

    def crossover_x1(self, nb_jobs, nb_machines, execution_times, graph, second_parent):
        """returns the child which is the crossover between self and second parent by the x1 method"""
        nb_tasks = nb_machines*nb_jobs
        taken = [False]*nb_tasks
        q = uniform_distribution(nb_tasks-1)
        child_schedule = [(0, 0)]*nb_tasks
        for i in range(0, q+1):
            child_schedule[i] = self.schedule[i]
            taken[tuple_to_int(self.schedule[i], nb_machines)] = True
        j = q+1
        for i in range(nb_tasks):
            if not taken[tuple_to_int(second_parent.schedule[i], nb_machines)]:
                child_schedule[j] = second_parent.schedule[i]
                j = j+1
        children = Schedule(nb_jobs, nb_machines, execution_times, graph, copying_list=child_schedule)
        return children

    def move(self, nb_jobs, nb_machines, execution_times, graph):
        """returns the mutated schedule obtained by the move method"""
        nb_tasks = nb_machines*nb_jobs
        p = uniform_distribution(nb_tasks)
        q = 1+uniform_distribution(nb_tasks-1)
        a = self.schedule[p]
        mutated_schedule = self.schedule.copy()
        for i in range(p, p+q):
            mutated_schedule[i % nb_tasks] = mutated_schedule[(i+1) % nb_tasks]
        mutated_schedule[(p+q) % nb_tasks] = a
        mutated = Schedule(nb_jobs, nb_machines, execution_times, graph, copying_list=mutated_schedule)
        return mutated

    def swap(self, nb_jobs, nb_machines, execution_times, graph):
        """returns the mutated schedule obtained by the swap method"""
        p = uniform_distribution(nb_machines*nb_jobs)
        q = uniform_distribution(nb_machines*nb_jobs-1)
        mutated_schedule = self.schedule.copy()
        if q >= p:
            q += 1
        a = mutated_schedule[p]
        mutated_schedule[p] = mutated_schedule[q]
        mutated_schedule[q] = a
        mutated = Schedule(nb_jobs, nb_machines, execution_times, graph, copying_list=mutated_schedule)
        return mutated


sort_functions = [lambda task: ExecutionTimes[task[0], task[1]],
                  lambda task: -ExecutionTimes[task[0], task[1]],
                  lambda task: ConflictGraph.degree(task[1]),
                  lambda task: -ConflictGraph.degree(task[1]),
                  lambda task: ExecutionTimes[task[0], task[1]]/(NbJobs-ConflictGraph.degree(task[1])),
                  lambda task: -ExecutionTimes[task[0], task[1]]/(NbJobs-ConflictGraph.degree(task[1]))]


class Population:
    def __init__(self, nb_jobs, nb_machines, execution_times, graph, lower_bound, nb_schedule=30, insert_sorted=False):
        """initiates a population of nb_schedule elements (if possible) with sorted schedules if insert_sorted"""
        self.population = []
        k = 0
        self.Used = [False]*(execution_times.sum()-lower_bound+1)
        if insert_sorted:
            for function in sort_functions:
                s = Schedule(nb_jobs, nb_machines, execution_times, graph, sort_function=function)
                if not self.Used[s.Cmax-lower_bound]:
                    k += 1
                    self.Used[s.Cmax-lower_bound] = True
                    self.population.append(s)
        n_tries = 0
        while k != nb_schedule and n_tries < 50:
            s = Schedule(nb_jobs, nb_machines, execution_times, graph)
            n_tries = 1
            while self.Used[s.Cmax-lower_bound] and n_tries < 50:
                s = Schedule(nb_jobs, nb_machines, execution_times, graph)
                n_tries += 1
            if not self.Used[s.Cmax-lower_bound]:
                k += 1
                self.population.append(s)
                self.Used[s.Cmax-lower_bound] = True
        self.population.sort(key=lambda sch: -sch.Cmax)

    def __str__(self):
        return "[" + ", ".join([str(sch) for sch in self.population]) + "]"

    def __repr__(self):
        str_to_return = "Le meilleur emploi du temps est : \n"
        str_to_return = str_to_return+str(self.population[len(self.population)-1])+"\n"
        str_to_return = str_to_return+"Le Cmax de cet emploi du temps est "
        str_to_return = str_to_return + str((self.population[len(self.population)-1]).Cmax) + "\n"
        return str_to_return


# MAIN ITERATION

def principal_loop(nb_jobs, nb_machines, execution_times,
                   mutation_probability, nb_schedule, conflict_graph, proba_first_parent=0.5):
    """runs the principal loop of our algorithm"""
    cliques = list(nx.algorithms.clique.find_cliques(conflict_graph))
    clique_max = cliques[np.argmax([len(a) for a in cliques])]
    lower_bound = lower_bound_calculus(execution_times, clique_max)
    initial_population = Population(nb_jobs, nb_machines, execution_times, conflict_graph,
                                    lower_bound, nb_schedule, insert_sorted=True)
    nb_schedule = len(initial_population.population)
    iteration_number = 60*nb_schedule*max(nb_machines, nb_jobs)
    iteration = 0
    while iteration < iteration_number and initial_population.population[nb_schedule-1].Cmax > lower_bound:
        print(iteration)
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
            child = first_parent.crossover_lox(nb_jobs, nb_machines, execution_times, conflict_graph, second_parent)
        else:
            child = second_parent.crossover_lox(nb_jobs, nb_machines, execution_times, conflict_graph, first_parent)
        proba = rd.random()
        if proba < mutation_probability:
            mutated_child = child.move(nb_jobs, nb_machines, execution_times, conflict_graph)
            if not initial_population.Used[mutated_child.Cmax-lower_bound]:
                child = mutated_child
        rank_to_replace = uniform_distribution(nb_schedule//2)
        if not initial_population.Used[child.Cmax-lower_bound]:
            initial_population.Used[initial_population.population[rank_to_replace].Cmax-lower_bound] = False
            initial_population.Used[child.Cmax-lower_bound] = True
            initial_population.population.pop(rank_to_replace)
            insert_sorted_list(child, initial_population.population, lambda sch: -sch.Cmax)
    return initial_population




if __name__ == "__main__":
    final_pop = principal_loop(NbJobs, NbMachines, ExecutionTimes,
                               MutationProbability, NbSchedules, ConflictGraph)
    optimal_schedule = final_pop.population[len(final_pop.population) - 1]
    optimal_schedule.diagram_repr() #displays the graph in a second window
    print("La valeur optimale trouvée est l'emploi du temps :")
    print(optimal_schedule)
    print("Dont le temps d'éxecution vaut : " + str(optimal_schedule.Cmax))
