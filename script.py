import random as rd
import json
import networkx as nx
from networkx.algorithms import approximation
from pylab import *
from graph import *
from math import inf
from create_taillard_instance import create_taillard_instance
import time
import os

# Créer les autres engines, affiner Lower Bound avec clique de poids maximal OK
# Changer la condition d'arrêt en regardant la première moitié
# plutôt que seulement le premier en faisant renvoyer à tri_insertion le rang?
# Passer le nom de l'instance et la définition de paramètres dans la boucle principale OK
# Passage dans des fonctions les créations d'instances aussi OK
# Interface avec l'utilisateur?


# DEFINITION OF SOME USEFUL TOOLS


def fitness_rank_distribution(number):
    """simulates a random variable on [0,number-1] whose
    probability is P(k)=2*(k+1)/(number*(number+1))"""
    value = rd.random()
    proba_sum = 0
    for k in range(1, number+1):
        proba_sum += 2*k/(number*(number+1))
        if value <= proba_sum:
            return k-1
    return number-1


def uniform_distribution(number):
    """simulates a uniform distribution on [0,number-1]"""
    value = rd.random()
    proba_sum = 0
    for k in range(number):
        proba_sum += 1/number
        if value <= proba_sum:
            return k
    return number-1


def tuple_to_int(tup, nb_machines):
    """converts any tuple into a number by a bijection"""
    return tup[1]*nb_machines+tup[0]


def int_to_tuple(integer, nb_machines):
    """the reverse conversion of tuple_to_int"""
    return integer % nb_machines, integer//nb_machines


def insert_sorted_list(element, sorted_list, sort_f):
    """inserts an element in the sorted_list where f is the sorting function"""
    i = 0
    for list_elt in sorted_list:
        if sort_f(list_elt) > sort_f(element):
            sorted_list.insert(i, element)
            return i
        i += 1
    sorted_list.append(element)
    return i


def lower_bound_calculus(execution_times, maximal_clique):
    """calculates the lower bound of our problem"""
    lb_machine = max(execution_times.sum(axis=1))
    jobs_times = execution_times.sum(axis=0)
    lb_jobs = max(execution_times.sum(axis=0))
    lb_clique = 0
    for j in maximal_clique:
        lb_clique += jobs_times[j]
    return max(lb_machine, lb_jobs, lb_clique)


def pop_interval_from_interval_list(interval_list, interval):
    """This function removes an interval from an interval list, knowing that the interval is in conflict
    with interval (each interval is represented by a tuple of the bounds"""
    conflict_interval_index = 0
    length = len(interval_list)
    # We find the interval from interval_list in conflict with the interval
    while conflict_interval_index < length and interval_list[conflict_interval_index][1] <= interval[0]:
        conflict_interval_index += 1
    while conflict_interval_index < length and interval_list[conflict_interval_index][0] <= interval[1]:
        if interval[0] > interval_list[conflict_interval_index][0]:
            if interval[1] >= interval_list[conflict_interval_index][1]:
                interval_list[conflict_interval_index] = interval_list[conflict_interval_index][0], interval[0]
                conflict_interval_index += 1
            else:
                interval_list.insert(conflict_interval_index + 1,
                                     (interval[1], interval_list[conflict_interval_index][1]))
                length += 1
                interval_list[conflict_interval_index] = interval_list[conflict_interval_index][0], interval[0]
                conflict_interval_index += 2
        elif interval[0] < interval_list[conflict_interval_index][0]:
            if interval[1] >= interval_list[conflict_interval_index][1]:
                interval_list.pop(conflict_interval_index)
                length -= 1
            else:
                interval_list[conflict_interval_index] = interval[1], interval_list[conflict_interval_index][1]
                conflict_interval_index += 1
        else:
            if interval[1] >= interval_list[conflict_interval_index][1]:
                interval_list.pop(conflict_interval_index)
                length -= 1
            else:
                interval_list[conflict_interval_index] = interval[1], interval_list[conflict_interval_index][1]
                conflict_interval_index += 1


# DEFINITION OF THE CENTRAL CLASSES

class Schedule:
    """This class will we the class of the schedules which are lists
    of tuples (i,j) in [1,m]x[1,n]"""

    def __init__(self, nb_jobs, nb_machines, executions_times,
                 graph, engine, copying_list=None, sort_function=None):
        """We create a schedule which is copied if a copying_list is given"""
        """If not this schedule is sorted if a sort_function is given"""
        """else this schedule is randomly sorted"""
        self.completion_time = 0
        task_list = [(i, j) for i in range(nb_machines)
                     for j in range(nb_jobs)]
        if not (copying_list is None):
            task_list = copying_list
        else:
            if sort_function is None:
                rd.shuffle(task_list)
            else:
                task_list.sort(key=sort_function)
        self.schedule = task_list
        beg_time = time.time()
        self.completion_matrix = np.zeros((nb_machines, nb_jobs), dtype=int)
        self.completion_matrix_computation(
            nb_jobs, nb_machines, executions_times, graph, engine)
        self.Cmax = np.max(self.completion_matrix)
        self.completion_time = time.time()-beg_time

    def __str__(self):
        return "[" + ", ".join([str(task) for task in self.schedule]) + "]"
    # à améliorer, afficher le graphe à côté

    def plot_graph(self, conflict_graph):
        nb_jobs = len(self.completion_matrix[0])
        figure(2)
        title("Conflict graph for jobs")
        labels = {}
        for i in range(nb_jobs):
            labels[i] = i+1
        nx.draw_networkx(conflict_graph, labels=labels)

    def visuale_repr(self, execution_times, lower_bound, best=True):
        figure(1)
        nb_machines = len(self.completion_matrix)
        nb_jobs = len(self.completion_matrix[0])
        xlabel("time")
        ylabel("Machine m")
        axvline(x=lower_bound, color='blue')
        text(lower_bound, nb_machines+0.5, 'Lower Bound', size=5 +
             60/nb_machines, color='blue', ha='center', va='center')
        axes = gca()
        axes.set_ylim(0, nb_machines)
        axes.set_xlim(0, self.Cmax)
        machine_labels = [f'$m_{i}$' for i in range(1, nb_machines + 1)]
        machine_ticks = [i + 0.5 for i in range(nb_machines)]
        yticks(machine_ticks, machine_labels)
        if best:
            title('The best solution found')
        else:
            title('The real time representation of the schedule')
        for i in range(nb_machines):
            hlines(i, 0, self.Cmax, colors='black')
            for j in range(nb_jobs):
                axes.add_patch(Rectangle((self.completion_matrix[i, j] - execution_times[i, j], i), execution_times[i, j], 1,
                                         edgecolor='black', facecolor=f'{j / nb_jobs}', fill=True))
                text(self.completion_matrix[i, j] - execution_times[i, j] / 2, i + 0.5, f'{j + 1}', color='red',
                     size=5 + 60/nb_machines, ha='center', va='center')

    def visualize(self, execution_times, conflict_graph, lower_bound):
        self.visuale_repr(execution_times, lower_bound)
        self.plot_graph(conflict_graph)
        show()

    def nd_engine(self, nb_jobs, nb_machines, execution_times, graph):
        eom = [0] * nb_machines
        eoj = [0] * nb_jobs
        adjacency_list = [list(graph.adj[i]) for i in range(nb_jobs)]
        untackled_tasks = self.schedule.copy()
        while untackled_tasks:
            # On parcourt la matrice C
            # Pour toutes les taches non 0 sur C
            earliest_times = [max(eom[i], eoj[j])  # 50 % de l'algorithme
                              for (i, j) in untackled_tasks]
            index = np.argmin(earliest_times)  # 25 % de l'algorithme
            t = earliest_times[index]
            (u, v) = untackled_tasks[index]
            completion_time = t + execution_times[u, v]
            self.completion_matrix[u, v] = completion_time
            eom[u] = completion_time
            eoj[v] = completion_time
            for j in adjacency_list[v]:
                eoj[j] = max(eoj[j], eoj[v])    # 10 % de l'algorithme
            untackled_tasks.pop(index)

    def giffler_engine(self, nb_jobs, nb_machines, execution_times, graph):
        eom = [0] * nb_machines
        eoj = [0] * nb_jobs
        adjacency_list = [list(graph.adj[i]) for i in range(nb_jobs)]
        untackled_tasks = self.schedule.copy()
        while untackled_tasks:
            earliest_times = [max(eom[i], eoj[j])+execution_times[i, j]
                              for (i, j) in untackled_tasks]
            # On parcourt la matrice C
            # Pour toutes les taches non 0 sur C
            t = min(earliest_times)
            index = 0
            for i, j in untackled_tasks:
                if eom[i] < t and eoj[j] < t:
                    u, v = i, j
                    break
                index += 1
            t = max(eom[u], eoj[v])
            completion_time = t + execution_times[u, v]
            self.completion_matrix[u, v] = completion_time
            eom[u] = completion_time
            eoj[v] = completion_time
            for j in adjacency_list[v]:
                eoj[j] = max(eoj[j], eoj[v])
            untackled_tasks.pop(index)

    def fifo_engine(self, nb_jobs, nb_machines, execution_times, graph):
        mg = [[(0, inf)] for _ in range(nb_machines)]
        jg = [[(0, inf)] for _ in range(nb_machines)]
        adjacency_list = [list(graph.adj[i]) for i in range(nb_jobs)]
        for (i_machine, i_job) in self.schedule:
            job_length = execution_times[i_machine, i_job]
            i = 0
            j = 0
            while min(mg[i_machine][i][1], jg[i_job][j][1])-max(mg[i_machine][i][0], jg[i_job][j][0]) < job_length:
                if mg[i_machine][i][0] >= jg[i_job][j][0]:
                    if mg[i_machine][i][1] >= jg[i_job][j][1]:
                        j = j+1
                    else:
                        i = i+1
                else:
                    if jg[i_job][j][1] >= mg[i_machine][i][1]:
                        i = i+1
                    else:
                        j = j+1
            t = max(mg[i_machine][i][0], jg[i_job][j][0])
            self.completion_matrix[i_machine, i_job] = t+job_length
            if t-mg[i_machine][i][0] == 0:
                if mg[i_machine][i][1]-(t + job_length) == 0:
                    mg[i_machine].pop(i)
                else:
                    mg[i_machine][i] = (t + job_length, mg[i_machine][i][1])
            elif mg[i_machine][i][1]-(t + job_length) == 0:
                mg[i_machine][i] = (mg[i_machine][i][0], t)
            else:
                mg[i_machine].insert(
                    i + 1, (t + job_length, mg[i_machine][i][1]))
                mg[i_machine][i] = (mg[i_machine][i][0], t)
            if t-jg[i_job][j][0] == 0:
                if jg[i_job][j][1]-(t + job_length) == 0:
                    jg[i_job].pop(j)
                else:
                    jg[i_job][j] = (t + job_length, jg[i_job][j][1])
            elif jg[i_job][j][1]-(t + job_length) == 0:
                jg[i_job][j] = (jg[i_job][j][0], t)
            else:
                jg[i_job].insert(j + 1, (t + job_length, jg[i_job][j][1]))
                jg[i_job][j] = (jg[i_job][j][0], t)
            for vertex in adjacency_list[i_job]:
                pop_interval_from_interval_list(jg[vertex], (t, t+job_length))

    def completion_matrix_computation(self, nb_jobs, nb_machines, execution_times, graph, engine):
        """calculates the completion matrix of a schedule with the given engine"""
        if engine == "ND":
            self.nd_engine(nb_jobs, nb_machines, execution_times, graph)
        if engine == "FIFO":
            self.fifo_engine(nb_jobs, nb_machines, execution_times, graph)
        if engine == "GIFFLER":
            self.giffler_engine(nb_jobs, nb_machines, execution_times, graph)

    def crossover_lox(self, nb_jobs, nb_machines, execution_times, graph, engine, second_parent):
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
        child = Schedule(nb_jobs, nb_machines, execution_times,
                         graph, engine, copying_list=child_schedule)
        return child

    def crossover_ox(self, nb_jobs, nb_machines, execution_times, graph, engine, second_parent):
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
        children = Schedule(nb_jobs, nb_machines, execution_times,
                            graph, engine, copying_list=child_schedule)
        return children

    def crossover_x1(self, nb_jobs, nb_machines, execution_times, graph, engine, second_parent):
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
        children = Schedule(nb_jobs, nb_machines, execution_times,
                            graph, engine, copying_list=child_schedule)
        return children

    def move(self, nb_jobs, nb_machines, execution_times, graph, engine):
        """returns the mutated schedule obtained by the move method"""
        nb_tasks = nb_machines*nb_jobs
        p = uniform_distribution(nb_tasks)
        q = 1+uniform_distribution(nb_tasks-1)
        a = self.schedule[p]
        mutated_schedule = self.schedule.copy()
        for i in range(p, p+q):
            mutated_schedule[i % nb_tasks] = mutated_schedule[(i+1) % nb_tasks]
        mutated_schedule[(p+q) % nb_tasks] = a
        mutated = Schedule(nb_jobs, nb_machines, execution_times,
                           graph, engine, copying_list=mutated_schedule)
        return mutated

    def swap(self, nb_jobs, nb_machines, execution_times, graph, engine):
        """returns the mutated schedule obtained by the swap method"""
        p = uniform_distribution(nb_machines*nb_jobs)
        q = uniform_distribution(nb_machines*nb_jobs-1)
        mutated_schedule = self.schedule.copy()
        if q >= p:
            q += 1
        a = mutated_schedule[p]
        mutated_schedule[p] = mutated_schedule[q]
        mutated_schedule[q] = a
        mutated = Schedule(nb_jobs, nb_machines, execution_times,
                           graph, engine, copying_list=mutated_schedule)
        return mutated


class Population:

    def __init__(self, nb_jobs, nb_machines, execution_times, conflict_graph, lower_bound, engine, nb_schedule=30, insert_sorted=False):
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
                             conflict_graph, engine, sort_function=function)
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
                         execution_times, conflict_graph, engine)
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


# MAIN ITERATION


def principal_loop(instance_file, engine='ND'):
    """runs the principal loop of our algorithm"""
# Initialize parameters of the instance
    function_times = {}
    last_time = time.time()
    with open(instance_file) as instance:
        parameters = json.load(instance)
    nb_schedule = parameters["nb_schedule"]
    nb_jobs = parameters["nb_job"]
    nb_machines = parameters["nb_machine"]
    execution_times = np.array(parameters["processing_times"])
    mutation_probability = parameters["mutation_probability"]
    insert_sorted = parameters["insert_sorted"]
    if "proba_first_parent" in parameters:
        proba_first_parent = parameters["proba_first_parent"]
    else:
        proba_first_parent = 0.5

    conflict_graph = nx.from_numpy_matrix(
        np.array(parameters["graph"]["adjacency_matrix"]))
    function_times['parameters'] = time.time()-last_time
    last_time = time.time()

# Compute the lower bound
    job_times = execution_times.sum(axis=0)
    clique_max = max_weight_clique(conflict_graph, list(job_times))
    lower_bound = lower_bound_calculus(execution_times, clique_max)
    function_times['lower_bound'] = time.time() - last_time
    last_time = time.time()

# Initialize population

    initial_population = Population(nb_jobs, nb_machines, execution_times, conflict_graph,
                                    lower_bound, engine, nb_schedule, insert_sorted)
    function_times['init_population'] = time.time() - last_time
    last_time = time.time()
    nb_schedule = len(initial_population.population)
    iteration_number = 60*nb_schedule*max(nb_machines, nb_jobs)
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
                nb_jobs, nb_machines, execution_times, conflict_graph, engine, second_parent)
            initial_population.completion_time += child.completion_time
        else:
            child = second_parent.crossover_lox(
                nb_jobs, nb_machines, execution_times, conflict_graph, engine, first_parent)
            initial_population.completion_time += child.completion_time
        proba = rd.random()
        if proba < mutation_probability:
            mutated_child = child.move(
                nb_jobs, nb_machines, execution_times, conflict_graph, engine)
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
    print('temps de calcul la matrice de completion : ',
          initial_population.completion_time)
    return (initial_population, lower_bound, function_times)


if __name__ == "__main__":
    instance_path = os.path.join(
        "instances", "10_10", "MD", "7", "10OSC10_MD_7_6.json")
    (final_pop, lower_bound, function_times) = principal_loop(
        instance_path, "ND")
    optimal_schedule = final_pop.population[len(final_pop.population) - 1]
    # print("La valeur optimale trouvée est l'emploi du temps :")
    # print(optimal_schedule)
    print("Dont le temps d'éxecution vaut : " + str(optimal_schedule.Cmax))
    print("Avec une borne inférieure de : " + str(lower_bound))
    with open(instance_path) as instance:
        parameters = json.load(instance)
    conflict_graph = nx.from_numpy_matrix(
        np.array(parameters["graph"]["adjacency_matrix"]))
    execution_times = np.array(parameters["processing_times"])
    optimal_schedule.visualize(execution_times, conflict_graph, lower_bound)
