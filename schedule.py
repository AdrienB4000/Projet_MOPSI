import random as rd
import networkx as nx
import time
import os
from pylab import *
from graph import *
from math import inf
from tools import *
import json


class Schedule:
    """This class will we the class of the schedules which are lists
    of tuples (i,j) in [1,m]x[1,n]"""

    def __init__(self, nb_jobs, nb_machines, executions_times,
                 adjacency_list, engine, copying_list=None, sort_function=None):
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
            nb_jobs, nb_machines, executions_times, adjacency_list, engine)
        self.Cmax = np.max(self.completion_matrix)
        self.completion_time += time.time()-beg_time

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

    def nd_engine(self, nb_jobs, nb_machines, execution_times, adjacency_list):
        last_time = time.time()
        eom = [0] * nb_machines
        eoj = [0] * nb_jobs
        untackled_tasks = self.schedule.copy()
        while untackled_tasks:
            # On parcourt la matrice C
            # Pour toutes les taches non 0 sur C
            # à modifier, ne pas stocker de liste, faire une fonction qui récupère index et max(eom[i],eoj[j])
            argmin_m, argmin_j, t, index = min_time(eom, eoj, untackled_tasks)
            completion_time = t + execution_times[argmin_m, argmin_j]
            self.completion_matrix[argmin_m, argmin_j] = completion_time
            eom[argmin_m] = completion_time
            eoj[argmin_j] = completion_time
            for j in adjacency_list[argmin_j]:
                eoj[j] = max(eoj[j], eoj[argmin_j])
            untackled_tasks.pop(index)
        print("Temps dans le engine ND", time.time()-last_time)

        # Version qui fonctionne mais un peu plus coûteuse :(
        # untackled_tasks = self.schedule.copy()
        # times = np.zeros((nb_machines, nb_jobs))
        # while untackled_tasks:
        #     (u, v) = arg_min_prio(times, untackled_tasks)  # (8 s)
        #     t = times[u, v]
        #     completion_time = t + execution_times[u, v]
        #     self.completion_matrix[u, v] = completion_time
        #     for j in range(nb_jobs):
        #         times[u, j] = max(
        #             completion_time, times[u, j])
        #     for i in range(nb_machines):  # (12 s)
        #         times[i, v] = max(
        #             completion_time, times[i, v])
        #         for j in adjacency_list[v]:
        #             times[i, j] = max(
        #                 times[i, j], completion_time)

        #     times[u, v] = np.inf
        #     untackled_tasks.remove((u, v))

    def nd_engine_2(self, nb_jobs, nb_machines, execution_times, adjacency_list):
        last_time = time.time()
        eom = [0] * nb_machines
        eoj = [0] * nb_jobs
        untackled_tasks = self.schedule.copy()
        while untackled_tasks:
            # On parcourt la matrice C
            # Pour toutes les taches non 0 sur C
            # à modifier, ne pas stocker de liste, faire une fonction qui récupère index et max(eom[i],eoj[j])
            argmin_m, argmin_j, t, index = min_time(eom, eoj, untackled_tasks)
            completion_time = t + execution_times[argmin_m, argmin_j]
            self.completion_matrix[argmin_m, argmin_j] = completion_time
            eom[argmin_m] = completion_time
            eoj[argmin_j] = completion_time
            for j in adjacency_list[argmin_j]:
                eoj[j] = max(eoj[j], eoj[argmin_j])
            untackled_tasks.pop(index)
        print("Temps dans le engine ND2", time.time()-last_time)

    def giffler_engine(self, nb_jobs, nb_machines, execution_times, adjacency_list):
        # a modifier de la même façon
        eom = [0] * nb_machines
        eoj = [0] * nb_jobs
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

    def fifo_engine(self, nb_jobs, nb_machines, execution_times, adjacency_list):
        mg = [[(0, inf)] for _ in range(nb_machines)]
        jg = [[(0, inf)] for _ in range(nb_machines)]
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

    def completion_matrix_computation(self, nb_jobs, nb_machines, execution_times, adjacency_list, engine):
        """calculates the completion matrix of a schedule with the given engine"""
        if engine == "ND":
            self.nd_engine(nb_jobs, nb_machines,
                           execution_times, adjacency_list)
        if engine == "ND2":
            self.nd_engine_2(nb_jobs, nb_machines, execution_times, adjacency_list)
        if engine == "FIFO":
            self.fifo_engine(nb_jobs, nb_machines,
                             execution_times, adjacency_list)
        if engine == "GIFFLER":
            self.giffler_engine(nb_jobs, nb_machines,
                                execution_times, adjacency_list)

    def crossover_lox(self, nb_jobs, nb_machines, execution_times, adjacency_list, engine, second_parent):
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
                         adjacency_list, engine, copying_list=child_schedule)
        return child

    def crossover_ox(self, nb_jobs, nb_machines, execution_times, adjacency_list, engine, second_parent):
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
                            adjacency_list, engine, copying_list=child_schedule)
        return children

    def crossover_x1(self, nb_jobs, nb_machines, execution_times, adjacency_list, engine, second_parent):
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
                            adjacency_list, engine, copying_list=child_schedule)
        return children

    def move(self, nb_jobs, nb_machines, execution_times, adjacency_list, engine):
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
                           adjacency_list, engine, copying_list=mutated_schedule)
        return mutated

    def swap(self, nb_jobs, nb_machines, execution_times, adjacency_list, engine):
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
                           adjacency_list, engine, copying_list=mutated_schedule)
        return mutated


if __name__ == "__main__":
    instance_path = os.path.join(
        "instances", "10_10", "MD", "7", "10OSC10_MD_7_6.json")
    with open(instance_path) as instance:
        parameters = json.load(instance)
    nb_jobs = parameters["nb_job"]
    nb_machines = parameters["nb_machine"]
    execution_times = np.array(parameters["processing_times"])
    conflict_graph = nx.from_numpy_matrix(
        np.array(parameters["graph"]["adjacency_matrix"]))
    adjacency_list = [list(conflict_graph.adj[i]) for i in range(nb_jobs)]
    s = Schedule(nb_jobs, nb_machines, execution_times, adjacency_list, "ND")
    s2 = Schedule(nb_jobs, nb_machines, execution_times, adjacency_list, "ND2", copying_list=s.schedule)
