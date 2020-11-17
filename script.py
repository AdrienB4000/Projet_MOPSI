import random as rd
import numpy as np
import networkx as nx
import json

#Initialization of the instance read in the json file

with open('instance.txt') as instance:
    parameters = json.load(instance)  #convert the json file into a dictionnary

NS = parameters["nb_schedule"]
n = parameters["nb_job"]
m = parameters["nb_machine"]
prices = np.array(parameters["processing_times"])
Pm = parameters["mutation_probability"]
if parameters["Graph"]["rand"]:
    G = nx.erdos_renyi_graph(n, parameters["Graph"]["edge_probability"])
else:
    Adj_matrix = np.array(parameters["Graph"]["Adj_matrix"])
    G = nx.from_numpy_matrix(Adj_matrix)

cliques = list(nx.algorithms.clique.find_cliques(G))
clique_max = cliques[np.argmax([len(a) for a in cliques])]
adjacency_list = [list(G.adj[i]) for i in range(n)]


## DEFINITION DE QUELQUES OUTILS UTILES

def fitness_rank_distribution(NS):
    value=rd.random()
    sum=0
    for k in range(1, NS+1):
        sum+=2*k/(NS*(NS+1))
        if value<sum:
            return k-1


def uniform_distribution(NS):
    value=rd.random()
    sum=0
    for k in range(NS):
        sum+=1/NS
        if value<sum:
            return k


def tuple_to_int(tup):
    return tup[1]*m+tup[0]


def int_to_tuple(integer):
    return (integer%m, integer//m)


def insert_sorted_list(elmt, L, f):
    sorted_list=[]
    for i in range(len(L)):
        if f(L[i])>f(elmt):
            L.insert(i,elmt)
            return L
    L.append(elmt)
    return L


def calcul_LB(prices, clique_max):
    LB_machine=max(prices.sum(axis=1))
    jobs_prices=prices.sum(axis=0)
    LB_jobs=max(jobs_prices)
    LB_clique=0
    for j in clique_max:
        LB_clique+=jobs_prices[j]
    return max(LB_machine, LB_jobs, LB_clique)


## DEFINITION DES CLASSES CENTRALES

class Schedule:
    """This class will we the class of the schedules which are lists
    of tuples (i,j) in [1,m]x[1,n]"""
    def __init__(self, n, m, prices, G, copying_list=None, sort_function=None):
        """We create a schedule which is copied if a copying_list is given"""
        """If not this schedule is sorted if a sort_function is given"""
        """else this schedule is randomly sorted"""
        task_list=[(i,j) for i in range(m) for j in range(n)]
        if copying_list!=None:
            task_list=copying_list
        else:
            if sort_function==None:
                rd.shuffle(task_list)
            else:
                task_list.sort(key = sort_function)
        self.schedule=task_list
        self.C = np.zeros((m,n), dtype=int)
        self.C_computation(n, m, prices, G)
        self.Cmax=np.max(self.C)

    def __str__(self):
        return "[" + ", ".join([str(task) for task in self.schedule]) + "]"

    def C_computation(self, n, m, prices, G):
        # Ceci est la représentation en liste d'adjacence du graphe de conflit
        EOM=[0]*m
        EOJ=[0]*n
        adjacency_list=adjacency_list=[list(G.adj[i]) for i in range(n)]
        untackled_tasks=(self.schedule).copy()
        while untackled_tasks!=[] :
            earliest_times=[max(EOM[i], EOJ[j]) for (i,j) in untackled_tasks]
            # On parcourt la matrice C
            # Pour toutes les taches non 0 sur C
            t = min(earliest_times)
            index = np.argmin(earliest_times)
            (u, v) = untackled_tasks[index]
            completion_time = t+prices[u, v]
            self.C[u, v] = completion_time
            EOM[u] = completion_time
            EOJ[v] = completion_time
            for j in adjacency_list[v]:
                EOJ[j]=max(EOJ[j], EOJ[v])
            untackled_tasks.pop(index)

    def crossover_LOX(self, n, m, prices, P2):
        Nb=m*n
        Taken=[False]*Nb
        p=uniform_distribution(Nb)
        schedule_C1=[(0,0)]*Nb
        if p==0:
            q=uniform_distribution(Nb-1)
        else:
            q=p+uniform_distribution(Nb-p)
        for i in range (p, q+1):
            schedule_C1[i]=self.schedule[i]
            Taken[tuple_to_int(self.schedule[i])]=True
        j=0
        for i in range(Nb):
            if not Taken[tuple_to_int(P2.schedule[i])]:
                if j==p:
                    j=q+1
                schedule_C1[j]=P2.schedule[i]
                j+=1
        C1=Schedule(n, m, prices, G, copying_list=schedule_C1)
        return C1


    def crossover_OX(self, n, m, prices, P2):
        Nb=m*n
        Taken=[False]*Nb
        p=uniform_distribution(Nb)
        schedule_C1=[(0,0)]*Nb
        if p==0:
            q=uniform_distribution(Nb-1)
        else:
            q=p+uniform_distribution(Nb-p)
        for i in range (p, q+1):
            schedule_C1[i]=self.schedule[i]
            Taken[tuple_to_int(self.schedule[i])]=True
        j=(q+1)%Nb
        for i in range (j, j+Nb):
            if not Taken[tuple_to_int(P2.schedule[i%Nb])]:
                schedule_C1[j]=P2.schedule[i%Nb]
                j=(j+1)%Nb
        C1=Schedule(n, m, prices, G, copying_list=schedule_C1)
        return C1


    def crossover_X1(self, n, m, prices, P2):
        Nb=m*n
        Taken=[False]*Nb
        q=uniform_distribution(Nb-1)
        schedule_C1=[(0,0)]*Nb
        for i in range (0, q+1):
            schedule_C1[i]=self.schedule[i]
            Taken[tuple_to_int(self.schedule[i])]=True
        j=q+1
        for i in range (Nb):
            if not Taken[tuple_to_int(P2.schedule[i])]:
                schedule_C1[j]=P2.schedule[i]
                j=j+1
        C1=Schedule(n, m, prices, G, copying_list=schedule_C1)
        return C1


    def move(self, n, m, prices):
        p=uniform_distribution(m*n)
        q=1+uniform_distribution(m*n-1)
        Nb=m*n
        a=self.schedule[p]
        mutated_schedule=self.schedule.copy()
        for i in range(p,p+q):
            mutated_schedule[i%Nb]=mutated_schedule[(i+1)%Nb]
        mutated_schedule[(p+q)%Nb]=a
        mutated=Schedule(n,m,prices, G, copying_list=mutated_schedule)
        return mutated


    def swap(self, n, m, prices):
        p=uniform_distribution(m*n)
        q=uniform_distribution(m*n-1)
        mutated_schedule=self.schedule.copy()
        if q>=p:
            q+=1
        a=mutated_schedule[p]
        mutated_schedule[p]=mutated_schedule[q]
        mutated_schedule[q]=a
        mutated=Schedule(n, m, prices, G, copying_list=mutated_schedule)
        return mutated


sort_functions=[lambda task : prices[task[0],task[1]],
                lambda task : -prices[task[0],task[1]],
                lambda task : G.degree(task[1]),
                lambda task : -G.degree(task[1]),
                lambda task : prices[task[0],task[1]]/(n-G.degree(task[1])),
                lambda task : -prices[task[0],task[1]]/(n-G.degree(task[1]))]


class Population:
    def __init__(self, n, m, prices, LB, NS=30, insert_sorted=False):
        self.population=[]
        k=0
        self.Used=[False]*(prices.sum()-LB+1)
        if insert_sorted:
            for function in sort_functions:
                s=Schedule(n, m, prices, G, sort_function=function)
                if not self.Used[s.Cmax-LB]:
                    k+=1
                    self.Used[s.Cmax-LB]=True
                    self.population.append(s)
        NTries=0
        while (k!=NS and NTries<50):
            s=Schedule(n, m, prices, G)
            NTries=1
            while (self.Used[s.Cmax-LB] and NTries<50):
                s=Schedule(n, m, prices, G)
                NTries+=1
            if not self.Used[s.Cmax-LB]:
                k+=1
                self.population.append(s)
                self.Used[s.Cmax-LB]=True
        self.population.sort(key = lambda sch : -sch.Cmax)
    def __str__(self):
        return "[" + ", ".join([str(sch) for sch in self.population]) + "]"
    def __repr__(self):
        repr="Le meilleur emploi du temps est : \n"
        repr=repr+str(self.population[len(self.population)-1])+"\n"
        repr=repr+"Le Cmax de cet emploi du temps est " + str((self.population[len(self.population)-1]).Cmax) +"\n"
        return repr



## ITERATION PRINCIPALE

def boucle_principale(prices,n,m,NS,Pm,G):
    cliques=list(nx.algorithms.clique.find_cliques(G))
    clique_max=cliques[np.argmax([len(a) for a in cliques])]
    LB=calcul_LB(prices, clique_max)
    S=Population(n, m, prices, LB, NS, insert_sorted=True)
    NS=len(S.population)
    NI=60*NS*max(m,n)
    Iter=0
    while(Iter<NI and S.population[NS-1].Cmax>LB):
        Iter+=1
        index_p1=fitness_rank_distribution(NS)
        index_p2=uniform_distribution(NS-1)
        if (index_p1>=index_p2):
            # On décale pour atteindre tout les éléments mais pas index_p1
            index_p2+=1
        P1=S.population[index_p1]
        P2=S.population[index_p2]
        proba=rd.random()
        if proba<=1/2:
            C1=P1.crossover_LOX(n, m, prices, P2)
        else:
            C1=P2.crossover_LOX(n, m, prices, P1)
        proba=rd.random()
        if proba<Pm:
            V=C1.move(n, m, prices)
            if not S.Used[V.Cmax-LB]:
                C1=V
        R=uniform_distribution(NS//2)
        if not S.Used[C1.Cmax-LB]:
            S.Used[S.population[R].Cmax-LB]=False
            S.Used[C1.Cmax-LB]=True
            S.population.pop(R)
            insert_sorted_list(C1, S.population, lambda sch : -sch.Cmax)
    return S

if __name__=="__main__":
    final_pop = boucle_principale(prices, n, m, NS, Pm, G)
    optimal_schedule = final_pop.population[len(final_pop.population)-1]
    print("La valeur optimale trouvée est l'emploi du temps :")
    print(optimal_schedule)
    print("Dont le temps d'éxecution vaut : "+str(optimal_schedule.Cmax))