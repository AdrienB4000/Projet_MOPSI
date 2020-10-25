import random as rd
import numpy as np
#test de git bash
#test 2 de github desktop
NS = 30
m = 3
n = 3
prices = np.random.randint(100, size = (m, n))
LB=max(prices.sum(axis=1))
Pm=0.2

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


class Schedule:
    """This class will we the class of the schedules which are lists
    of tuples (i,j) in [1,m]x[1,n]"""
    def __init__(self):
        """We create a random schedule because we won't use not random ones"""
        rd_list=[i for i in range(m*n)]
        rd.shuffle(rd_list)
        self.schedule=[int_to_tuple(x) for x in rd_list]
        self.C = np.zeros((m,n), dtype=int)
        self.C_computation()
        self.Cmax=np.max(self.C)

    def __str__(self):
        return "[" + ", ".join([str(task) for task in self.schedule]) + "]"

    def C_computation(self):
        EOM=[0]*m
        EOJ=[0]*n
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
            untackled_tasks.pop(index)

    def crossover_LOX(self, P2):
        Nb=m*n
        Taken=[False]*Nb
        p=uniform_distribution(Nb)
        C1=Schedule()
        if p==0:
            q=uniform_distribution(Nb-1)
        else:
            q=p+uniform_distribution(Nb-p)
        for i in range (p, q+1):
            C1.schedule[i]=self.schedule[i]
            Taken[tuple_to_int(self.schedule[i])]=True
        j=0
        for i in range(Nb):
            if not Taken[tuple_to_int(P2.schedule[i])]:
                if j==p:
                    j=q+1
                C1.schedule[j]=P2.schedule[i]
                j+=1
        C1.C_computation()
        C1.Cmax=np.max(C1.C)
        return C1


    def crossover_OX(self, P2):
        Nb=m*n
        Taken=[False]*Nb
        p=uniform_distribution(Nb)
        C1=Schedule()
        if p==0:
            q=uniform_distribution(Nb-1)
        else:
            q=p+uniform_distribution(Nb-p)
        for i in range (p, q+1):
            C1.schedule[i]=self.schedule[i]
            Taken[tuple_to_int(self.schedule[i])]=True
        j=(q+1)%Nb
        for i in range (j, j+Nb):
            if not Taken[tuple_to_int(P2.schedule[i%Nb])]:
                C1.schedule[j]=P2.schedule[i%Nb]
                j=(j+1)%n
        C1.C_computation()
        C1.Cmax=np.max(C1.C)
        return C1


    def crossover_X1(self, P2):
        Nb=m*n
        Taken=[False]*Nb
        q=uniform_distribution(Nb-1)
        C1=Schedule()
        for i in range (0, q+1):
            C1.schedule[i]=self.schedule[i]
            Taken[tuple_to_int(self.schedule[i])]=True
        j=q+1
        for i in range (Nb):
            if not Taken[tuple_to_int(P2.schedule[i])]:
                C1.schedule[j]=P2.schedule[i]
                j=j+1
        C1.C_computation()
        C1.Cmax=np.max(C1.C)
        return C1

    def move(self):
        p=uniform_distribution(m*n)
        q=1+uniform_distribution(m*n-1)
        Nb=m*n
        a=self.schedule[p]
        mutated=Schedule()
        mutated.schedule=self.schedule.copy()
        for i in range(p,p+q):
            mutated.schedule[i%Nb]=mutated.schedule[(i+1)%Nb]
        mutated.schedule[(p+q)%Nb]=a
        return mutated


class Population:
    def __init__(self):
        self.population=[]
        k=0
        Used=[False]*(prices.sum()-LB+1)
        NTries=0
        while (k!=NS and NTries<50):
            s=Schedule()
            NTries=1
            while (Used[s.Cmax-LB] and NTries<50):
                s=Schedule()
                NTries+=1
            if not Used[s.Cmax-LB]:
                k+=1
                self.population.append(s)
                Used[s.Cmax-LB]=True
        self.population.sort(key = lambda sch : sch.Cmax)


S = Population()
NS = len(S.population)
NI = 60*NS*max(m,n)
Iter = 0

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
        C1=P1.crossover_LOX(P2)
    else:
        C1=P2.crossover_LOX(P1)
    R=NS//2
    proba=rd.random()
    if proba<Pm:
        V=C1.move()

