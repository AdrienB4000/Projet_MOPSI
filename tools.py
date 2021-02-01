import random as rd


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


def arg_min_prio(matrice, schedule):
    val = matrice[schedule[0]]
    mini = schedule[0]
    for (i, j) in schedule:
        if matrice[i, j] < val:
            val = matrice[i, j]
            mini = (i, j)
    return mini


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


def min_time_to_begin(eom, eoj, untackled_tasks):
    """Calculates the indexes i and j which minimizes (max(eom[i], eoj[j]))"""
    """We have to go in the order of untackled_tasks"""
    arg_m, arg_j = untackled_tasks[0]
    index = 0
    min_t = max(eom[arg_m], eoj[arg_j])
    i = 0
    for m, j in untackled_tasks:
        if max(eom[m], eoj[j]) < min_t:
            arg_m = m
            arg_j = j
            min_t = max(eom[m], eoj[j])
            index = i
        i += 1
    return arg_m, arg_j, min_t, index


def min_time_to_finish(eom, eoj, untackled_tasks, execution_times):
    """Calculates the indexes i and j which minimizes (max(eom[i], eoj[j])+execution_times[i,j])"""
    """We have to go in the order of untackled_tasks"""
    arg_m, arg_j = untackled_tasks[0]
    index = 0
    min_t = max(eom[arg_m], eoj[arg_j])+execution_times[arg_m, arg_j]
    i = 0
    for m, j in untackled_tasks:
        if max(eom[m], eoj[j])+execution_times[m, j] < min_t:
            arg_m = m
            arg_j = j
            min_t = max(eom[m], eoj[j])+execution_times[m, j]
            index = i
        i += 1
    return arg_m, arg_j, min_t, index