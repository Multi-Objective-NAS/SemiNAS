import functools

OBJECTIVES = ['acc', 'lat']  # Objectives of sorting.
OPT = [1, -1]  # 1 : Minimizing is objective ; -1 : Maximizing is objective.
INFINITE = 1000000000


'''
Returns which one Pareto dominated another.
'''
def dominate_operator(elem1, elem2):
    dominate_count = [0, 0]
    # Counts number of winning in each objectives.

    for obj, criteria in zip(OBJECTIVES, OPT):
        if elem1[obj] == elem2[obj]:
            continue
        elif ((elem1[obj] - elem2[obj]) * criteria) > 0.0:
            dominate_count[0] += 1
        else:
            dominate_count[1] += 1

    if dominate_count[0] == 0 and dominate_count[1] > 0:
        # elem2 dominates elem1
        return 1
    elif dominate_count[1] == 0 and dominate_count[0] > 0:
        # elem1 dominates elem2
        return -1
    else:
        return 0


'''
Assign rank as non-domination level.
'''


def fast_non_dominated_sort(population):
    S = []  # S[p] = set of solutions; the solution p dominates.
    n = []  # N[p] = domination count; the number of solutions which dominate p.
    sorted_by_rank = {}  # key = rank    value = set of indices.

    for p, p_idx in zip(population, range(len(population))):
        # initialize
        S.append(set())
        n.append(0)

        for q, q_idx in zip(population, range(len(population))):
            judge = dominate_operator(p, q)
            if judge == -1:
                # p dominates q
                S[p_idx].add(q_idx)
            elif judge == 1:
                # q dominates p
                n[p_idx] += 1

        if n[p_idx] == 0:
            p['rank'] = 1
            if not 1 in sorted_by_rank:
                sorted_by_rank[1] = set()
            sorted_by_rank[1].add(p_idx)

    pre_rank = 1
    next_rank = 2
    while len(sorted_by_rank[pre_rank]) != 0:
        sorted_by_rank[next_rank] = set()

        for p_idx in sorted_by_rank[pre_rank]:
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    population[q_idx]['rank'] = next_rank
                    sorted_by_rank[next_rank].add(q_idx)
        pre_rank = next_rank
        next_rank += 1


'''
Assign crowding distance as density estimation.
'''


def crowding_distance_assignment(population):
    # initialize
    for elem in population:
        elem['dist'] = 0

    # Calculate the sum of individual distance values
    # corresponding to each objective.
    for obj in OBJECTIVES:
        population.sort(key=lambda e: e[obj])
        max_val = population[-1][obj]
        min_val = population[0][obj]
        population[0]['dist'] = population[-1]['dist'] = INFINITE

        for i in range(1, len(population) - 1):
            cur_dist = population[i]['dist']
            if cur_dist == INFINITE:
                continue
            cur_dist += (population[i + 1][obj] - population[i - 1][obj]) / (max_val - min_val)
            population[i]['dist'] = cur_dist


def crowded_comparison_operator(elem1, elem2):
    # elem is dictionary{'acc', 'time', 'spec', 'rank'}
    # return -1: elem1 is optimal / 1: elem2 is optimal.
    if 'rank' in elem1.keys():
        if elem1['rank'] != elem2['rank']:
            return 1 if elem2['rank'] - elem1['rank'] > 0 else -1

    if elem1['dist'] == elem2['dist']:
        return 0
    else:
        return 1 if elem2['dist'] - elem1['dist'] > 0 else -1


def multi_objective_sort(arch_pool, seq_pool, arch_pool_valid_acc, arch_pool_lat):
    # Each element is one dictionary as { rank, val acc, lat, arch, seq }
    pool = []
    for arch, seq, acc, lat in zip(arch_pool, seq_pool, arch_pool_valid_acc, arch_pool_lat):
        elem = {}
        elem['arch'] = arch
        elem['seq'] = seq
        elem['acc'] = acc
        elem['lat'] = lat
        pool.append(elem)

    crowding_distance_assignment(pool)
    fast_non_dominated_sort(pool)
    sorted(pool, key=functools.cmp_to_key(crowded_comparison_operator))

    sorted_arch_pool = []
    sorted_seq_pool = []
    sorted_arch_pool_valid_acc = []
    sorted_arch_pool_lat = []
    for elem in pool:
        sorted_arch_pool.append(elem['arch'])
        sorted_seq_pool.append(elem['seq'])
        sorted_arch_pool_valid_acc.append(elem['acc'])
        sorted_arch_pool_lat.append(elem['lat'])

    return sorted_arch_pool, sorted_seq_pool, sorted_arch_pool_valid_acc, sorted_arch_pool_lat
