import random
import time
from ortools.sat.python import cp_model

def generate_erdos_renyi(N, p, seed=None):
    if seed is not None:
        random.seed(seed)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if random.random() < p:
                edges.append((i, j))
    return edges

def color_graph(C, N, edges):
    model = cp_model.CpModel()

    vars = [model.new_int_var(0, C - 1, f"v{i}") for i in range(N)]

    model.add(vars[0] == 0)

    for i, j in edges:
        model.add(vars[i] != vars[j])

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [solver.value(v) for v in vars]

    return None

def find_min_coloring(N, edges):
    low, high = 1, N
    best_solution = None
    best_k = None
    while low <= high:
        mid = (low + high) // 2
        t0 = time.time()
        sol = color_graph(mid, N, edges)
        t1 = time.time()
        print(f"C = {mid}, time used = {t1 - t0:.2f} sec")
        if sol is not None:
            best_k, best_solution = mid, sol
            high = mid - 1
        else:
            low = mid + 1
    return best_k, best_solution

if __name__ == "__main__":
    N = 55
    P = 0.3
    SEED = 42

    edges = generate_erdos_renyi(N, P, seed=SEED)

    start = time.time()
    k, coloring = find_min_coloring(N, edges)
    elapsed = time.time() - start

    print(f"Minimum number of colors: {k}")
    print(coloring)
    print(f"Total time elapsed: {elapsed:.2f} seconds")
