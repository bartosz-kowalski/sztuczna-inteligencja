import random
import math
import tsplib95
import matplotlib.pyplot as plt

def wczytaj_probelm(path):
    problem = tsplib95.load(path)
    nodes = list(problem.get_nodes())

    kordy = {}
    if problem.node_coords:
        for node, (x, y) in problem.node_coords.items():
            kordy[node] = (x, y)
    else:
        kordy = None

    return problem, nodes, kordy

def trasa_length(trasa, problem):
    total = 0
    n = len(trasa)
    for i in range(n):
        a = trasa[i]
        b = trasa[(i + 1) % n] 
        total += problem.get_weight(a, b)
    return total

def two_opt_neighbor(trasa):
    """Zwraca nową trasę powstałą przez odwrócenie losowego fragmentu (2-opt)."""
    n = len(trasa)
    i, j = sorted(random.sample(range(n), 2))
    neighbor = trasa[:]
    neighbor[i:j+1] = reversed(neighbor[i:j+1])
    return neighbor

def wyzarzanie(problem, nodes, initial_temp=1000.0, final_temp=1e-3, alpha=0.995,iterations_per_temp=500):
    obecna_trasa = nodes[:]
    random.shuffle(obecna_trasa)

    obecna_dlug = trasa_length(obecna_trasa, problem)
    naj_trasa = obecna_trasa[:]
    naj_dlug = obecna_dlug

    temp = initial_temp

    naj_dlug_poprzednie = []

    while temp > final_temp:
        for _ in range(iterations_per_temp):
            candidate_trasa = two_opt_neighbor(obecna_trasa)
            candidate_dlug = trasa_length(candidate_trasa, problem)

            delta = candidate_dlug - obecna_dlug

            if delta < 0:
                # lepsze rozwiązanie akcepotwane zawsze
                obecna_trasa = candidate_trasa
                obecna_dlug = candidate_dlug
                if candidate_dlug < naj_dlug:
                    naj_trasa = candidate_trasa
                    naj_dlug = candidate_dlug
            else:
                # gorsze rozwiązanie akceptujemy z pewnym prawdopowdobieństwem
                if random.random() < math.exp(-delta / temp):
                    obecna_trasa = candidate_trasa
                    obecna_dlug = candidate_dlug

            naj_dlug_poprzednie.append(naj_dlug)

        # zmniejszanie temperatury
        temp *= alpha

    return naj_trasa, naj_dlug, naj_dlug_poprzednie

tsp_path = "xqf131.tsp"

problem, nodes, kordy = wczytaj_probelm(tsp_path)
print(f"Liczba miast: {len(nodes)}")

naj_trasa, naj_dlug, poprzednie = wyzarzanie(problem, nodes, initial_temp=1000.0, final_temp=1e-3, alpha=0.995, iterations_per_temp=500)

print("Najlepsza znaleziona trasa długość:", naj_dlug)
print("Najlepsza trasa pierwsze 30 miast:", naj_trasa[:30])

plt.figure(figsize=(8, 4))
plt.plot(poprzednie)
plt.xlabel("Iteracja")
plt.ylabel("Najlepsza długość trasy")
plt.title("Zbieżność algorytmu symulowanego wyżarzania")
plt.grid(True)
plt.tight_layout()
plt.show()

xs = [kordy[node][0] for node in naj_trasa] + [kordy[naj_trasa[0]][0]]
ys = [kordy[node][1] for node in naj_trasa] + [kordy[naj_trasa[0]][1]]

plt.figure(figsize=(6, 6))
plt.scatter(xs, ys, c="red", s=10)
plt.plot(xs, ys, c="blue", linewidth=0.8)
plt.title("Najlepsza znaleziona trasa")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.tight_layout()
plt.show()
