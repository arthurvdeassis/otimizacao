import random
import networkx as nx
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from deap.benchmarks.tools import hypervolume

def create_network_topology():
    G = nx.DiGraph()
    nodes = ['S', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'T']
    G.add_nodes_from(nodes)

    edges = [
        ('S', 'A', {'latency': 10, 'cost': 8, 'loss': 0.01, 'bw': 100}),
        ('S', 'B', {'latency': 20, 'cost': 4, 'loss': 0.005, 'bw': 150}),
        ('A', 'C', {'latency': 15, 'cost': 5, 'loss': 0.015, 'bw': 100}),
        ('A', 'D', {'latency': 25, 'cost': 3, 'loss': 0.02, 'bw': 80}),
        ('B', 'D', {'latency': 10, 'cost': 4, 'loss': 0.005, 'bw': 150}),
        ('B', 'E', {'latency': 30, 'cost': 2, 'loss': 0.01, 'bw': 200}),
        ('C', 'F', {'latency': 20, 'cost': 6, 'loss': 0.01, 'bw': 100}),
        ('D', 'F', {'latency': 15, 'cost': 4, 'loss': 0.005, 'bw': 150}),
        ('D', 'G', {'latency': 8, 'cost': 9, 'loss': 0.03, 'bw': 60}),
        ('E', 'G', {'latency': 12, 'cost': 3, 'loss': 0.01, 'bw': 200}),
        ('F', 'H', {'latency': 10, 'cost': 7, 'loss': 0.01, 'bw': 120}),
        ('F', 'T', {'latency': 25, 'cost': 5, 'loss': 0.02, 'bw': 80}),
        ('G', 'H', {'latency': 5, 'cost': 10, 'loss': 0.04, 'bw': 60}),
        ('H', 'T', {'latency': 15, 'cost': 4, 'loss': 0.01, 'bw': 120}),
        ('C', 'G', {'latency': 40, 'cost': 1, 'loss': 0.05, 'bw': 50}),
    ]
    G.add_edges_from(edges)
    return G


NETWORK = create_network_topology()
START_NODE, END_NODE = 'S', 'T'

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()


def get_random_path(graph, source, target):
    return random.choice(list(nx.all_simple_paths(graph, source, target)))


toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: get_random_path(NETWORK, START_NODE, END_NODE))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    path = individual
    total_latency, total_cost, total_reliability, bottleneck_bw = 0, 0, 1.0, float(
        'inf')
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = NETWORK[u][v]
        total_latency += edge_data['latency']
        total_cost += edge_data['cost']
        total_reliability *= (1 - edge_data['loss'])
        bottleneck_bw = min(bottleneck_bw, edge_data['bw'])

    total_loss = 1 - total_reliability
    return total_latency, total_cost, total_loss, bottleneck_bw


def crossover_paths(ind1, ind2):
    common_nodes = list(set(ind1[1:-1]) & set(ind2[1:-1]))
    if not common_nodes:
        return creator.Individual(ind1), creator.Individual(ind2)

    cross_node = random.choice(common_nodes)
    idx1, idx2 = ind1.index(cross_node), ind2.index(cross_node)

    child1_path = ind1[:idx1] + ind2[idx2:]
    child2_path = ind2[:idx2] + ind1[idx1:]

    return creator.Individual(child1_path), creator.Individual(child2_path)


def mutate_path(individual):
    if len(individual) < 3:
        return (individual,)

    start_idx, end_idx = sorted(random.sample(range(len(individual)), 2))
    if end_idx - start_idx < 2:
        return (individual,)
    
    start_node, end_node = individual[start_idx], individual[end_idx]
    original_sub_path = individual[start_idx:end_idx+1]

    try:
        alternative_paths = list(
            nx.all_simple_paths(NETWORK, start_node, end_node))
        if original_sub_path in alternative_paths:
            alternative_paths.remove(original_sub_path)

        if alternative_paths:
            new_sub_path = random.choice(alternative_paths)
            new_full_path = individual[:start_idx] + \
                new_sub_path + individual[end_idx+1:]
            return (creator.Individual(new_full_path),)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return (individual,)

    return (individual,)


toolbox.register("evaluate", evaluate)
toolbox.register("mate", crossover_paths)
toolbox.register("mutate", mutate_path)
toolbox.register("select", tools.selNSGA2)

def main():
    random.seed(42)

    NGEN = 100
    MU = 100
    CXPB = 0.7
    MUTPB = 0.2

    pop = toolbox.population(n=MU)
    pareto_front = tools.ParetoFront()

    ref_point = (200.0, 100.0, 1.0, 0.0)

    logbook = tools.Logbook()
    logbook.header = "gen", "nevals", "hypervolume"

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pareto_front.update(pop)

    hv = hypervolume(pareto_front.items, ref_point)
    logbook.record(gen=0, nevals=len(invalid_ind), hypervolume=hv)
    print(logbook.stream)

    for gen in range(1, NGEN + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pareto_front.update(offspring)
        pop = toolbox.select(pop + offspring, MU)
        hv = hypervolume(pareto_front.items, ref_point)
        logbook.record(gen=gen, nevals=len(invalid_ind), hypervolume=hv)

        print(logbook.stream)

    print(
        f"\nOtimização concluída. Fronteira de Pareto contém {len(pareto_front)} soluções.")

    qos_profiles = {
        "VoIP": {"latency": 0.5, "cost": 0.1, "loss": 0.4, "bw": 0.0},
        "File Transfer": {"latency": 0.0, "cost": 0.4, "loss": 0.0, "bw": 0.6},
        "General": {"latency": 0.25, "cost": 0.25, "loss": 0.25, "bw": 0.25}
    }

    def select_best_for_qos(pareto, profile):
        best_sol, best_score = None, -float('inf')

        for sol in pareto:
            lat, cost, loss, bw = sol.fitness.values
            # Normalização simples para o score (valor / valor_maximo_estimado)
            score = (profile['bw'] * (bw / 200.0)) - \
                    (profile['latency'] * (lat / 200.0)) - \
                    (profile['cost'] * (cost / 100.0)) - \
                    (profile['loss'] * loss)

            if score > best_score:
                best_score, best_sol = score, sol
        return best_sol

    print("\n--- Análise Detalhada da Fronteira de Pareto ---")
    for i, sol in enumerate(pareto_front):
        m = sol.fitness.values
        print(f"\nSolução #{i+1}:")
        print(f"  - Caminho: {' -> '.join(sol)}")
        print(
            f"  - Métricas: Lat={m[0]:.1f}, Custo={m[1]:.1f}, Perda={m[2]*100:.2f}%, BW={m[3]:.1f}")

    print("\n--- Políticas de Roteamento por QoS ---")
    if not pareto_front:
        print("Nenhuma solução foi encontrada na fronteira de Pareto.")
        return

    for name, profile in qos_profiles.items():
        best_route = select_best_for_qos(pareto_front, profile)
        metrics = best_route.fitness.values
        print(f"\nMelhor Rota para '{name}':")
        print(f"  - Caminho: {' -> '.join(best_route)}")
        print(f"  - Latência: {metrics[0]:.2f} ms")
        print(f"  - Custo: {metrics[1]:.2f}")
        print(f"  - Perda: {metrics[2]*100:.2f}%")
        print(f"  - Largura de Banda: {metrics[3]:.2f} Mbps")

    latencies = [s.fitness.values[0] for s in pareto_front]
    costs = [s.fitness.values[1] for s in pareto_front]
    bws = [s.fitness.values[3] for s in pareto_front]

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(latencies, costs, c=bws,
                          cmap='viridis', s=150, alpha=0.8, edgecolors='k')
    plt.title("Fronteira de Pareto Ótima (Latência vs. Custo)", fontsize=16)
    plt.xlabel("Latência Total (ms)", fontsize=12)
    plt.ylabel("Custo Total", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Largura de Banda (Mbps)', fontsize=12)

    plt.savefig("pareto_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    gen_nums = logbook.select("gen")
    hv_values = logbook.select("hypervolume")

    plt.figure(figsize=(12, 8))
    plt.plot(gen_nums, hv_values, '-o', color='royalblue', markersize=5)
    plt.title("Convergência do Indicador de Hipervolume", fontsize=16)
    plt.xlabel("Geração", fontsize=12)
    plt.ylabel("Valor do Hipervolume", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.savefig("hypervolume_convergence.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
