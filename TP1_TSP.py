import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_cities(file_path):
    pontos_x = []
    pontos_y = []
    with open(file_path, 'r') as file:
        for linha in file:
            coordenadas = linha.strip().replace(',', ' ').split()
            if len(coordenadas) == 2:
                try:
                    x = float(coordenadas[0])
                    y = float(coordenadas[1])
                    pontos_x.append(x)
                    pontos_y.append(y)
                except ValueError:
                    continue

    # Plotando os pontos
    plt.scatter(pontos_x, pontos_y, color='black', marker='o')
    plt.xlabel('COORDENADA X')
    plt.ylabel('COORDENADA Y')
    plt.title('CIDADES')
    plt.grid(True)
    plt.xlim(min(pontos_x) - 2, max(pontos_x) + 2)
    plt.ylim(min(pontos_y) - 2, max(pontos_y) + 2)
    plt.show()


class AntColonyOptimization:
    def __init__(self, num_cities, distance_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_constant, initial_pheromone_level, pheromone_min, pheromone_max, chromosome_size, mutation_rate, elitism_rate, crossover_rate):
        self.num_cities = num_cities
        self.distance_matrix = distance_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_constant = pheromone_constant
        self.pheromone_levels = np.ones(
            (num_cities, num_cities)) * initial_pheromone_level
        self.pheromone_min = pheromone_min
        self.pheromone_max = pheromone_max
        self.chromosome_size = chromosome_size
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.crossover_rate = crossover_rate
        self.best_route = None
        self.best_route_length = float('inf')
        self.fitness_history = []

    def run(self):
        for iteration in range(self.num_iterations):
            routes = []
            route_lengths = []
            for ant in range(self.num_ants):
                route, route_length = self.construct_solution()
                routes.append(route)
                route_lengths.append(route_length)
                if route_length < self.best_route_length:
                    self.best_route_length = route_length
                    self.best_route = route
            self.update_pheromones(routes, route_lengths)
            self.fitness_history.append(self.best_route_length)
        return self.best_route, self.best_route_length

    def construct_solution(self):
        start_city = np.random.randint(0, self.num_cities)
        route = [start_city]
        unvisited_cities = set(range(self.num_cities)) - {start_city}
        route_length = 0
        current_city = start_city

        while unvisited_cities:
            next_city = self.select_next_city(current_city, unvisited_cities)
            route.append(next_city)
            route_length += self.distance_matrix[current_city, next_city]
            unvisited_cities.remove(next_city)
            current_city = next_city

        route_length += self.distance_matrix[current_city, start_city]
        route.append(start_city)
        return route, route_length

    def select_next_city(self, current_city, unvisited_cities):
        pheromones = np.array(
            [self.pheromone_levels[current_city, next_city] for next_city in unvisited_cities])
        distances = np.array([self.distance_matrix[current_city, next_city]
                             for next_city in unvisited_cities])
        desirability = pheromones ** self.alpha * \
            (1.0 / distances) ** self.beta
        probabilities = desirability / desirability.sum()
        next_city = np.random.choice(list(unvisited_cities), p=probabilities)
        return next_city

    def update_pheromones(self, routes, route_lengths):
        self.pheromone_levels *= (1 - self.evaporation_rate)
        for route, route_length in zip(routes, route_lengths):
            pheromone_delta = self.pheromone_constant / route_length
            for i in range(len(route) - 1):
                updated_pheromone = self.pheromone_levels[route[i],
                                                          route[i + 1]] + pheromone_delta
                self.pheromone_levels[route[i], route[i + 1]] = np.clip(
                    updated_pheromone, self.pheromone_min, self.pheromone_max)
                self.pheromone_levels[route[i + 1], route[i]] = np.clip(
                    updated_pheromone, self.pheromone_min, self.pheromone_max)


def run_experiment(num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_constant, chromosome_size, mutation_rate, elitism_rate, crossover_rate, num_runs):
    try:
        cities = pd.read_csv("cidades.txt", sep=r'\s+',
                             header=None, names=["x", "y"])
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return

    num_cities = cities.shape[0]
    coordinates = cities.values

    # Matriz de distâncias
    distance_matrix = np.linalg.norm(
        coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=-1)

    all_fitness_histories = []
    best_overall_route = None
    best_overall_length = float('inf')

    for run in range(num_runs):
        aco = AntColonyOptimization(num_cities, distance_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_constant, initial_pheromone_level=0.1,
                                    pheromone_min=0.1, pheromone_max=10.0, chromosome_size=chromosome_size, mutation_rate=mutation_rate, elitism_rate=elitism_rate, crossover_rate=crossover_rate)
        best_route, best_route_length = aco.run()
        all_fitness_histories.append(aco.fitness_history)

        if best_route_length < best_overall_length:
            best_overall_length = best_route_length
            best_overall_route = best_route

    all_fitness_histories = np.array(all_fitness_histories)
    mean_fitness = np.mean(all_fitness_histories, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_iterations + 1), mean_fitness, marker="o",
             linestyle="-", color="black", label="FITNESS")
    plt.title("FITNESS AO LONGO DAS GERAÇÕES")
    plt.xlabel("GERAÇÃO")
    plt.ylabel("MENOS DISTÂNCIA")
    plt.grid(True)
    plt.legend()
    plt.show()

    best_coordinates = coordinates[best_overall_route]
    plt.figure(figsize=(10, 5))
    plt.plot(best_coordinates[:, 0], best_coordinates[:,
             1], marker="o", linestyle="-", color="black")
    plt.scatter(best_coordinates[0, 0],
                best_coordinates[0, 1], color="red", s=100)
    plt.title(f"MELHOR COMPRIMENTO DE ROTA ENCONTRADO: {best_overall_length}")
    plt.xlabel("COORDENADA X")
    plt.ylabel("COORDENADA Y")
    plt.grid(False)
    plt.show()

    return best_overall_length

if __name__ == "__main__":
    plot_cities('cidades.txt')
#     run_experiment( # Cenário 1 - Equilíbrio
#     num_ants=50,
#     num_iterations=150,
#     alpha=1.0,
#     beta=3.0,
#     evaporation_rate=0.5,
#     pheromone_constant=100,
#     chromosome_size=10,
#     mutation_rate=0.05,
#     elitism_rate=0.55,
#     crossover_rate=0.6,
#     num_runs=10
# )

#     run_experiment( # Cenário 2 - Alta exploração (menos feromônio, mais aleatoriedade)
#     num_ants=30,
#     num_iterations=100,
#     alpha=0.5,
#     beta=2.0,
#     evaporation_rate=0.7,
#     pheromone_constant=100,
#     chromosome_size=10,
#     mutation_rate=0.05,
#     elitism_rate=0.6,
#     crossover_rate=0.6,
#     num_runs=10
# )

#     run_experiment( # Cenário 3 - Alta exploração e maior elitismo
#         num_ants=20,
#         num_iterations=150,
#         alpha=1.0,
#         beta=2.5,
#         evaporation_rate=0.5,
#         pheromone_constant=80,
#         chromosome_size=10,
#         mutation_rate=0.05,
#         elitism_rate=0.75,
#         crossover_rate=0.6,
#         num_runs=10
# )
