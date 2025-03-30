import numpy as np
import matplotlib.pyplot as plt
import time


def rastrigin_function(x, y, A=10):
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))


class HybridACO_GA:
    def __init__(self, population_size, num_generations, grid_size, mutation_rate, crossover_rate, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_constant):
        self.population_size = population_size
        self.num_generations = num_generations
        self.grid_size = grid_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_constant = pheromone_constant
        self.population = np.random.uniform(-5.12, 5.12, (population_size, 2))
        self.best_position = None
        self.best_value = float('inf')
        self.pheromone_levels = np.ones((grid_size, grid_size))
        self.x = np.linspace(-5.12, 5.12, grid_size)
        self.y = np.linspace(-5.12, 5.12, grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def run(self):
        for generation in range(self.num_generations):
            fitness_values = self.evaluate_population()
            selected_parents = self.selection(fitness_values)
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected_parents[i], selected_parents[i+1]
                offspring1, offspring2 = self.crossover(parent1, parent2)
                new_population.append(offspring1)
                new_population.append(offspring2)
            self.mutation(new_population)
            self.population = np.array(new_population)
            self.aco_exploration()
            fitness_values = self.evaluate_population()
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < self.best_value:
                self.best_value = fitness_values[best_idx]
                self.best_position = self.population[best_idx]
        return self.best_position, self.best_value

    def evaluate_population(self):
        fitness_values = []
        for individual in self.population:
            x, y = individual
            fitness = rastrigin_function(x, y)
            fitness_values.append(fitness)
        return np.array(fitness_values)

    def selection(self, fitness_values):
        selected = []
        for _ in range(self.population_size):
            tournament = np.random.choice(self.population_size, 4)
            best_idx = tournament[np.argmin(fitness_values[tournament])]
            selected.append(self.population[best_idx])
        return np.array(selected)

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(0, 2)
        offspring1 = np.copy(parent1)
        offspring2 = np.copy(parent2)
        offspring1[crossover_point:] = parent2[crossover_point:]
        offspring2[crossover_point:] = parent1[crossover_point:]
        return offspring1, offspring2

    def mutation(self, new_population):
        for i in range(len(new_population)):
            if np.random.rand() < self.mutation_rate:
                mutation_point = np.random.randint(0, 2)
                new_population[i][mutation_point] = np.random.uniform(
                    -5.12, 5.12)

    def aco_exploration(self):
        solutions = []
        solution_values = []
        for ant in range(self.num_ants):
            x_idx, y_idx = self.construct_solution()
            solutions.append((x_idx, y_idx))
            value = rastrigin_function(self.x[x_idx], self.y[y_idx])
            solution_values.append(value)
            if value < self.best_value:
                self.best_value = value
                self.best_position = (self.x[x_idx], self.y[y_idx])
        self.update_pheromones(solutions, solution_values)

    def construct_solution(self):
        x_idx = np.random.randint(0, self.grid_size)
        y_idx = np.random.randint(0, self.grid_size)
        for _ in range(10):
            neighbors = self.get_neighbors(x_idx, y_idx)
            probabilities = self.calculate_probabilities(
                x_idx, y_idx, neighbors)
            next_point = np.random.choice(len(neighbors), p=probabilities)
            x_idx, y_idx = neighbors[next_point]
        return x_idx, y_idx

    def get_neighbors(self, x_idx, y_idx):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x_idx + dx, y_idx + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    neighbors.append((nx, ny))
        return neighbors

    def calculate_probabilities(self, x_idx, y_idx, neighbors):
        pheromones = np.array([self.pheromone_levels[nx, ny]
                              for nx, ny in neighbors])
        values = np.array([rastrigin_function(self.x[nx], self.y[ny])
                          for nx, ny in neighbors])
        values = values - values.min() + 1e-6
        desirability = pheromones**self.alpha * values**self.beta
        desirability = np.clip(desirability, a_min=0, a_max=None)
        if desirability.sum() == 0:
            probabilities = np.ones(len(neighbors)) / len(neighbors)
        else:
            probabilities = desirability / desirability.sum()
        probabilities = probabilities / probabilities.sum()
        return probabilities

    def update_pheromones(self, solutions, solution_values):
        self.pheromone_levels *= (1 - self.evaporation_rate)
        for (x_idx, y_idx), value in zip(solutions, solution_values):
            self.pheromone_levels[x_idx,
                                  y_idx] += self.pheromone_constant * value


if __name__ == "__main__":
    grid_size = 100
    population_size = 60
    num_generations = 40
    mutation_rate = 0.04
    crossover_rate = 0.65
    num_ants = 20
    num_iterations = 120
    alpha = 0.8
    beta = 2.5
    evaporation_rate = 0.45
    pheromone_constant = 1.0
    num_runs = 10

    best_value_over_runs = []
    best_position_over_runs = []

    start_time_total = time.time()

    for _ in range(num_runs):
        hybrid_algorithm = HybridACO_GA(population_size, num_generations, grid_size, mutation_rate,
                                        crossover_rate, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_constant)
        best_position, best_value = hybrid_algorithm.run()
        best_value_over_runs.append(best_value)
        best_position_over_runs.append(best_position)

    end_time_total = time.time()

    min_value_found = np.min(best_value_over_runs)
    best_position = best_position_over_runs[np.argmin(best_value_over_runs)]
    mean_best_value = np.mean(best_value_over_runs)
    total_execution_time = end_time_total - start_time_total

    print(f"Valor mínimo encontrado: {min_value_found}")
    print(
        f"Melhor posição encontrada: x = {best_position[0]}, y = {best_position[1]}")
    print(
        f"Média do valor mínimo ao longo das 10 execuções: {mean_best_value}")
    print(f"Tempo total de execução: {total_execution_time:.4f} segundos")

    best_position = best_position_over_runs[np.argmin(best_value_over_runs)]
    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(hybrid_algorithm.X, hybrid_algorithm.Y, rastrigin_function(
        hybrid_algorithm.X, hybrid_algorithm.Y), cmap='viridis')
    ax.scatter(best_position[0], best_position[1],
               min_value_found, color='r', s=100, label="Melhor solução")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    ax.view_init(elev=10, azim=-65)

    ax2 = fig.add_subplot(122)
    contour_levels = np.linspace(np.min(rastrigin_function(hybrid_algorithm.X, hybrid_algorithm.Y)),
                                 np.max(rastrigin_function(hybrid_algorithm.X, hybrid_algorithm.Y)), 20)
    contour = ax2.contour(hybrid_algorithm.X, hybrid_algorithm.Y, rastrigin_function(
        hybrid_algorithm.X, hybrid_algorithm.Y), contour_levels, cmap='viridis')
    ax2.scatter(best_position[0], best_position[1],
                color='r', s=100, label="Melhor solução")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()

    plt.show()
