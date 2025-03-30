import numpy as np
import matplotlib.pyplot as plt
import time  # Importando o módulo para medir o tempo de execução

# Definir a função Rastrigin


def rastrigin_function(x, y, A=10):
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

# Algoritmo Híbrido ACO-GA


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

        # População inicial (cromossomos)
        # Rastrigin typically works in [-5.12, 5.12]
        self.population = np.random.uniform(-5.12, 5.12, (population_size, 2))
        self.best_position = None
        self.best_value = float('inf')  # Rastrigin is a minimization problem

        # Feromônios
        self.pheromone_levels = np.ones((grid_size, grid_size))

        # Criar o grid de valores x e y no domínio [-5.12, 5.12]
        self.x = np.linspace(-5.12, 5.12, grid_size)
        self.y = np.linspace(-5.12, 5.12, grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def run(self):
        for generation in range(self.num_generations):
            # Avaliar a população
            fitness_values = self.evaluate_population()

            # Seleção: Torneio para escolher os pais
            selected_parents = self.selection(fitness_values)

            # Cruzamento e mutação
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected_parents[i], selected_parents[i+1]
                offspring1, offspring2 = self.crossover(parent1, parent2)
                new_population.append(offspring1)
                new_population.append(offspring2)

            # Mutação na população
            self.mutation(new_population)

            # Atualizar a população
            self.population = np.array(new_population)

            # ACO: Exploração local com formigas
            self.aco_exploration()

            # Atualizar o melhor valor encontrado
            fitness_values = self.evaluate_population()
            # Como é minimização, procuramos o menor valor
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
        # Seleção por torneio
        selected = []
        for _ in range(self.population_size):
            tournament = np.random.choice(self.population_size, 4)
            # Seleção do menor valor (minimização)
            best_idx = tournament[np.argmin(fitness_values[tournament])]
            selected.append(self.population[best_idx])
        return np.array(selected)

    def crossover(self, parent1, parent2):
        # Crossover simples de 1 ponto
        crossover_point = np.random.randint(0, 2)
        offspring1 = np.copy(parent1)
        offspring2 = np.copy(parent2)
        offspring1[crossover_point:] = parent2[crossover_point:]
        offspring2[crossover_point:] = parent1[crossover_point:]
        return offspring1, offspring2

    def mutation(self, new_population):
        # Mutação: alteração aleatória de um gene
        for i in range(len(new_population)):
            if np.random.rand() < self.mutation_rate:
                mutation_point = np.random.randint(0, 2)
                new_population[i][mutation_point] = np.random.uniform(
                    -5.12, 5.12)

    def aco_exploration(self):
        # Exploração das soluções com formigas
        solutions = []
        solution_values = []

        for ant in range(self.num_ants):
            # Cada formiga começa com uma solução aleatória
            x_idx, y_idx = self.construct_solution()
            solutions.append((x_idx, y_idx))
            value = rastrigin_function(self.x[x_idx], self.y[y_idx])
            solution_values.append(value)

            # Atualizar o melhor valor encontrado
            if value < self.best_value:
                self.best_value = value
                self.best_position = (self.x[x_idx], self.y[y_idx])

        # Atualizar feromônios após cada iteração de formigas
        self.update_pheromones(solutions, solution_values)

    def construct_solution(self):
        # Construção da solução pelas formigas
        x_idx = np.random.randint(0, self.grid_size)
        y_idx = np.random.randint(0, self.grid_size)

        for _ in range(10):  # Número de passos da formiga
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

        # Garantir que os valores sejam positivos
        # Shift para evitar valores negativos ou zero
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


# Função principal para rodar o experimento
if __name__ == "__main__":
    grid_size = 100
    population_size = 60  # População intermediária (60 indivíduos)
    num_generations = 40  # Número de gerações moderado (40)
    mutation_rate = 0.04  # 4% de mutação
    crossover_rate = 0.65  # 65% de taxa de cruzamento
    num_ants = 20  # Número de formigas (número par)
    num_iterations = 120  # Número de iterações intermediário para o ACO
    alpha = 0.8  # Reduzir a influência das feromônias no ACO
    beta = 2.5  # Aumentar o peso de qualidade das soluções
    evaporation_rate = 0.45  # Moderada evaporação
    pheromone_constant = 1.0  # Padrão de feromônio constante
    num_runs = 10

    best_value_over_runs = []
    best_position_over_runs = []

    start_time_total = time.time()  # Começar a medir o tempo total

    # Rodar o algoritmo múltiplas vezes
    for _ in range(num_runs):
        hybrid_algorithm = HybridACO_GA(population_size, num_generations, grid_size, mutation_rate,
                                        crossover_rate, num_ants, num_iterations, alpha, beta, evaporation_rate, pheromone_constant)
        best_position, best_value = hybrid_algorithm.run()
        best_value_over_runs.append(best_value)
        best_position_over_runs.append(best_position)

    end_time_total = time.time()  # Fim da medição do tempo total

    # Calcular as métricas
    min_value_found = np.min(best_value_over_runs)
    best_position = best_position_over_runs[np.argmin(best_value_over_runs)]
    mean_best_value = np.mean(best_value_over_runs)
    total_execution_time = end_time_total - \
        start_time_total  # Tempo total de execução

    # Print das métricas
    print(
        f"Valor mínimo encontrado: {min_value_found}")
    print(
        f"Melhor posição encontrada: x = {best_position[0]}, y = {best_position[1]}")
    print(
        f"Média do valor mínimo ao longo das 10 execuções: {mean_best_value}")
    print(
        f"Tempo total de execução: {total_execution_time:.4f} segundos")

    # Plotar a solução 3D para a melhor posição encontrada
    best_position = best_position_over_runs[np.argmin(best_value_over_runs)]
    fig = plt.figure(figsize=(10, 8))

    # Plot 3D
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(hybrid_algorithm.X, hybrid_algorithm.Y, rastrigin_function(
        hybrid_algorithm.X, hybrid_algorithm.Y), cmap='viridis')
    ax.scatter(best_position[0], best_position[1],
            min_value_found, color='r', s=100, label="Melhor solução")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Ajustar a visualização para ver de frente
    ax.view_init(elev=10, azim=-65)

    # Plot contorno 2D
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
