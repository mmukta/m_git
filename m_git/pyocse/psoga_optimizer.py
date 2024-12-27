import numpy as np
import matplotlib.pyplot as plt

class PSOGAOptimizer:
    def __init__(self, obj_function, bounds, num_particles=30, dimensions=2,
                 inertia=0.7, cognitive=1.5, social=1.5, mutation_rate=0.1, 
                 crossover_rate=0.8, max_iter=100, verbose=True, debug=False):
        """
        Initialize the PSO-GA hybrid optimizer.

        Parameters:
        - obj_function: Objective function to minimize.
        - bounds: Tuple (lower_bounds, upper_bounds) for each dimension.
        - num_particles: Number of particles (and individuals in GA).
        - dimensions: Number of dimensions in the search space.
        - inertia: Inertia weight for velocity update (PSO).
        - cognitive: Cognitive (personal best) weight (PSO).
        - social: Social (global best) weight (PSO).
        - mutation_rate: Probability of mutation (GA).
        - crossover_rate: Probability of crossover (GA).
        - max_iter: Maximum number of iterations.
        - verbose: If True, prints progress during optimization.
        - debug: If True, prints detailed debug information.
        """
        self.obj_function = obj_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_iter = max_iter
        self.verbose = verbose
        self.debug = True

        # Initialize bounds
        self.lower_bounds = np.array([b[0] for b in bounds])
        self.upper_bounds = np.array([b[1] for b in bounds])

        # Initialize particles
        self.positions = np.random.uniform(self.lower_bounds, self.upper_bounds, (num_particles, dimensions))
        self.velocities = np.random.uniform(self.lower_bounds, self.upper_bounds, (num_particles, dimensions))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.array([self.safe_evaluate(p) for p in self.positions])
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_scores)]
        self.global_best_score = np.min(self.personal_best_scores)

    def safe_evaluate(self, position):
        """Evaluate the objective function, handling exceptions gracefully."""
        try:
            score = self.obj_function(position)
            if np.isnan(score) or np.isinf(score):
                return np.inf  # Penalize invalid scores
            return score
        except Exception as e:
            if self.debug:
                print(f"Error evaluating position {position}: {e}")
            return np.inf

    def pso_step(self):
        """Perform one step of PSO."""
        for i in range(self.num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            #self.positions[i] = np.clip(self.positions[i], self.lower_bounds, self.upper_bounds)
            self.positions[i] += self.velocities[i]
            self.velocities[i] = (
                self.inertia * self.velocities[i] +
                self.cognitive * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                self.social * r2 * (self.global_best_position - self.positions[i])
            )
            #if self.debug:
            #print(f"Particle {i}: Velocity = {self.velocities[i]}")
            #print(f"Particle {i}: Position = {self.positions[i]}")
        
        
            score = self.safe_evaluate(self.positions[i])
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.positions[i]

        min_idx = np.argmin(self.personal_best_scores)
        if self.personal_best_scores[min_idx] < self.global_best_score:
            self.global_best_score = self.personal_best_scores[min_idx]
            self.global_best_position = self.personal_best_positions[min_idx]

    def ga_step(self):
        """Perform one step of the Genetic Algorithm (GA)."""
        new_population = []
        fitness = 1 / (1 + self.personal_best_scores)
        probabilities = fitness / np.sum(fitness)

        for _ in range(self.num_particles // 2):
            parents_idx = np.random.choice(range(self.num_particles), size=2, p=probabilities)
            parent1 = self.personal_best_positions[parents_idx[0]]
            parent2 = self.personal_best_positions[parents_idx[1]]

            # Crossover
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.dimensions)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            else:
                child1, child2 = parent1, parent2

            # Mutation
            for child in [child1, child2]:
                if np.random.rand() < self.mutation_rate:
                    mutation_idx = np.random.randint(0, self.dimensions)
                    child[mutation_idx] += np.random.normal(0, 0.1)
                    child = np.clip(child, self.lower_bounds, self.upper_bounds)
                #if self.debug:
                #print(f"Child Position after mutation: {child}")
                new_population.append(child)

        self.positions = np.array(new_population)
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.array([self.safe_evaluate(p) for p in self.positions])
        
    
    def optimize(self):
        """Perform optimization using the PSO-GA hybrid algorithm."""
        for iteration in range(self.max_iter):
            self.ga_step()
            self.pso_step()
            

            if self.verbose and iteration % 1 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.global_best_score:.4f}")
                #if self.debug:
                #print(f"Global Best Position: {self.global_best_position}")
                #print(f"Velocities:\n{self.velocities}")
                #print(f"Positions:\n{self.positions}")
        return self.global_best_position, self.global_best_score
