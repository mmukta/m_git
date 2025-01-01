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

        # Initialize particles and Rescale to (0, 1)
        self.positions = np.random.uniform(self.lower_bounds, self.upper_bounds, (num_particles, dimensions))
        self.positions = (self.positions - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
        self.velocities = 0.1 * np.random.uniform(-1, 1, (num_particles, dimensions))

        self.personal_best_positions = np.copy(self.positions)
        # evaluate the score for each particle
        self.personal_best_scores = np.zeros(len(self.positions))
        for i, p in enumerate(self.positions):
            p_actual = self.rescale(p)
            score = self.safe_evaluate(p_actual)
            self.personal_best_scores[i] = score
            print(f"{i} score: {score} max_V: {np.abs(self.velocities[i]).max()}")
            #print(self.positions[i])
        #self.personal_best_scores = np.array([self.safe_evaluate(p) for p in self.positions])

        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_scores)]
        self.global_best_score = np.min(self.personal_best_scores)

    def rescale(self, scaled_values):
        """
        Rescale values from (0, 1) back to original bounds.
        """
        return scaled_values * (self.upper_bounds - self.lower_bounds) + self.lower_bounds

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

            #print("PSO_step", i, self.positions[i], self.velocities[i])
            # update velocity
            r1, r2 = np.random.rand(), np.random.rand()
            self.positions[i] += self.velocities[i]
            #self.velocities[i] = (
            v = (
                self.inertia * self.velocities[i] + 
                self.cognitive * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                self.social * r2 * (self.global_best_position - self.positions[i]) + 
                0.05 * np.random.uniform(-1, 1, (self.dimensions))
            )
            v /= np.abs(v).max()
            self.velocities[i] = 0.1 * v #(-0.1, 0.1)
            
            #if self.debug:
            #print(f"Particle {i}: Velocity = {self.velocities[i]}")
            #print(f"Particle {i}: Position = {self.positions[i]}")
        
            # update position       
            #self.positions[i] += self.velocities[i]
            # position 100, vel: 10 (inertia: 11, congitive -1, social -1) 110 105
            #self.positions[i] = np.clip(self.positions[i], self.lower_bounds, self.upper_bounds)
            self.positions[i] = np.clip(self.positions[i], 0, 1)
    
            # update score
            p_actual = self.rescale(self.positions[i])
            score = self.safe_evaluate(p_actual)
            #print('\nP scale', self.positions[i]); print("\nP_actual", p_actual); import sys; sys.exit()

            strs = f"{i} score: {score}  pbest: {self.personal_best_scores[i]}"
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.positions[i]
                strs += " ++++++++++++"
            print(strs)

        strs = f"Best Score: {self.global_best_score:.4f}"
        min_idx = np.argmin(self.personal_best_scores)
        if self.personal_best_scores[min_idx] < self.global_best_score:
            self.global_best_score = self.personal_best_scores[min_idx]
            self.global_best_position = self.personal_best_positions[min_idx]
            strs += " ============================="

        print(strs)

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
        
    
    def optimize(self, x0=None):
        """Perform optimization using the PSO-GA hybrid algorithm."""
        if x0 is not None:
            self.positions = x0

        for iteration in range(self.max_iter):
            #self.ga_step()
            self.pso_step()
            
            if self.verbose and iteration % 1 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.global_best_score:.4f}")
                #if self.debug:
                #print(f"Global Best Position: {self.global_best_position}")
                #print(f"Velocities:\n{self.velocities}")
                #print(f"Positions:\n{self.positions}")
        return self.global_best_position, self.global_best_score
