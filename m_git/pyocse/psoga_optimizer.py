import numpy as np

class PSOGAOptimizer:
    def __init__(self, obj_function, obj_args, bounds, seed=None,
                 num_particles=30, dimensions=2,
                 inertia=0.7, cognitive=1.5, social=1.5, 
                 mutation_rate=0.1, crossover_rate=0.8, 
                 max_iter=100, ncpu=1, verbose=True, debug=False):
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
        - ncpu: number of parallel processes
        - verbose: If True, prints progress during optimization.
        - debug: If True, prints detailed debug information.
        """
        self.obj_function = obj_function
        self.obj_args = obj_args
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
        self.ncpu = ncpu
        self.debug = True
        self.obj_args = self.obj_args + (self.ncpu, )

        # Initialize bounds
        self.lower_bounds = np.array([b[0] for b in bounds])
        self.upper_bounds = np.array([b[1] for b in bounds])

        # Initialize particles and Rescale to (0, 1)
        self.positions = np.random.uniform(self.lower_bounds, 
                                           self.upper_bounds, 
                                           (num_particles, dimensions))
        self.positions = (self.positions - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
        self.velocities = 0.1 * np.random.uniform(-1, 1, (num_particles, dimensions))

        if seed is not None:
            self.set_seeds(seed)

        # Init: evaluate the score for each particle
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.zeros(len(self.positions))
        scores = self.safe_evaluate_par()
        for i in range(self.num_particles):
            #score = self.safe_evaluate(p_actual)
            self.personal_best_scores[i] = scores[i]
            print(f"{i} score: {scores[i]}")

        min_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[min_idx]
        self.global_best_score = np.min(self.personal_best_scores)

    def set_seeds(self, seed):
        """
        reset the positions to the seed
        """
        n_seeds = len(seed)
        print("Set Seeds", n_seeds)
        self.positions[:n_seeds] = (seed - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)

    def rescale(self, scaled_values):
        """
        Rescale values from (0, 1) back to original bounds.
        """
        return scaled_values * (self.upper_bounds - self.lower_bounds) + self.lower_bounds

    def safe_evaluate_obsolete(self, position):
        """Evaluate the objective function, handling exceptions gracefully."""
        score = self.obj_function(position, *self.obj_args)
        return score

    def safe_evaluate_par(self):
        p_actuals = []
        for i, p in enumerate(self.positions):
            p_actual = self.rescale(p)
            p_actuals.append(p_actual)

        scores = self.obj_function(p_actuals, *self.obj_args)
        return scores

    def pso_step(self):
        """Perform one step of PSO."""
        for i in range(self.num_particles):
            # update velocity
            r1, r2 = np.random.rand(), np.random.rand()
            self.positions[i] += self.velocities[i]
            v = (
                 self.inertia * self.velocities[i] + 
                 self.cognitive * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                 self.social * r2 * (self.global_best_position - self.positions[i]) + 
                 0.05 * np.random.uniform(-1, 1, (self.dimensions))
            )
            v /= np.abs(v).max()
            self.velocities[i] = 0.1 * v #(-0.1, 0.1)
            self.positions[i] = np.clip(self.positions[i], 0, 1)
        
        # Evaluate results in parallel 
        scores = self.safe_evaluate_par()

        for i in range(self.num_particles):
            score = scores[i]
            strs = f"{i} score: {score}  pbest: {self.personal_best_scores[i]}"
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.positions[i]
                strs += " ++++++++++++"
            print(strs)

        min_idx = np.argmin(self.personal_best_scores)
        strs = f"Best Score: {min_idx} {self.global_best_score:.4f}"
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
        best_position = self.rescale(self.global_best_position)
        return best_position, self.global_best_score
