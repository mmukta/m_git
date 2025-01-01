from psoga_optimizer import PSOGAOptimizer
from pyocse.parameters import ForceFieldParameters
from time import time
import os
import numpy as np

np.random.seed(7)

params = ForceFieldParameters(
    smiles=['CC(=O)OC1=CC=CC=C1C(=O)O'],
    f_coef=1,
    s_coef=1,
    e_coef=1,
    style='openff',
    ref_evaluator='mace',
    ncpu=1
)

p0, errors = params.load_parameters("dataset/parameters.xml")
ref_dics = params.load_references("dataset/references.xml")[:20]

os.makedirs("ASP2", exist_ok=True)
os.chdir("ASP2")

t0 = time()
e_offset, params_opt = params.optimize_offset(ref_dics, p0)
params.update_ff_parameters(params_opt)
print(params.get_objective(ref_dics, e_offset, obj="MSE"))


def generate_bounds(params, terms, params_opt):
    """
    Generate realistic bounds for the selected terms in the force field parameters.

    Args:
        params: ForceFieldParameters object.
        terms: List of terms (e.g., ["bond", "angle"]).
        params_opt: Current parameters.

    Returns:
        List of tuples representing bounds for each parameter.
    """
    opt_dict = params.get_opt_dict(terms, None, params_opt)
    bounds = []

    for term, values in opt_dict.items():
        for v in values:
            if term == "bond":  # Bond force constants and lengths
                bounds.append((max(50, v * 0.75), min(500, v * 1.5)))  # k
                bounds.append((v - 0.01, v + 0.01))  # r_eq
            elif term == "angle":  # Angle force constants and equilibrium angles
                bounds.append((max(50, v * 0.75), min(800, v * 1.25)))  # k_theta
                bounds.append((v - 0.05, v + 0.05))  # theta_eq
            elif term == "vdW":  # van der Waals parameters
                bounds.append((v - 0.25, v + 0.25))  # rmin
                bounds.append((max(0.05, v * 0.75), min(1.2, v * 1.25)))  # epsilon
            elif term == "charge":  # Atomic charges
                bounds.append((v - 0.1, v + 0.1))  # charge
            else:
                # General fallback: limit to +/-10% for other terms
                bounds.append((v - abs(v * 0.1), v + abs(v * 0.1)))

    return bounds


# Stepwise optimization loop
solutions = None
for data in [
    #(["bond", "angle", "proper"], 10),
    #(["proper", "vdW", "charge"], 10),
    (["bond", "angle", "proper", "vdW", "charge"], 100),
]:
    (terms, steps) = data

    bounds = generate_bounds(params, terms, params_opt)

    def obj_function(parameter_values):
        """
        Objective function for PSOGAOptimizer.

        Args:
            parameter_values: Array of parameter values to evaluate.

        Returns:
            Objective score (lower is better).
        """
        # Update the parameters in the force field
        updated_params = params.set_sub_parameters(parameter_values, terms, params_opt)
        params.update_ff_parameters(updated_params)

        # Reset the LAMMPS input if necessary
        lmp_in = params.ff.get_lammps_in()

        # Calculate the objective (e.g., MSE)
        objective_score = params.get_objective(
            ref_dics=ref_dics,
            e_offset=e_offset,
            lmp_in=lmp_in,
            obj="MSE"
        )

        # Add regularization to prevent overfitting and improve generalization
        #regularization = 0.01 * np.sum(np.square(parameter_values))  # L2 regularization
        #objective_score += regularization

        ## Introduce stochastic noise to avoid getting stuck in local minima
        #stochastic_noise = np.random.uniform(-0.01, 0.01)  # Small random perturbation
        #total_score += stochastic_noise

        ## Penalize unphysical parameter values (optional, depends on system constraints)
        #penalty = 0.0
        #for value in parameter_values:
        #    if value < 0 or value > 10:  # Example range for physical validity
        #        penalty += 100  # High penalty for out-of-bound parameters
        #objective_score += penalty

        #print(f"Objective Score: {objective_score}, Regularization: {regularization}, Total Score: {total_score}")

        return objective_score

    # PSO-GA optimization
    optimizer = PSOGAOptimizer(
        obj_function=obj_function,
        bounds=bounds,
        num_particles=50,
        dimensions=len(bounds),
        inertia=0.4,
        cognitive=0.1,
        social=0.9,
        mutation_rate=0.3,
        crossover_rate=0.5,
        max_iter=steps,
        verbose=True
    )

    best_position, best_score = optimizer.optimize(x0=solutions)

    # Update `params_opt` with the optimized values
    params_opt = params.set_sub_parameters(best_position, terms, params_opt)
    _, params_opt = params.optimize_offset(ref_dics, params_opt)
    solutions = optimizer.positions

    t = (time() - t0) / 60
    print(f"\nStepwise optimization for terms {terms} completed in {t:.2f} minutes.")
    print(f"Best Score: {best_score:.4f}")

# Final evaluation and saving results
errs = params.plot_ff_results("performance_opt_pso_102.png", ref_dics, [params_opt])
params.export_parameters("parameters_opt_pso.xml", params_opt, errs[0])
print("Optimization completed successfully.")
