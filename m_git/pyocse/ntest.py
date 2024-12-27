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
ref_dics = params.load_references("dataset/references.xml")[:120]

os.makedirs("ASP2", exist_ok=True)
os.chdir("ASP2")

t0 = time()
e_offset, params_opt = params.optimize_offset(ref_dics, p0)
params.update_ff_parameters(params_opt)
print(params.get_objective(ref_dics, e_offset, obj="MSE"))


def generate_bounds(params, terms, params_opt):
    """
    Generate realistic bounds dynamically for the selected terms in the force field parameters.

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
                bounds.append((v - abs(v * 0.15), v + abs(v * 0.15)))

    return bounds


# Stepwise optimization loop
for data in [
    (["bond", "angle", "proper"], 100),
    (["proper", "vdW", "charge"], 100),
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
        try:
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
            regularization = 0.01 * np.sum(np.square(parameter_values))  # L2 regularization
            total_score = objective_score + regularization

            # Introduce stochastic noise to avoid getting stuck in local minima
            stochastic_noise = np.random.uniform(-0.01, 0.01)  # Small random perturbation
            total_score += stochastic_noise

            # Penalize unphysical parameter values (optional, depends on system constraints)
            penalty = 0.0
            for value in parameter_values:
                if value < 0 or value > 10:  # Example range for physical validity
                    penalty += 100  # High penalty for out-of-bound parameters
            total_score += penalty

            #print(f"Objective Score: {objective_score}, Regularization: {regularization}, Total Score: {total_score}")

            return total_score
        except Exception as e:
            print(f"Error evaluating parameters: {e}")
            return np.inf

    # PSO-GA optimization
    optimizer = PSOGAOptimizer(
        obj_function=obj_function,
        bounds=bounds,
        num_particles=102,
        dimensions=len(bounds),
        inertia=0.4,
        cognitive=0.1,
        social=0.9,
        mutation_rate=0.3,
        crossover_rate=0.5,
        max_iter=steps,
        verbose=True
    )

    best_position, best_score = optimizer.optimize()

    # Update `params_opt` with the optimized values
    params_opt = params.set_sub_parameters(best_position, terms, params_opt)
    _, params_opt = params.optimize_offset(ref_dics, params_opt)

    t = (time() - t0) / 60
    print(f"\nStepwise optimization for terms {terms} completed in {t:.2f} minutes.")
    print(f"Best Score: {best_score:.4f}")


def _plot_ff_results(self, axes, parameters, ref_dics, label, max_E=1000, max_dE=1000, size=None):
    """
    Plot the results of FF prediction as compared to the references in
    terms of Energy, Force, and Stress values.
    """
    # Set up the ff engine
    self.update_ff_parameters(parameters)

    results = self.evaluate_multi_references(ref_dics, parameters, max_E, max_dE)
    (ff_values, ref_values, rmse_values, r2_values) = results
    (ff_eng, ff_force, ff_stress) = ff_values
    (ref_eng, ref_force, ref_stress) = ref_values
    (mse_eng, mse_for, mse_str) = rmse_values
    (r2_eng, r2_for, r2_str) = r2_values

    if len(ff_eng) == 0 or len(ref_eng) == 0:
        print("Error: Empty energy arrays. Optimization may have failed.")
        return axes, {"rmse_values": (None, None, None), "r2_values": (None, None, None)}

    print('\nMin_values: {:.4f} {:.4f}'.format(ff_eng.min(), ref_eng.min()))

    label1 = '{:s}. Energy ({:d})\n'.format(label, len(ff_eng))
    label1 += 'Unit: [eV/mole]\n'
    label1 += 'RMSE: {:.4f}\n'.format(mse_eng)
    label1 += 'R2:   {:.4f}'.format(r2_eng)

    label2 = '{:s}. Forces ({:d})\n'.format(label, len(ff_force))
    label2 += 'Unit: [eV/A]\n'
    label2 += 'RMSE: {:.4f}\n'.format(mse_for)
    label2 += 'R2:   {:.4f}'.format(r2_for)

    label3 = '{:s}. Stress ({:d})\n'.format(label, len(ff_stress))
    label3 += 'Unit: [GPa]\n'
    label3 += 'RMSE: {:.4f}\n'.format(mse_str)
    label3 += 'R2:   {:.4f}'.format(r2_str)

    axes[0].scatter(ref_eng, ff_eng, s=size, label=label1)
    axes[1].scatter(ref_force, ff_force, s=size, label=label2)
    axes[2].scatter(ref_stress, ff_stress, s=size, label=label3)

    for ax in axes:
        ax.set_xlabel('Reference')
        ax.set_ylabel('FF')
        ax.legend(loc=2)

    err_dict = {
        'rmse_values': (mse_eng, mse_for, mse_str),
        'r2_values': (r2_eng, r2_for, r2_str),
        'min_values': (ff_eng.min(), ref_eng.min()),
    }
    return axes, err_dict


# Final evaluation and saving results
errs = params.plot_ff_results("performance_opt_pso_102.png", ref_dics, [params_opt])
params.export_parameters("parameters_opt_pso.xml", params_opt, errs[0])
print("Optimization completed successfully.")
