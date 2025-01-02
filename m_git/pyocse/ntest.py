from psoga_optimizer import PSOGAOptimizer
from pyocse.parameters import ForceFieldParameters
from time import time
import os
import numpy as np

def obj_function(para_values, params, para0, terms, ref_dics, e_offset):
    """
    Objective function for PSOGAOptimizer.

    Args:
        para_values: 1D-Array of parameter values to evaluate.
        params: parameter instance
        para0: Array of all FF parameter as the base
        terms: list of FF terms to optimize
        ref_dics: dictionary of dataset
        e_offset: offset value

    Returns:
        Objective score (lower is better).
    """

    # Split 1D array of para_values to a list grouped by each term
    sub_values = []
    count = 0
    for term in terms:
        N = getattr(params, 'N_'+term)
        sub_values.append(para_values[count:count+N])
        count += N

    #print("debug subvals", sub_values[0][:5], para0[:5])
    updated_params = params.set_sub_parameters(sub_values, terms, para0)

    # Update the parameters in the force field with the base parameter
    params.update_ff_parameters(updated_params)

    # Reset the LAMMPS input if necessary
    lmp_in = params.ff.get_lammps_in()

    # Calculate the objective (e.g., MSE)
    objective_score = params.get_objective(
        ref_dics=ref_dics,
        e_offset=e_offset,
        lmp_in=lmp_in,
        #obj="MSE"
        obj="R2"
    )

    return objective_score


np.random.seed(7)

params = ForceFieldParameters(
    smiles=['CC(=O)OC1=CC=CC=C1C(=O)O'],
    f_coef=1, #0.1,
    s_coef=1, #0,
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
errs = params.plot_ff_results("performance_opt_pso_0.png", ref_dics, [params_opt])
print("MSE objective", params.get_objective(ref_dics, e_offset, obj="MSE"))
print("R2 objective", params.get_objective(ref_dics, e_offset, obj="R2"))
#import sys; sys.exit()

# Stepwise optimization loop
for data in [
    (["bond", "angle", "proper", "vdW"], 10), #00),
]:
    (terms, steps) = data

    sub_vals, sub_bounds, _ = params.get_sub_parameters(params_opt, terms)
    vals = np.concatenate(sub_vals)
    bounds = np.concatenate(sub_bounds)

    # PSO-GA optimization
    optimizer = PSOGAOptimizer(
        obj_function=obj_function,
        obj_args=(params, params_opt, terms, ref_dics, e_offset),
        bounds=bounds,
        seed=vals.reshape((1, len(vals))),
        num_particles=10, #0,
        dimensions=len(bounds),
        inertia=0.5,
        cognitive=0.2,
        social=0.8,
        mutation_rate=0.3,
        crossover_rate=0.5,
        max_iter=steps,
        verbose=True
    )

    best_position, best_score = optimizer.optimize()

    # Update `params_opt` with the optimized values
    # Split 1D array of para_values to a list grouped by each term
    sub_values = []
    count = 0
    for term in terms:
        N = getattr(params, 'N_'+term)
        sub_values.append(best_position[count: count+N])
        count += N

    params_opt = params.set_sub_parameters(sub_values, terms, params_opt)
    e_offset, params_opt = params.optimize_offset(ref_dics, params_opt)
    params.update_ff_parameters(params_opt)
    print("e_offset", e_offset)

    t = (time() - t0) / 60
    print(f"\nStepwise optimization for terms {terms} completed in {t:.2f} minutes.")
    print(f"Best Score: {best_score:.4f}")

# Final evaluation and saving results
errs = params.plot_ff_results("performance_opt_pso_1.png", ref_dics, [params_opt])
params.export_parameters("parameters_opt_pso.xml", params_opt, errs[0])
print("Optimization completed successfully.")
