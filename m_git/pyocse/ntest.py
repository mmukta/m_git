from psoga_optimizer import PSOGAOptimizer
from pyocse.parameters import ForceFieldParameters
from time import time
import os
import numpy as np
import multiprocessing as mp
import copy

# Global shared arguments for all workers
def worker_init(shared_params, shared_para0, shared_terms, shared_ref_dics, shared_e_offset, shared_obj):
    global params, para0, terms, ref_dics, e_offset, obj
    params = shared_params
    para0 = shared_para0
    terms = shared_terms
    ref_dics = shared_ref_dics
    e_offset = shared_e_offset
    obj = shared_obj

def worker(args):
    para_values, path = args
    # Get the current worker's process ID and name
    process = mp.current_process()  # Get process info
    worker_id = process.name        # Name of the worker process

    # Print worker ID along with the path
    #print(f'Worker {worker_id} is evaluating path: {path}')
    return obj_function(para_values, params, para0, terms, ref_dics, e_offset, obj, path)

def obj_function_par(para_values_list, params, para0, terms, ref_dics, e_offset, ncpu, obj="R2"):
    """
    Parallel evaluation of objective function for multiple sets of parameters.

    Args:
        para_values_list: List of 1D-arrays, each containing parameter values.
        params: Force field parameter instance (shared among all tasks).
        para0: Base parameters.
        terms: List of force field terms.
        ref_dics: Reference dataset dictionary.
        e_offset: Energy offset.
        obj: Objective metric (default is "R2").
        num_workers: Number of parallel processes.

    Returns:
        List of objective scores.
    """
    if ncpu == 1:
        scores = []
        for i, para_value in enumerate(para_values_list):
            print(f'evaluating: {i}')
            score = obj_function(para_value, params, para0, terms, ref_dics, e_offset, obj, '.')
            scores.append(score)
        return scores

    # Prepare input data
    input_data = [(para, f'tmp_{i}') for i, para in enumerate(para_values_list)]

    print(f"Parallel Mode {ncpu}")
    t0 = time()

    # Use multiprocessing with shared arguments
    with mp.Pool(
        processes=ncpu,
        initializer=worker_init,
        initargs=(params, para0, terms, ref_dics, e_offset, obj)
    ) as pool:
        results = pool.map(worker, input_data)

    print(f"Time for parallel computation: {time()-t0}")

    ## Use multiprocessing to parallelize computations
    #input_data = []
    ## timing copy
    #t0 = time()
    #for i in range(len(para_values_list)):
    #    local_params = copy.deepcopy(params)  # Or params.clone() if implemented
    #    para = para_values_list[i]
    #    input_data.append((para, local_params, para0, terms, ref_dics, e_offset, obj, f'tmp_{i}'))

    #print(f"Parallel Mode {ncpu}, time for copying variables: {time()-t0}")
    #with mp.Pool(processes=ncpu) as pool:
    #    results = pool.map(worker, input_data)

    return results

def obj_function(para_values, params, para0, terms, ref_dics, e_offset, obj, path):
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
        obj=obj,
        path=path,
    )

    return objective_score

if __name__ == "__main__":
    np.random.seed(7)
    
    params = ForceFieldParameters(
        smiles=['CC(=O)OC1=CC=CC=C1C(=O)O'],
        f_coef=1, #0.1,
        s_coef=1, #0,
        e_coef=1,
        style='openff',
        ref_evaluator='mace',
        ncpu=1,
    )
    
    p0, errors = params.load_parameters("dataset/parameters.xml")
    ref_dics = params.load_references("dataset/references.xml")[:200]
    
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
        (["bond", "angle", "proper", "vdW"], 15),
    ]:
        (terms, steps) = data
    
        sub_vals, sub_bounds, _ = params.get_sub_parameters(params_opt, terms)
        vals = np.concatenate(sub_vals)
        bounds = np.concatenate(sub_bounds)
    
        # PSO-GA optimization
        optimizer = PSOGAOptimizer(
            obj_function=obj_function_par,
            obj_args=(params, params_opt, terms, ref_dics, e_offset),
            bounds=bounds,
            seed=vals.reshape((1, len(vals))),
            num_particles=100, #0,
            dimensions=len(bounds),
            inertia=0.5,
            cognitive=0.2,
            social=0.8,
            mutation_rate=0.3,
            crossover_rate=0.5,
            max_iter=steps,
            verbose=True,
            ncpu=4,
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
