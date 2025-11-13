# This file contains the code to run the pretraining analysis, including:
# 1. Complexity of the environment (number of lifted/grounded operators, number of grounded predicates)
# 2. Difficulty of the environment (success rate, plan time, and plan steps of hierarchical planning with a random policy)

import main_train
from hddl_env import HDDLEnv
from HDDL_files.problems_dir import PROBLEMS
from learning_methods import PPO_discrete
import os
import json

opt = main_train.parse_arguments()

def get_env_complexity(opt, run_all_problems=False, debug=True):
    data = {}
    if not run_all_problems:
        env = HDDLEnv(opt.domain, opt.problem)
        results = {}
        for key in env.env_dictionary.keys():
            #print len of each value:
            if not isinstance(env.env_dictionary[key], (list, set)):
                continue
            if debug: print(f"len of {key}: {len(env.env_dictionary[key])}")
            results[f"length of {key}"] = len(env.env_dictionary[key])
        data[opt.problem_name] = results 
    else:
        for prob in PROBLEMS.keys():
            results = {}
            if debug: print(f"Complexity of problem {prob} is \n")
            for key in env.env_dictionary.keys():
                #print len of each value:
                if debug: print(f"len of {key}: {len(env.env_dictionary[key])}")
                results[f"length of {key}"] = len(env.env_dictionary[key])
            data[prob] = results 

    # Save data to json file pretrain_analysis_complexity_result.json
    with open("pretrain_analysis_complexity_result.json", "w") as file:
        json.dump(data, file, indent=4)

    return data




def get_env_difficulty(opt, run_all_problems=False, debug=False):
    problem_result_dict = {}
    opt.Loadmodel = False
    opt.Model = None
    opt.use_central_planner = True
    opt.max_e_steps = 100
    if not run_all_problems:
        if debug: print("Domain, problem", opt.domain, opt.problem)
        results = main_train.run_hddlgym_without_policy(opt, turns=100, return_results=True, debug=False)
        problem_result_dict[opt.problem_name] = results
        if debug: print("Results - score, time, steps, success rate", results)
        # Save the current problem result dictionary to a file
        with open("pretrain_analysis_difficulty_result.json", "w") as file:
            json.dump(problem_result_dict, file, indent=4)
    else:
        for prob in PROBLEMS.keys():
            if debug: print("\nWorking with problem: ", prob)
            opt.problem_name = prob
            opt.domain, opt.problem = PROBLEMS[prob]
            if debug: print("Domain, problem", opt.domain, opt.problem)
            results = main_train.run_hddlgym_without_policy(opt, turns=100, return_results=True, debug=False)
            problem_result_dict[prob] = results
            if debug: print("Results - score, time, steps, success rate", results)
            # Save the current problem result dictionary to a file
            with open("pretrain_analysis_difficulty_result.json", "w") as file:
                json.dump(problem_result_dict, file, indent=4)

    return problem_result_dict

if __name__ == "__main__":
    opt = main_train.parse_arguments()
    run_all_problems = opt.problem_name == "ALL"
    data = get_env_complexity(opt)
    problem_result_dict = get_env_difficulty(opt)
    for problem, results in problem_result_dict.items():
        print(f"Problem: {problem}")
        print(f"Results - score, planning time, success rate, plan steps: {results}\n")
    # Save the dictionary to a file using json
    with open("pretrain_problem_result_dict.json", "w") as file:
        json.dump(problem_result_dict, file, indent=4)
    print("Dictionary saved to pretrain_problem_result_dict.json")
        

