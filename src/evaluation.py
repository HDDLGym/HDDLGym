''' This evaluation methods are for 
1. RL policies performance: deploying the learned policy in the environment and evaluating its performance through:
- Average reward
- Planning time
- Plan time steps
2. Training process:
- Training time
- Training convergence
3. Scalability:
- Sizes of states and actions spaces vs training time to convergence
- Number of agents vs training time and convergence
- Number of agents vs planning time
'''
import os
import time
import random
import copy
import numpy as np
from learning_methods import evaluate_policy, PPO_discrete, str2bool
from HDDL_files.problems_dir import PROBLEMS
from hddl_env import HDDLEnv

import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import torch

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))


def parse_arguments():
    '''Hyperparameter Setting'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dvc', type=str, default=None, help='running device: cuda or cpu')
    parser.add_argument('--problem-name', type=str, default='default', help=f'Provide the name of the problem, should be in one of: {PROBLEMS.keys()}')
    parser.add_argument('--domain', type=str, default=str(script_dir / "HDDL_files/Overcooked_specialization/overcooked_short_domain.hddl"), help='Which domain HDDL file to load?')
    parser.add_argument('--problem', type=str, default=str(script_dir / "HDDL_files/Overcooked_specialization/overcooked_short_prob2.hddl"), help='Which problem HDDL file to load?')
    parser.add_argument('--start', type=int, default=10000, help='Which episode to start from?')
    parser.add_argument('--end', type=int, default=260000, help='Which episode to end at?')
    parser.add_argument('--step', type=int, default=10000, help='Step size between two models (default = 1000)')
    parser.add_argument('--turns', type=int, default=100, help='Number of episodes to run for evaluation (default = 100)')
    parser.add_argument('--use-central-planner', type=str2bool, default=False,help='Whether to run centralized planner for multi-agent planning, default to False')
    parser.add_argument('--debug', type=str2bool, default=False,help='Debug mode or Not')
    parser.add_argument('--planner-time-limit', type=int, default=5, help='The time limit (in seconds) for running the planner for each agent at each step')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--T-horizon', type=int, default=2048, help='lenth of long trajectory')
    parser.add_argument('--max-e-steps', type = int, default=50, help="Max number of steps in each episode")
    parser.add_argument('--net-width', type=int, default=64, help='Hidden net width')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')

    opt = parser.parse_args()
    if opt.dvc is None:
        opt.dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.dvc = torch.device(opt.dvc) # from str to torch.device
    return opt


def training_evolution(start, end, opt, step=1000, turns=100, deterministic = False):
    '''Regenerate the evaluation score of models:
    inputs: 
    -model_folder_dir: the directory of the model folder
    -start: the start episode of the model
    -end: the end episode of the model
    -step: the step of between two models (default = 1000), which should be the same as opt.eval_interval
    -turn: number of episode to run the evaluation for each model (default = 100)
    outputs:
    - a list of average reward, planning time, and plan time steps
    '''

    score_list = []
    plan_time_list = []
    success_rate_list = []
    success_avg_step_list = []
    for i in range(start, end, step):
        opt.Loadmodel = True
        opt.Model = i
        print("Evaluating model at episode {}".format(i))
        eval_env = HDDLEnv(opt.domain, opt.problem)
        eval_env.action_space.seed(opt.seed)
        eval_env.observation_space.seed(opt.seed)
        eval_env._max_episode_steps = opt.max_e_steps
        opt.state_dim = eval_env.observation_space.n
        opt.action_dim = eval_env.action_space.n
        print("State space size: ", opt.state_dim)
        print("Action space size: ", opt.action_dim)
        
        # Load the trained policy
        policy = PPO_discrete(**vars(opt)).to(opt.dvc)
        # policy.load(opt.Model)  # Load the model checkpoint based on the `opt.Model`
        # policy.load(opt.Model, map_location=opt.dvc)
        # random_policy = copy.deepcopy(policy)
        if opt.Model != -1 and opt.Model is not None:
            # Load the model checkpoint based on the `opt.Model`
            try:
                policy.load(opt.Model)
            except:
                end = i
                break

        else:
            print("No model to load, using random policy")

        # Evaluate the policy
        score, plan_time, success_rate, success_avg_step = evaluate_policy(eval_env, policy, turns=turns, opt=opt, debug=False, deterministic=deterministic, return_plan_time=True, return_success_rate=True)  # Run the evaluation for 10 episodes
        score_list.append(score)
        plan_time_list.append(plan_time)
        success_rate_list.append(success_rate)  
        success_avg_step_list.append(success_avg_step)
    
    # Plot and save the evaluation results
    # Save data:
    fig_dir = script_dir / f'model/{opt.problem_name}'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    with open(str(fig_dir / 'eval_results.txt'), 'w') as f:
        f.write("Average Reward: {}\n".format(score_list))
        f.write("Average Planning Time: {}\n".format(plan_time_list))
        f.write("Success Rate: {}\n".format(success_rate_list))
        f.write("Average Steps to Success: {}\n".format(success_avg_step_list))
    # Plot scores:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns

    # First subplot: Average Reward
    axes[0, 0].plot(range(start, end, step), score_list, label='Average Reward')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Average Reward')

    # Second subplot: Average Planning Time
    axes[0, 1].plot(range(start, end, step), plan_time_list, label='Average Planning Time')
    axes[0, 1].set_ylabel('Average Planning Time')
    axes[0, 1].set_title('Average Planning Time')

    # Third subplot: Success Rate
    axes[1, 0].plot(range(start, end, step), success_rate_list, label='Success Rate')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_xlabel('Training Steps')
    axes[1, 0].set_title('Success Rate')

    # Fourth subplot: Average Steps to Success
    axes[1, 1].plot(range(start, end, step), success_avg_step_list, label='Average Steps to Success')
    axes[1, 1].set_ylabel('Average Steps to Success')
    axes[1, 1].set_xlabel('Training Steps')
    axes[1, 1].set_title('Average Steps to Success')

    # Adjust layout and add a main title
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust spacing to fit the title
    fig.suptitle('Evaluation Results', fontsize=16)
    fig.savefig(str(fig_dir / 'eval_results.png'))


def plot_evaluation_results(file_dir, file_name):
    # Read data from the file
    file_path = os.path.join(file_dir, file_name)
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse the data
    average_reward = eval(lines[0].split(":")[1].strip())
    average_planning_time = eval(lines[1].split(":")[1].strip())
    success_rate = eval(lines[2].split(":")[1].strip())
    average_steps_to_success = eval(lines[3].split(":")[1].strip())
    
    # Generate x-axis (training steps)
    training_steps = list(range(1, len(average_reward) + 1))
    
    # Create the first plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # First plot: Average Reward and Success Rate
    axes[0].plot(training_steps, average_reward, label='Reward', color='blue', linestyle='--')
    axes[0].set_ylabel('Reward', color='blue', fontsize=16)
    axes[0].tick_params(axis='y', labelcolor='blue')
    axes[0].set_xlabel('Training Steps (x10,000)', fontsize=20)
    axes[0].grid(True)
    
    ax2 = axes[0].twinx()  # Create a secondary y-axis
    ax2.plot(training_steps, success_rate, label='Success Rate', color='green')
    ax2.set_ylabel('Success Rate (%)', color='green', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='green')
    # axes[0].set_title('Average Reward and Success Rate')
    # axes[0].set_title('Analyzing Evolution of RL Policy During Training', fontsize=20)

    # Add legend for the first plot
    axes[0].legend(loc='lower left', fontsize=16)  # Legend for Average Reward
    ax2.legend(loc='lower right',fontsize=16)    # Legend for Success Rate
    
    # Second plot: Average Planning Time and Average Steps to Success
    axes[1].plot(training_steps, average_planning_time, label='Planning Time', color='black')
    axes[1].set_ylabel('Planning Time (s)', color='black', fontsize=16)
    axes[1].tick_params(axis='y', labelcolor='black')
    axes[1].set_xlabel('Training Steps (x10,000)', fontsize=20)
    axes[1].grid(True)
    
    ax3 = axes[1].twinx()  # Create a secondary y-axis
    ax3.plot(training_steps, average_steps_to_success, label='Average Steps to Success', color='red', linestyle='-.')
    ax3.set_ylabel('Average Steps to Success', color='red', fontsize=20)
    ax3.tick_params(axis='y', labelcolor='red')

    # Add legend for the second plot
    axes[1].legend(loc='upper left',fontsize=16)  # Legend for Average Planning Time
    ax3.legend(loc='upper right',fontsize=16)    # Legend for Average Steps to Success
    
    
    # Adjust layout and show the plots
    plt.tight_layout()
    fig_file = os.path.join(file_dir, 'eval_2plots.png')
    fig.savefig(fig_file)
    # plt.show()

# Example usage
plot_evaluation_results('./model/transport_1agent_no_collab','eval_results.txt')
    


if __name__ == "__main__":
    opt = parse_arguments()
    start = input("Enter the start episode (default is 10000): ")
    if start == '':
        start = opt.start
    else:
        start = int(start)
    end = input("Enter the end episode (default is 260000): ")
    if end == '':
        end = opt.end
    else:
        end = int(end)
    step = input("Enter the step size (default is 10000): ")
    if step == '':
        step = opt.step
    else:
        step = int(step)
    turns = input("Enter the number of episodes to run for evaluation (default is 100): ")
    if turns == '' and opt.turns is not None:
        turns = opt.turns
    else:
        turns = int(turns)
    

    if opt.problem_name != 'default':
        if opt.problem_name not in PROBLEMS.keys():
            raise ValueError(f"Problem name '{opt.problem_name}' not found in predefined problems. Here is the list of predifined problems: {PROBLEMS.keys()}")
        opt.domain, opt.problem = PROBLEMS[opt.problem_name]
    else:
        print(f"Using default or manual directions for domain and problem files: \n- DOMAIN: {opt.domain} \n- PROBLEM: {opt.problem}")
    training_evolution(start, end, opt, step=step, turns=turns, deterministic = False)


# python evaluation.py --problem-name transport_2agents_heterogeneous_with_noop --start 10000 --end 760000 --use-central-planner True --step 10000 --turns 50