# MODIFY if NEEDED: Change learning_methods to learning_methods file that you design:
from learning_methods import PPO_discrete, evaluate_policy, str2bool, get_observation_one_hot_vector, enumerate_action, enumerate_state

from hddl_utils import enumerate_list
from hddl_env import HDDLEnv
from central_planner import centralized_planner
from utils import run_planner_and_get_action_dict, get_logprob, parallelly_run_planner_and_get_action_dict

import copy
import random
import time
import json
import numpy as np
from datetime import datetime
import gymnasium as gym
import os, shutil
import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from HDDL_files.problems_dir import PROBLEMS


script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    '''Hyperparameter Setting'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dvc', type=str, default=None, help='running device: cuda or cpu')
    parser.add_argument('--problem-name', type=str, required=True, help=f'Provide the name of the problem, should be in one of: {PROBLEMS.keys()}')
    parser.add_argument('--domain', type=str, default=str(script_dir / "HDDL_files/Custom_environments/Overcooked_specialization/overcooked_short_domain.hddl"), help='Which domain HDDL file to load?')
    parser.add_argument('--problem', type=str, default=str(script_dir / "HDDL_files/Custom_environments/Overcooked_specialization/overcooked_short_prob2.hddl"), help='Which problem HDDL file to load?')
    parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
    # Indicate if run training or evaluation:
    parser.add_argument('--run-training', type=str2bool, default=True, help='Train or evaluate the model, True for training, False for evaluation')
    # Planner parameters:
    parser.add_argument('--use-central-planner', type=str2bool, default=False,help='Whether to run centralized planner for multi-agent planning, default to False')
    parser.add_argument('--debug', type=str2bool, default=False,help='Debug mode or Not')
    parser.add_argument('--planner-time-limit', type=int, default=5, help='The time limit (in seconds) for running the planner for each agent at each step')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--T-horizon', type=int, default=2048, help='lenth of long trajectory')
    parser.add_argument('--max-e-steps', type = int, default=50, help="Max number of steps in each episode")
    
    # Training parameters
    parser.add_argument('--monitor-training', type=str2bool, default=False, help='Monitor training or not')
    parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
    parser.add_argument('--Model', type=int, default=None, help='which model to load')
    parser.add_argument('--Max-train-steps', type=int, default=1e8, help='Max training steps')
    parser.add_argument('--save-interval', type=int, default=1e3, help='Model saving interval, in steps.')
    parser.add_argument('--eval-interval', type=int, default=1e3, help='Model evaluating interval, in steps.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
    parser.add_argument('--clip-rate', type=float, default=0.2, help='Clip rate')
    parser.add_argument('--K-epochs', type=int, default=10, help='update times - number of epochs of training')
    parser.add_argument('--net-width', type=int, default=64, help='Hidden net width')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--l2-reg', type=float, default=0, help='L2 regulization coefficient for Critic (PPO)')
    parser.add_argument('--batch-size', type=int, default=2048, help='lenth of sliced trajectory')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient of Actor')
    parser.add_argument('--entropy-coef-decay', type=float, default=0.99, help='Decay rate of entropy_coef')
    parser.add_argument('--adv-normalization', type=str2bool, default=False, help='Advantage normalization')
    parser.add_argument('--exploration-decay', type=float, default=0.999, help="decay rate of exploration rate")
    parser.add_argument('--activation', type=str, default='tanh',help='activation function for learning model')
    parser.add_argument('--convergence-threshold', type=float, default=1e-3, help='Convergence threshold for training')
    
    
    opt = parser.parse_args()
    if opt.problem_name != 'ALL':
        if opt.problem_name not in PROBLEMS.keys():
            raise ValueError(f"Problem name '{opt.problem_name}' not found in predefined problems. Here is the list of predifined problems: {PROBLEMS.keys()}")
        opt.domain, opt.problem = PROBLEMS[opt.problem_name]
    else:
        print(f"Using default or manual directions for domain and problem files: \n- DOMAIN: {opt.domain} \n- PROBLEM: {opt.problem}")

    if opt.dvc is None:
        opt.dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.dvc = torch.device(opt.dvc) # from str to torch.device
    #     
    print(opt)
    return opt



def main_train(opt, debug=False, monitor_training=False):
    # Build Training Env and Evaluation Env
    env = HDDLEnv(opt.domain, opt.problem)
    env._max_episode_steps = opt.max_e_steps
    eval_env = HDDLEnv(opt.domain, opt.problem)
    eval_env._max_episode_steps = opt.max_e_steps
    opt.state_dim = env.observation_space.n
    print('opt.state_dim:', opt.state_dim)
    opt.action_dim = env.action_space.n
    print("opt.action_dim:",opt.action_dim)
    monitor_training = opt.monitor_training
    actor_loss_list = []
    critic_loss_list = []
    deterministic_eval_score = []
    nondeterministic_eval_score = []
    success_step = 0
    success_ep_count = 0
    exploration_rate = 1.0 # start training with full exploration
    exploration_min = 0.1 # minimum exploration

    # Seed Everything
    env_seed = opt.seed
    # torch.manual_seed(opt.seed)
    # torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Env:',opt.problem_name ,'  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,'   Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps)
    print('\n')
    #
    # Use tensorboard to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        # writepath = 'runs/{}'.format(BriefEnvName[opt.EnvIdex]) + timenow
        writepath = '{}'.format(opt.problem.split('.hddl')[0]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    if not os.path.exists('model'): os.mkdir('model')
    policy = PPO_discrete(**vars(opt)).to(opt.dvc)
    # print('policy:',policy)
    if opt.Loadmodel: policy.load(opt.Model)
    else: opt.Model = 0

    # if True: #else:
    traj_lenth, total_steps = 0, 0
    episode = 0
    converged = False
    while total_steps < opt.Max_train_steps and not converged:
        if episode % 30 == 0 and monitor_training:
            print("\n\n*** Episode {} ***\n".format(episode), end='\r', flush=True)
            print("-- total train steps:",total_steps)
            print("Average step of successfull episodes:",success_step)
            print("number of successful ep", success_ep_count)
        # s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
        policy_list = [{"all operators":policy}] * len(env.agents)
        env.reset(policy_list = policy_list)
        env_seed += 1
        random.seed(env_seed)
        done = False
        s = env.current_state
        s_num = enumerate_state(env, opt)
        # s_num = get_observation_one_hot_vector(env.current_state, env.env_dictionary['htn goal'].tasks, env.env_dictionary, device=opt.dvc).clone().detach()
        # print dynamic success episode count:
        print(f"Successful episode count: {success_ep_count} / {episode} episodes", end='\r', flush=True)
        episode +=1
        episode_reward = 0
        episode_data = []
        step = 0
        ep_hierarchy_record = []
        exploration_rate = max(exploration_min, exploration_rate * opt.exploration_decay)
        prev_train_loss = None
        
        '''Interact & train'''
        while not done:
            # print("step: ",step)
            # Exploration determines whether to set deterministic to False or True:
            if np.random.rand() < exploration_rate:
              deterministic = False # Explore with policy as a reference
            else:
              deterministic = True # Exploit 
            # Run the planner and get action dictionary:
            # start_time = time.time()
            # action_dict_parallel, env_parallel, hierarchies_parallel = parallelly_run_planner_and_get_action_dict(policy_list, env, opt, deterministic=False, debug=debug)
            # time_run_parallel = time.time() - start_time
            # print("time_run_parallel:",time_run_parallel)
            start_no_parallel_time = time.time()
            action_dict, env, hierarchies = run_planner_and_get_action_dict(policy_list, env, opt, deterministic=deterministic, debug=debug)
            time_run_no_parallel = time.time() - start_no_parallel_time

            # print("\n\nstep:",step,"\n time_run_parallel:",time_run_parallel,"\n time_run_no_parallel:",time_run_no_parallel)
            ep_hierarchy_record.append(hierarchies)

            if debug:
              print('\n\n>> Step {} has action dict: {}\n'.format(step ,action_dict))
              for agent_id, agent_debug in enumerate(env.agents):
                print('Agent {} has hierarchy: {}'.format(agent_debug.name,'\n==> '.join(agent_debug.task_method_hierarchy)))

            # enumerate action
            a_num = enumerate_action(env,opt)

            # Step thru environment
            s_next, r, dw, tr, completed_goal_tasks_list = env.step(action_dict) # dw: dead&win; tr: truncated
            step+=1
            done = (dw or tr)
            # Get one-hot version of new state
            s_next_num = enumerate_state(env, opt)
            logprob_a = get_logprob(s_num, policy, env, opt)
            # print('logprob_a:',logprob_a)

            # Update agents' hierarchies after step thru env:
            for agent in env.agents:
              agent.update_agent_hierarchy_by_checking_with_world_state(env.current_state, env.env_dictionary)

            # episode_data.append([s_num, a_num, r, s_next_num, torch.exp(logprob_a).item(), done, dw, traj_lenth])
            if done and len(completed_goal_tasks_list) == len(env.env_dictionary['htn goal'].tasks): #goal reached
              # print("ep {} completed {} after {} steps\n".format(episode,completed_goal_tasks_list,step))
              success_step = (success_step*success_ep_count + step+1)/(success_ep_count+1)
              success_ep_count +=1
              # print("Record for successful ep: ")
              # for hierarchies_ in ep_hierarchy_record:
              #   print(hierarchies_)
              #   print('--')
              # for data in episode_data:
              #   policy.put_data(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])

            if done and len(completed_goal_tasks_list)>0 and debug:
              print("completed goal tasks: ", completed_goal_tasks_list)
              

            '''Store the current transition'''
            policy.put_data(s_num, a_num, r, s_next_num, torch.exp(logprob_a).item(), done, dw, idx = traj_lenth)
            
            traj_lenth += 1
            total_steps += 1
            s = s_next
            s_num = s_next_num
            episode_reward += r
            if done and debug:
                print("Episode reward of episode {} is {}".format(episode, episode_reward))
            
            '''Update if its time'''
            if traj_lenth % opt.T_horizon == 0 and traj_lenth>0:
            # if traj_lenth > opt.T_horizon - opt.max_e_steps:
                if monitor_training:
                    print("TRAINING policy at traj_lenth ", traj_lenth)
                a_loss, c_loss = policy.train()
                # take average of the current_train_loss, which is a list of epoch losses:
                if len(a_loss) > 0:
                    current_actor_train_loss = sum(a_loss)/len(a_loss)
                    current_critic_train_loss = sum(c_loss)/len(c_loss)
                    current_train_loss = np.array([current_actor_train_loss, current_critic_train_loss])
                else:
                   current_train_loss = None
                if prev_train_loss is not None and current_train_loss is not None and len(prev_train_loss) == len(current_train_loss):
                    if np.all(np.abs(current_train_loss - prev_train_loss) < opt.convergence_threshold):
                        print("Training converged at traj_lenth ", traj_lenth)
                        converged = True
                        break
                elif current_actor_train_loss is not None:
                   prev_train_loss = copy.copy(current_train_loss)
                        
      
                actor_loss_list += a_loss
                critic_loss_list += c_loss
                traj_lenth = 0

            if traj_lenth % opt.T_horizon*20 == 0:
                fig = plt.figure()
                # First subplot for Actor loss:
                plt.subplot(211)
                plt.plot(actor_loss_list,'b')
                plt.ylabel("Actor Loss")
                # Second subplot for Critic loss:
                plt.subplot(212)
                plt.plot(critic_loss_list, 'r')
                plt.xlabel("Epochs")
                plt.ylabel("Critic Loss")

                plt.suptitle("Actor Loss and Critic Loss")
                plt.close(fig)
                fig_dir = script_dir / f'model/{opt.problem_name}'
                if not os.path.exists(fig_dir):
                    os.makedirs(fig_dir)
                fig.savefig(str(fig_dir / 'loss_plot.png'))

            '''Record & log'''
            if total_steps % opt.eval_interval == 0 and total_steps>0:
                deterministic_score = evaluate_policy(eval_env, policy, turns=1, opt=opt, deterministic=True) # evaluate the policy for 'turns' times, and get averaged result
                deterministic_eval_score.append(deterministic_score)
                nondeterministic_score = evaluate_policy(eval_env, policy, turns=10, opt=opt, deterministic=False) # evaluate the policy for 'turns' times, and get averaged result
                nondeterministic_eval_score.append(nondeterministic_score)
                if opt.write: writer.add_scalar('ep_r', deterministic_score, global_step=total_steps)
                if monitor_training:
                  print('Problem name:',opt.problem_name,'seed:',env_seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)
                if deterministic_score > 0:
                  # Save model in another folder if the score is greater than 0:
                  model_dir = script_dir / f'model/{opt.problem_name}'
                  if not os.path.exists(model_dir):
                      os.makedirs(model_dir)                     
                  policy.save(str(total_steps + opt.Model) + '_working')
                
                if total_steps % opt.eval_interval*10 == 0:
                  fig = plt.figure()
                  # First subplot for Deterministic score:
                  plt.subplot(211)
                  plt.plot(deterministic_eval_score, "b")
                  plt.ylabel("Deterministic Score \n(Cum. Disc. Reward)")

                  # Second subplot for Nondeterministic score:
                  plt.subplot(212)
                  plt.plot(nondeterministic_eval_score, "r")
                  plt.ylabel("Nondeterministic Score \n(Cum. Disc. Reward)")
                  plt.xlabel("x {} steps".format(opt.eval_interval))
                  plt.suptitle("Evaluate RL models")
                  plt.close(fig)
                  fig_eval_dir = script_dir / f'model/{opt.problem_name}'
                  if not os.path.exists(fig_eval_dir):
                    os.makedirs(fig_eval_dir)
                  fig.savefig(str(fig_eval_dir / 'evaluate_policy_plot.png'))

            '''Save model'''
            if total_steps % opt.save_interval==0:
                model_dir = script_dir / f'model/{opt.problem_name}'
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)                     
                policy.save(total_steps + opt.Model)
                # Save data:
                data = {
                    'actor_loss': actor_loss_list,
                    'critic_loss': critic_loss_list,
                    'deterministic_score': deterministic_eval_score,
                    'nondeterministic_score': nondeterministic_eval_score
                }
                with open(model_dir / 'data.json', 'w') as f:
                    json.dump(data, f)

        # End of episode
    env.close()
    eval_env.close()


def evaluate_model(opt, turns=100, return_results=False, compare_with_random=True):
    # opt.domain = str(script_dir / "overcooked_short_domain.hddl")
    # opt.problem = str(script_dir / "overcooked_short_prob2.hddl")
    eval_env = HDDLEnv(opt.domain, opt.problem)
    eval_env.action_space.seed(opt.seed)
    eval_env.observation_space.seed(opt.seed)
    eval_env._max_episode_steps = opt.max_e_steps
    opt.state_dim = eval_env.observation_space.n
    opt.action_dim = eval_env.action_space.n
    print('*** Env:',opt.problem_name ,'\n  state_dim:',opt.state_dim,\
          ' \n action_dim:',opt.action_dim,' \n  Random Seed:',opt.seed, \
            ' \n max_e_steps:',opt.max_e_steps)
    
    # Load the trained policy
    policy = PPO_discrete(**vars(opt)).to(opt.dvc)
    # policy.load(opt.Model)  # Load the model checkpoint based on the `opt.Model`
    # policy.load(opt.Model, map_location=opt.dvc)
    random_policy = copy.deepcopy(policy)
    if opt.Model != -1 and opt.Model is not None:
        # Load the model checkpoint based on the `opt.Model`
        policy.load(opt.Model)
        random_only = False
    else:
        print("No model to load, using random policy")
        random_only = True

    # Evaluate the policy
    score, plan_time, success_rate, success_plan_step  = evaluate_policy(eval_env, policy, turns=turns, opt=opt, debug=False, deterministic=False, return_plan_time=True, return_success_rate=True)  # Run the evaluation for 10 episodes
    # Random policy:
    if not random_only and compare_with_random:
        score_random, plan_time_random, success_rate_random, success_plan_step_random = evaluate_policy(eval_env, random_policy, turns=10, opt=opt, debug=False, deterministic=False, return_plan_time=True, return_success_rate=True)  # Run the evaluation for 10 episodes
        print("\n\n")
        print(f"|--------------------------------|--------------------|-------------------|")
        print(f"|             Metrics            |   Trained Policy   |   Random Policy   |")
        print(f"|--------------------------------|--------------------|-------------------|")
        print(f"|         Score each turn        | {score:^18.3f} | {score_random:^18.3f}|")
        print(f"|  Planning time each turn (sec) | {plan_time:^18.3f} | {plan_time_random:^18.3f}|")
        print(f"|  Success rate each turn (%)     | {success_rate*100:^18.3f} | {success_rate_random*100:^18.3f}|")
        print(f"|  Success plan step each turn    | {success_plan_step:^18.3f} | {success_plan_step_random:^18.3f}|")
        print(f"|--------------------------------|--------------------|-------------------|")

    eval_env.close()
    if return_results:
        return score, plan_time, success_rate, success_plan_step

def run_hddlgym_without_policy(opt, turns = 10, return_results=False, debug=False):
    '''
    Test HDDLGym without policy (random policy)
    input:
    - opt: parameters
    output: (void)
    '''
    env = HDDLEnv(opt.domain, opt.problem)
    opt.state_dim = env.observation_space.n
    opt.action_dim = env.action_space.n
    policy = None
    results = evaluate_policy(env, policy,turns = turns, opt=opt, debug=debug, deterministic = False, return_plan_time=True, return_success_rate=True)  # Run the evaluation for 10 episodes
    if return_results:
        return results

opt = parse_arguments()
# opt.domain = str(script_dir / "HDDL_files/Overcooked_specialization/overcooked_short_domain.hddl")
# opt.problem = str(script_dir / "HDDL_files/Overcooked_specialization/overcooked_short_prob2.hddl")

### Call main_train:
if __name__ == "__main__":
    opt = parse_arguments()
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    # Reset domain and problem if problem_name is provided correctly
    if opt.problem_name != 'default':
        if opt.problem_name not in PROBLEMS.keys():
            raise ValueError(f"Problem name '{opt.problem_name}' not found in predefined problems. Here is the list of predifined problems: {PROBLEMS.keys()}")
        opt.domain, opt.problem = PROBLEMS[opt.problem_name]
    else:
        print(f"Using default or manual directions for domain and problem files: \n- DOMAIN: {opt.domain} \n- PROBLEM: {opt.problem}")
    # Run training or evaluation based on the Loadmodel and Model parameters:
    if opt.run_training:
        main_train(opt, debug=opt.debug)
    elif not opt.Loadmodel or opt.Model is None:
        user_input = input("Loadmodel flag is set to False or no input for model. Do you want to run HDDLGym without policy? (y/n): ")
        if user_input.lower() == 'y':
            run_hddlgym_without_policy(opt)
        else:
            user_input2 = input("Do you want to run HDDLGym with a policy? If yes, input your model ID (int), else, type 'n' to exit: \n")
            if user_input2.lower() != 'n' and user_input2.isdigit():
                opt.Loadmodel = True
                opt.Model = int(user_input2)
                results = evaluate_model(opt, return_results=True)
                print(f"Evaluation results: {results}")
            else:
                print("Exiting the program.")
                exit() 
    elif opt.Loadmodel and opt.Model is not None:
        # Load the model and evaluate it
        results = evaluate_model(opt, return_results=True, compare_with_random=False)
        print(f"Evaluation results: {results}")  


# command line:
# python main_train.py --problem-name overcooked_2agents_collab --run-training True --use-central-planner True --max-e-steps 25