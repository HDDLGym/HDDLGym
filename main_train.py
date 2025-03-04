from hddl_utils import enumerate_list
from hddl_env import HDDLEnv
from learning_methods_lifted import PPO_discrete, evaluate_policy, str2bool, get_observation_one_hot_vector, enumerate_action, enumerate_state
from central_planner import centralized_planner
from utils import run_planner_and_get_action_dict, get_logprob
import copy
import random
import numpy as np
from datetime import datetime
import gymnasium as gym
import os, shutil
import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path


script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
  '''Hyperparameter Setting'''
  parser = argparse.ArgumentParser()
  parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
  # parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
  parser.add_argument('--domain', type=str, default=str(script_dir / "HDDL_files/Overcooked_specialization/overcooked_short_domain.hddl"), help='Which domain HDDL file to load?')
  parser.add_argument('--problem', type=str, default=str(script_dir / "HDDL_files/Overcooked_specialization/overcooked_short_prob2.hddl"), help='Which problem HDDL file to load?')
  parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
  # parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
  parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
  parser.add_argument('--Model', type=int, default=2, help='which model to load')

  parser.add_argument('--seed', type=int, default=209, help='random seed')
  parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
  parser.add_argument('--Max_train_steps', type=int, default=5e7, help='Max training steps')
  parser.add_argument('--save_interval', type=int, default=1e3, help='Model saving interval, in steps.')
  parser.add_argument('--eval_interval', type=int, default=1e3, help='Model evaluating interval, in steps.')
  parser.add_argument('--planner_time_limit', type=int, default=5, help='The time limit (in seconds) for running the planner for each agent at each step')

  parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
  parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
  parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
  parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
  parser.add_argument('--net_width', type=int, default=64, help='Hidden net width')
  parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
  parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
  parser.add_argument('--batch_size', type=int, default=2048, help='lenth of sliced trajectory')
  parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient of Actor')
  parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
  parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
  parser.add_argument('--max_episode',type=int, default=50, help="Max number of episodes in training")
  parser.add_argument('--max_step', type = int, default=25, help="Max number of steps in each episode")
  parser.add_argument('--exploration_decay', type=float, default=0.999, help="decay rate of exploration rate")
  parser.add_argument('--activation', type=str, default='tanh',help='activation function for learning model')
  parser.add_argument('--debug', type=str2bool, default=False,help='Debug mode or Not')
  parser.add_argument('--use_central_planner', type=str2bool, default=False,help='Whether to run centralized planner for multi-agent planning, default to False')
  opt = parser.parse_args()
  opt.dvc = torch.device(opt.dvc) # from str to torch.device
#   opt.dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
  print(opt)
  return opt



def main_train(opt, debug=False):
    # Build Training Env and Evaluation Env
    env = HDDLEnv(opt.domain, opt.problem)
    env._max_episode_steps = opt.max_step
    eval_env = HDDLEnv(opt.domain, opt.problem)
    eval_env._max_episode_steps = opt.max_step
    opt.state_dim = env.observation_space.n
    print('opt.state_dim:', opt.state_dim)
    opt.action_dim = env.action_space.n
    print("opt.action_dim:",opt.action_dim)
    opt.max_e_steps = min(opt.max_step, env._max_episode_steps)

    actor_loss_list = []
    critic_loss_list = []
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

    print('Env:',opt.problem ,'  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,'   Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps)
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
    if opt.Loadmodel: policy.load(opt.ModelIdex)

    # if True: #else:
    traj_lenth, total_steps = 0, 0
    episode = 0
    eval_score = []
    while total_steps < opt.Max_train_steps:
        if episode % 30 == 0:
            print("\n\n*** Episode {} ***\n".format(episode), end='\r', flush=True)
            print("-- total train steps:",total_steps)
            print("Average step of successfull episodes:",success_step)
            print("number of successful ep", success_ep_count)
        # s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
        policy_list = [policy] * len(env.agents)
        env.reset(policy_list = policy_list)
        env_seed += 1
        random.seed(env_seed)
        done = False
        s = env.current_state
        s_num = enumerate_state(env, opt)
        # s_num = get_observation_one_hot_vector(env.current_state, env.env_dictionary['htn goal'].tasks, env.env_dictionary, device=opt.dvc).clone().detach()

        episode +=1
        episode_reward = 0
        episode_data = []
        step = 0
        ep_hierarchy_record = []
        exploration_rate = max(exploration_min, exploration_rate * opt.exploration_decay)
        
        '''Interact & train'''
        while not done:
            # print("step: ",step)
            # Exploration determines whether to set deterministic to False or True:
            if np.random.rand() < exploration_rate:
              deterministic = False # Explore with policy as a reference
            else:
              deterministic = True # Exploit 
            # Run the planner and get action dictionary:
            action_dict, env, hierarchies = run_planner_and_get_action_dict(policy_list, env, opt, deterministic=False, debug=debug)
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
                print("TRAINING policy at traj_lenth ", traj_lenth)
                a_loss, c_loss = policy.train()
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

                plt.suptitle("Actic Loss and Critic Loss")
                plt.close(fig)
                fig_dir = script_dir / 'model'
                if not os.path.exists(fig_dir):
                    os.makedirs(fig_dir)
                fig.savefig(str(fig_dir / 'loss_plot.png'))

            '''Record & log'''
            if total_steps % opt.eval_interval == 0 and total_steps>0:
                score = evaluate_policy(eval_env, policy, turns=1, opt=opt) # evaluate the policy for 'turns' times, and get averaged result
                eval_score.append(score)
                if opt.write: writer.add_scalar('ep_r', score, global_step=total_steps)
                print('EnvName:',opt.problem,'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)
                
                if total_steps % opt.eval_interval*10 == 0:
                  fig = plt.figure()
                  plt.plot(eval_score)
                  plt.xlabel("x {} steps".format(opt.eval_interval))
                  plt.ylabel("Reward")
                  plt.title("Evaluate policy")
                  plt.close(fig)
                  fig_eval_dir = script_dir / 'model'
                  if not os.path.exists(fig_eval_dir):
                    os.makedirs(fig_eval_dir)
                  fig.savefig(str(fig_eval_dir / 'evaluate_policy_plot.png'))

            '''Save model'''
            if total_steps % opt.save_interval==0:
                model_dir = script_dir / 'hddl' / 'model'
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)                     
                policy.save(total_steps)

    env.close()
    eval_env.close()


def evaluate_model(opt):
    opt.domain = str(script_dir / "overcooked_short_domain.hddl")
    opt.problem = str(script_dir / "overcooked_short_prob2.hddl")
    eval_env = HDDLEnv(opt.domain, opt.problem)
    eval_env.action_space.seed(opt.seed)
    eval_env.observation_space.seed(opt.seed)

    opt.Model = 1
    opt.state_dim = eval_env.observation_space.n
    opt.action_dim = eval_env.action_space.n
    opt.max_e_steps = min(opt.max_step, eval_env._max_episode_steps)
    
    # Load the trained policy
    policy = PPO_discrete(**vars(opt)).to(opt.dvc)
    # policy.load(opt.Model)  # Load the model checkpoint based on the `opt.Model`
    # policy.load(opt.Model, map_location=opt.dvc)
    policy.load(opt.Model)

    # Evaluate the policy
    score = evaluate_policy(eval_env, policy, turns=1, opt=opt)  # Run the evaluation for 10 episodes
    print(f'Evaluation Score for {opt.problem}: {score}')

    eval_env.close()

def run_hddlgym_without_policy(opt):
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
    score = evaluate_policy(env, policy,turns = 1, opt=opt, debug=True, deterministic = False)

opt = parse_arguments()
opt.domain = str(script_dir / "HDDL_files/Overcooked_specialization/overcooked_short_domain.hddl")
opt.problem = str(script_dir / "HDDL_files/Overcooked_specialization/overcooked_short_prob2.hddl")

### Call main_train:
if __name__ == "__main__":
    opt = parse_arguments()
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    # If you want to overwrite domain and problem files (so the command is shorter), follows are examples:
    # 1. Overcooked domain:
    # opt.domain = str(script_dir / "HDDL_files/Overcooked_specialization/overcooked_short_domain.hddl")
    # opt.problem = str(script_dir / "HDDL_files/Overcooked_specialization/overcooked_short_prob2.hddl")
    # 2. Transport domain with collaboration:
    # opt.domain = str(script_dir / "HDDL_files/ipc2023_domains/Transport/transport_domain_hddlgym_with_collab.hddl")
    # opt.problem = str(script_dir / "HDDL_files/ipc2023_domains/Transport/transport_collab_pfile01.hddl")
    # 3. Satellite domain:
    # opt.domain = str(script_dir / "HDDL_files/ipc2023_domains/Satellite/domain_hddlgym.hddl")
    # opt.problem = str(script_dir / "HDDL_files/ipc2023_domains/Satellite/2obs-1sat-2mod.hddl")

    print(opt.problem)
    main_train(opt, debug=opt.debug)

    # If want to evaluate the model instead of training: comment the previous line and call the following line
    # Make sure to have correct model ID (which is the number of training steps in its name) by including "--Model <ID>" in the command
    # evaluate_model(opt)


# command line:
# python main_train.py --domain ./HDDL_files/overcooked_short_domain.hddl --problem ./HDDL_files/overcooked_short_prob2.hddl --dvc cpu

