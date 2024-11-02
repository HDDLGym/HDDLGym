import numpy as np
from hddl_utils import enumerate_list, convert_grounded_to_lifted_index, enumerate_index_to_binary_matrix
import copy 
from hddl_env import HDDLEnv
from learning_methods import PPO_discrete, get_observation_one_hot_vector
import torch 
from main_train import opt
import os, sys, json
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir + '/hddl')       
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

def run_hddl_model(opt):
    eval_env = HDDLEnv(opt.domain, opt.problem)
    eval_env.action_space.seed(opt.seed)
    eval_env.observation_space.seed(opt.seed)

    opt.Model = 2
    opt.state_dim = eval_env.observation_space.n
    opt.action_dim = eval_env.action_space.n
    opt.max_e_steps = min(opt.max_step, eval_env._max_episode_steps)
    
    policy = PPO_discrete(**vars(opt)).to(opt.dvc)
    policy.load(opt.Model)

    score, action_array_list, hierarchies_array_list, belief_hierarchy_0_array, belief_hierarchy_1_array = get_policy_output(eval_env, policy, turns=1, opt=opt)  # Run the evaluation for 10 episodes
    print(f'Evaluation Score for {opt.problem}: {score}')

    eval_env.close()

    return action_array_list, hierarchies_array_list, belief_hierarchy_0_array, belief_hierarchy_1_array


def get_policy_output(env, policy, turns = 3, opt=None):
        print("start evaluating policy ...")
        total_scores = 0

        action_array_list = []
        hierarchies_array_list = []
        belief_hierarchy_0_array = []
        belief_hierarchy_1_array = []


        for j in range(turns):
                env.reset(policy_list=[policy]*len(env.agents))
                s = env.current_state

                '''Interact & test'''
                done = False
                step = 0
                turn_score = 0
                hierarchies_record = []
                actions_record = []
                while not done:
                    '''Interact with Env'''
                    for ag in env.agents:
                        belief_other_agents = copy.deepcopy(env.agents)
                        belief_other_agents.remove(ag)
                        belief_other_agents = [] # comment this line if want manually embed belief to be groundtruth
                        ag.decentralize_planner_agent(env, belief_other_agents = belief_other_agents,agent_policy = policy, other_agent_policy_list = [policy]*(len(env.agents)-1), deterministic=True, device=opt.dvc)
                    #extract action and convert to dict of string
                    action_dict = {}
                    all_operators = set()
                    for a in env.agents:
                        all_operators.update(a.task_method_hierarchy)
                        if len(a.task_method_hierarchy)>0:
                            action_dict[a.name] = a.task_method_hierarchy[-1]
                        else:
                            action_dict[a.name] = 'none {}'.format(a.name)
                            print("evaluate_policy: feeding none to action_dict of {} bc no hierarchy".format(a.name))

                    a_num = torch.zeros((len(env.env_dictionary['lifted operators list']) + len(env.env_dictionary['objects list']),), device=opt.dvc, dtype=torch.float32)
                    for a in env.agents:
                      a_lifted_ind, a_object_ind = convert_grounded_to_lifted_index(a.task_method_hierarchy, env.env_dictionary['lifted operators list'], env.env_dictionary['objects list'])

                      a_num_a_lifted = enumerate_index_to_binary_matrix(a_lifted_ind, array_len=len(env.env_dictionary['lifted operators list']), device=opt.dvc)
                      a_num_a_objects = enumerate_index_to_binary_matrix(a_object_ind, array_len=len(env.env_dictionary['objects list']), device=opt.dvc)
                      a_num_a = torch.cat([a_num_a_lifted.clone().detach(),
                                          a_num_a_objects.clone().detach()], dim=0).to(opt.dvc, dtype=torch.float32)

                      a_num += a_num_a.clone().detach()

                    print(f'action_dict {action_dict}')
                    s_next, r, dw, tr, info = env.step(action_dict) # dw: dead&win; tr: truncated
                    step += 1
                    done = (dw or tr)
                    s_next_num = get_observation_one_hot_vector(env.current_state, 
                                                            all_operators, 
                                                            env.env_dictionary, 
                                                            device=opt.dvc).clone().detach()
                    s = s_next
                    s_num = s_next_num
                    turn_score += r
                    hierarchies = []
                    actions_record.append(action_dict)
                    for ag in env.agents:
                        hierarchies.append(ag.task_method_hierarchy)
                    hierarchies_record.append(hierarchies)

                    # Update agents' hierarchies after step thru env:
                    for agent in env.agents:
                        agent.update_agent_hierarchy_by_checking_with_world_state(env.current_state, env.env_dictionary, debug=False)

                    if done and len(info) == len(env.env_dictionary['htn goal'].tasks): #goal reached
                        print("Evaluation turn {} completed {} after {} steps, getting score of {}\n".format(j,info,step, turn_score))
                    # if done:
                        print("Hierarchies Record for turn ", j)
                        for ii,h in enumerate(hierarchies_record):
                            print(h)
                            print("--")
                            print(actions_record[ii])

                    if done:
                        print("Episode reward of evaluating episode {} is {}".format(j, turn_score))

                    action_array_list.append(action_dict)
                    hierarchies_array_list.append(hierarchies)

                    belief_hierarchy_0 = env.agents[0].belief_other_agents[0].task_method_hierarchy
                    belief_hierarchy_1 = env.agents[1].belief_other_agents[0].task_method_hierarchy

                    belief_hierarchy_0_array.append(belief_hierarchy_0)
                    belief_hierarchy_1_array.append(belief_hierarchy_1)

                total_scores += turn_score

        return int(total_scores/turns), action_array_list, hierarchies_array_list, belief_hierarchy_0_array, belief_hierarchy_1_array


if __name__ == "__main__":
    action_list, hierarchies_array_list, belief_hierarchy_0_array, belief_hierarchy_1_array = run_hddl_model(opt)
    data = {
        "actionArrayList": action_list,
        "hierarchiesArrayList": hierarchies_array_list,
        "beliefHierarchy0Array": belief_hierarchy_0_array,
        "beliefHierarchy1Array": belief_hierarchy_1_array
    }
    data_file_path = script_dir / 'website' / "data.json"

    with open(data_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)