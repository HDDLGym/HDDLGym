''' This file defines RL policy that is used to support hierarchical planning.
The RL policy is PPO. 
Its input is the one-hot code array of the dynamic grounded predicates, lifted operators and related objects used in the previous step.
Its output is the probability of the lifted operators and related objects. 
From this output, the probability of the grounded operators can be calculated by combine lifted operators and objects.

Also, in this design, we assume policy is similar to all agents.
'''
from hddl_utils import enumerate_list, enumerate_index_to_binary_matrix, extract_object_list
from hddl_parser import convert_grounded_to_lifted_index

from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import torch
import copy
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import os
from pathlib import Path

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

    def forward(self, state):
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim = 0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob

class Critic(nn.Module):
    def __init__(self, state_dim,net_width):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.relu(self.C1(state))
        v = torch.relu(self.C2(v))
        v = self.C3(v)
        return v

def get_observation_space(env_dictionary):
    '''
    This function return the size of observation space (or state space), which is also the output size of the RL policy
    input:
    - env_dictionary: environment dictionary, result of parsing the HDDL domain and problem file
    output:
    - gym.spaces.Discrete(n), with n is the len of observation
    Note: 
    - in this approach, the observation space involves grounded dynamic predicate list, lifted action, and object list
    '''
    observation_space = gym.spaces.Discrete(n=len(env_dictionary['grounded dynamic predicate list'])+\
          len(env_dictionary['lifted operators list'])+len(env_dictionary['objects list']))

    # Replace the above line of code by the below line, if just don't want to have objects list in observation:
    # observation_space = gym.spaces.Discrete(n=len(env_dictionary['grounded dynamic predicate list'])+\
          # len(env_dictionary['lifted operators list']))
    return observation_space

def get_action_space(env_dictionary):
    '''
    This function return the size of action space, which is also the output size of the RL policy
    input:
    - env_dictionary: environment dictionary, result of parsing the HDDL domain and problem file
    output:
    - gym.spaces.Discrete(n), with n is the len of action
    Note: 
    - in this approach, the action space involves lifted action and object list
    '''
    action_space = gym.spaces.Discrete(n=len(env_dictionary['lifted operators list'])+len(env_dictionary['objects list']))
    # Use the below line of code instead if don't want include objects in action space
    # action_space = gym.spaces.Discrete(n=len(env_dictionary['lifted operators list']))
    return action_space

def get_prob_from_prob_list(operator_str, prob_list, env_dictionary,device='cpu'):
  '''This function provide probability value (prob) for the operator_str according to probabilities in the prob_list
  inputs:
  - operator_str: a string of operator_name + objects
  - prob_list: a tensor, listing prob of lifted operators and objects
  - env_dictionary: dictionary of environment's elements
  outpus:
  - prob: a float number of the probability of the operator_str
  '''
  oper_name, oper_obj = extract_object_list(operator_str)
  oper_lifted_id = None
  for lifted_ind, lifted_oper in enumerate(env_dictionary['lifted operators list']):
    if lifted_oper.name == oper_name:
      oper_lifted_id = lifted_ind
  assert oper_lifted_id != None, "Could not find lifted operators of oper {}".format(operator_str)
  indices_list = [oper_lifted_id]
  for obj in oper_obj:
    indices_list.append(env_dictionary['objects list'].index(obj) + len(env_dictionary['lifted operators list']))

  relevant_prob_tensor = prob_list[indices_list]
  prob = torch.exp(torch.div(torch.sum(torch.log(relevant_prob_tensor)),len(indices_list)+1e-20))
  return prob

def get_grounded_prob_list_from_policy_output(grounded_oper_list, prob_list, env_dictionary, device=False):
  '''
  Get grounded probability list from lifted probability list:
  inputs:
  - grounded_oper_list: list of grounded operators
  - prob_list: a tensor, listing probabilities
  - env_dictionary: dictionary of environment's elements
  '''
  if not device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  grounded_prob_list = []
  for oper in grounded_oper_list:
    grounded_prob_list.append(get_prob_from_prob_list(oper, prob_list, env_dictionary,device=device))
  return torch.tensor(grounded_prob_list, device=device)

def get_observation_one_hot_vector(current_state, all_operators, env_dictionary, device=False):
    '''Get observation from current state and operators from action hierarchies
    inputs:
    - current_state: list of predicate
    - all_operators: list of all operators
    - env_dictionary: environment dictionary
    - device: the device (CPU/GPU) to place the tensors on
    output:
    - observation: PyTorch tensor on the specified device
    '''
    if not device:
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_num = enumerate_list(current_state, env_dictionary['grounded dynamic predicate list'])
    state_num_tensor = torch.tensor(state_num, device=device, dtype=torch.float32)
    lifted_operators_id, objects_id = convert_grounded_to_lifted_index(
        all_operators, env_dictionary['lifted operators list'], env_dictionary['objects list']
    )

    lifted_operator_matrix = enumerate_index_to_binary_matrix(
        lifted_operators_id, array_len=len(env_dictionary['lifted operators list']), device=device
    )

    object_matrix = enumerate_index_to_binary_matrix(
        objects_id, array_len=len(env_dictionary['objects list']), device=device
    )

    observation = torch.cat([state_num_tensor, lifted_operator_matrix, object_matrix])

    return observation

def get_probabilities_of_operators(operators_list, agent, current_state, env_dictionary, policy=None):
    '''Get the list of probabilities of grounded operators
    inputs:
    - operators_list: list of grounded operators
    - agent: Agent instance, mainly to use its method: Agent.get_agent_observation
    - current_state: list of predicates representing the world state
    - env_dictionary: environment dictionary, containing information fo domain and problem
    - policy: RL policy, have method select_action to return probability list, if None ==> generate probability randomly
    outputs:
    - probabilites: tensor, listing probabilities of operators in operators_list
    '''
    if policy == None:
        return torch.rand(len(operators_list))
    observation_num = agent.get_agent_observation(current_state,env_dictionary)
    prob_list = subtask_policy.select_action(observation_num, value=True)
    probabilites = get_grounded_prob_list_from_policy_output(operators_list, prob_list, env_dictionary)
    return probabilites

def get_probability_of_valid_operator_combinations(valid_operator_combination_list, policy_list, main_agent_index, current_state, env_dictionary, agents, device=False):
    '''
    Generate the list of probabilities for the list of valid operator combinations 

    Inputs:
    - valid_operator_combination_list: list of valid operator combinations, 
                                        e.g [('none agent-1', 'none agent-2'), ('none agent-1', 'task-move agent-2 loc-1 loc-2')]
    - policy_list: list of policy of agents, e.g. [policy_agent_1, policy_agent_2,...] or 
                    [(policy_agent1_method, policy_agent1_task_action), (policy_agent2_method, policy_agent2_task_action)]
    - main_agent_index: index of the main agent in the planning, in range [0, len(policy_list))
    - env_dictionary: dictionary of info from the HDDL domain and problem 
    - agents: list of all agents (Agent instances), mainly to use info of their current hierarchies if needed
    - device: whether 'cpu' or 'gpu', default is False, then will use gpu if possible

    outputs:
    - probability_list_of_combinations: list of probabilities


    '''
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probability_list_of_combinations = []
    # For this method, we assume that each agent has only 1 policy that works for all operators of the hierarchy
    main_policy = policy_list[main_agent_index]

    all_operators = []
    for agent in agents:
        all_operators += agent.prev_task_method_hierarchy
        if agent.htn_goal != None:
            all_operators += agent.htn_goal.pending_tasks
            all_operators += agent.htn_goal.remaining_tasks
    all_operators = set(all_operators)
    all_operators = list(all_operators)

    observation_one_hot_vector = get_observation_one_hot_vector(current_state, all_operators, env_dictionary, device=device)
    # Note: assume that we use policy of the main agent to get probability for all operators of all agents (including belief of other agents)
    prob_list = main_policy.select_action(observation_one_hot_vector, value=True)
    for com in valid_operator_combination_list:
        com_values = get_grounded_prob_list_from_policy_output(com, prob_list, env_dictionary, device=device)
        probability_list_of_combinations.append(torch.prod(com_values))

    probability_list_of_combinations = torch.tensor(probability_list_of_combinations,device=device)
    probability_list_of_combinations = probability_list_of_combinations/(torch.sum(probability_list_of_combinations)+1e-20)
    return probability_list_of_combinations

def enumerate_state(env, opt):
    '''
    Generate a one-hot code for current state of the world
    inputs:
    - env: an HDDLEnv instance
    - opt: parameters
    outputs:
    - s_num: a one-hot array for the current state
    '''
    all_operators = set(env.env_dictionary['htn goal'].pending_tasks)
    for agent_ in env.agents:
        all_operators.update(agent_.task_method_hierarchy)
    s_num = get_observation_one_hot_vector(env.current_state, 
                                                        all_operators, 
                                                        env.env_dictionary, 
                                                        device=opt.dvc).clone().detach()
    return s_num

def enumerate_action(env, opt):
    ''' Generate a one-hot code for action hierarchies of all agents
    inputs:
    - env: an HDDLEnv instance
    - opt: parameters
    output:
    - a_num: a one-hot array of action hierarchies
    '''
    a_num = torch.zeros((len(env.env_dictionary['lifted operators list']) + len(env.env_dictionary['objects list']),), device=opt.dvc, dtype=torch.float32)
    for agent_ in env.agents:
      a_lifted_ind, a_object_ind = convert_grounded_to_lifted_index(agent_.task_method_hierarchy, env.env_dictionary['lifted operators list'], env.env_dictionary['objects list'])
      a_num_a_lifted = enumerate_index_to_binary_matrix(a_lifted_ind, array_len=len(env.env_dictionary['lifted operators list']), device=opt.dvc)
      a_num_a_objects = enumerate_index_to_binary_matrix(a_object_ind, array_len=len(env.env_dictionary['objects list']), device=opt.dvc)
      a_num_a = torch.cat([a_num_a_lifted.clone().detach(),
                          a_num_a_objects.clone().detach()], dim=0).to(opt.dvc, dtype=torch.float32)
      a_num += a_num_a.clone().detach()
    return torch.clamp(a_num, max=1.0)

def evaluate_policy(env, policy, turns = 3, opt=None):
        print("start evaluating policy ...")
        total_scores = 0
        for j in range(turns):
                env.reset(policy_list=[policy]*len(env.agents))
                s = env.current_state
                s_num = np.array(enumerate_list(s, env.env_dictionary['grounded dynamic predicate list']))
                s_num = np.concatenate((np.array(enumerate_list(s, env.env_dictionary['grounded dynamic predicate list'])),np.zeros((opt.action_dim,))))

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
                        ag.decentralize_planner_agent(env, belief_other_agents = belief_other_agents,agent_policy = policy,\
                         other_agent_policy_list = [policy]*(len(env.agents)-1), deterministic=True, device=opt.dvc, debug=opt.debug)
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
                        # print("action record: ")
                        # for a in actions_record:
                            print(actions_record[ii])

                    if done:
                        print("Episode reward of evaluating episode {} is {}".format(j, turn_score))

                total_scores += turn_score


        return int(total_scores/turns)


#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        print('Wrong Input.')
        raise


class PPO_discrete():
    def __init__(self, **kwargs):
        # Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)

        '''Build Actor and Critic'''
        self.actor = Actor(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic = Critic(self.state_dim, self.net_width).to(self.dvc)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        '''Build Trajectory holder'''
        self.s_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.a_hoder = np.zeros((self.T_horizon, self.action_dim), dtype=np.int64)
        self.r_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.s_next_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.prob_a_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.done_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)
        self.dw_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)


    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        return self

    def select_action(self, s, deterministic=False, value=False):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float().to(self.dvc)
        else:
            s = s.float().to(self.dvc)

        # s = torch.from_numpy(s).float().to(self.dvc)
        with torch.no_grad():
            pi = self.actor.pi(s, softmax_dim=0)
            if value:
                return pi.cpu().detach()
            if deterministic:
                a = torch.argmax(pi).item()
                return a, None
            else:
                m = Categorical(pi)
                a = m.sample().item()
                # pi_a = pi[a].item()
                pi_a = pi.gather(1, torch.tensor([[a]])).item()  # For batch dimension
                return a, pi_a

    def train(self):
        a_loss_list = []
        c_loss_list = []
        self.entropy_coef *= self.entropy_coef_decay #exploring decay
        '''Prepare PyTorch data from Numpy data'''
        s = torch.from_numpy(self.s_hoder).to(self.dvc)
        a = torch.from_numpy(self.a_hoder).to(self.dvc)
        r = torch.from_numpy(self.r_hoder).to(self.dvc)
        s_next = torch.from_numpy(self.s_next_hoder).to(self.dvc)
        old_prob_a = torch.from_numpy(self.prob_a_hoder).to(self.dvc)
        done = torch.from_numpy(self.done_hoder).to(self.dvc)
        dw = torch.from_numpy(self.dw_hoder).to(self.dvc)

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)

            '''dw(dead and win) for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, done in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)
            td_target = adv + vs
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-20))  #sometimes helps

        """PPO update"""
        #Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))

        for _ in range(self.K_epochs):
            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.dvc)
            a_loss_epoch = []
            c_loss_epoch = []
            s, a, td_target, adv, old_prob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone()

            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))

                '''actor update'''
                prob = self.actor.pi(s[index], softmax_dim=1)                
                prob_a = (prob*a[index]).sum(dim=1, keepdim=True)
                log_prob_a = torch.log(prob_a + 1e-20)
                log_old_prob_a = torch.log(old_prob_a[index] + 1e-20)
                ratio = torch.exp(log_prob_a - log_old_prob_a)

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                entropy = Categorical(probs = prob_a).entropy().sum(0, keepdim=True)
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy
                a_loss_epoch.append(a_loss.mean().item())

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                '''critic update'''
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                c_loss_epoch.append(c_loss.item())
                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()
            
            if len(a_loss_epoch)>0:
                a_loss_list.append(sum(a_loss_epoch)/len(a_loss_epoch))
                c_loss_list.append(sum(c_loss_epoch)/len(c_loss_epoch))
        return a_loss_list, c_loss_list

    def put_data(self, s, a, r, s_next, prob_a, done, dw, idx):
        self.s_hoder[idx] = s
        self.a_hoder[idx] = a
        self.r_hoder[idx] = r
        self.s_next_hoder[idx] = s_next
        self.prob_a_hoder[idx] = prob_a
        self.done_hoder[idx] = done
        self.dw_hoder[idx] = dw

    def save(self, episode):
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        torch.save(self.critic.state_dict(), str(script_dir / "model/ppo_critic{}.pth").format(episode))
        torch.save(self.actor.state_dict(), str(script_dir / "model/ppo_actor{}.pth").format(episode))

    def load(self, episode):
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.critic.load_state_dict(torch.load(str(script_dir / "model/ppo_critic{}.pth").format(episode)))
        self.actor.load_state_dict(torch.load(str(script_dir / "model/ppo_actor{}.pth").format(episode)))