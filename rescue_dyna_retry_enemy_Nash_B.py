#消融实验，只有威胁表示+博弈，没有分层架构
# python3
import numpy
import numpy as np
import matplotlib
import copy
from sympy import *  # solve the equation
import pandas as pd
import pylab as pl
import seaborn as sns; sns.set()
import csv
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from copy import deepcopy
import random

# A wrapper class for a maze, containing all the information about the maze.
# Basically it's initialized to DynaMaze by default, however it can be easily adapted
# to other maze
class Maze:
    def __init__(self):
        # maze width
        self.size = 20
        self.WORLD_WIDTH = self.size
        self.grid_map = np.loadtxt('map2.txt', dtype=int)
        #第一个环境用的map2.txt
        #self.grid_map = np.loadtxt('map0_nowall.txt', dtype=int)

        self.WORLD_HEIGHT = self.size  # maze height

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.ACTION_STOP = 4
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT, self.ACTION_RIGHT]

        self.obsvt_theta = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.obsvt_Manhattandistc = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # start state
        # self.START_STATE = [2, 0]
        self.START_STATE = []

        # goal state
        # self.GOAL_STATES = [[2, 5]]
        self.GOAL_STATES = []

        # human state
        # self.Human_STATES = [[2, 5], [3, 6], [1, 4]]    #有三个目标
        self.Human_STATES = []

        # all obstacles
        # self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.obstacles = []

        for i in range(0, self.size):
            for j in range(0, self.size):
                if self.grid_map[i, j] == 1:   #out door
                    self.obstacles.extend([[i, j]])
                elif self.grid_map[i, j] == 2:
                    self.Human_STATES.extend([[i, j]])
                elif self.grid_map[i, j] == 4:
                    self.START_STATE.extend([i, j])
                elif self.grid_map[i, j] == 5:
                    self.GOAL_STATES.extend([[i, j]])

        # time to change obstacles
        self.obstacle_switch_time = None

        # initial state action pair values
        # self.stateActionValues = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))

        # the size of q value
        self.q_size = (self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions))

        # max steps
        #self.max_steps = float('inf')
        self.max_steps = 2000

        # track the resolution for this maze
        self.resolution = 1


    def step_enemy(self, state, action, GOAL_STATES, enemy, rescue_str):
        x, y = state
        # print(state)
        reward = 0

        if action == self.ACTION_UP:
            x = max(x - 1, 0)
            #action0 = [-0.4, 0, 0, 0]
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
            #action0 = [0.4, 0, 0, 0]
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
            #action0 = [0, -0.4, 0, 0]
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
            #action0 = [0, 0.4, 0, 0]
        elif action == self.ACTION_STOP:
            y = y
            x = x

        if [(x),(y)] in self.obstacles:
            x, y = state
            action = 4  #影响不大
        #if [x, y] in GOAL_STATES:
        #    reward = 4.0
        if state[0] == enemy[0] and state[1]==enemy[1]:
            reward = -2

        for i in self.Human_STATES :
            if np.sqrt(np.sum(np.square(np.array(state) - np.array(i)))) <= 0.2 :
                if i in rescue_str:
                    reward = 4
                    rescue_str.remove(i)
                    self.human_num += 1


        return [x, y], reward, action, rescue_str

# a wrapper class for parameters of dyna algorithms
class DynaParams:
    def __init__(self):
        # discount
        self.gamma = 0.95

        # probability for exploration
        self.epsilon = 0.1

        # step size
        self.alpha = 0.1

        # weight for elapsed time
        self.time_weight = 0

        # n-step planning
        self.planning_steps = 5

        # average over several independent runs
        self.runs = 10

        # algorithm names
        self.methods = ['Dyna-Q', 'Dyna-Q+']

        # threshold for priority queue
        self.theta = 0


# choose an action based on epsilon-greedy algorithm
def choose_action(state, q_value, maze, dyna_params):
    if np.random.binomial(1, dyna_params.epsilon) == 1:
        return np.random.choice(maze.actions)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])


def choose_action_enemy(obsvt, q_value_enemy, maze, dyna_params):
    if np.random.binomial(1, dyna_params.epsilon) == 1:
        return np.random.choice(maze.actions)
    else:
        values = q_value_enemy[obsvt[0], obsvt[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])


# Trivial model for planning in Dyna-Q
class TrivialModel:
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand

    # feed the model with previous experience
    def feed(self, state, action, next_state, reward):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        if tuple(state) not in self.model.keys():
            self.model[tuple(state)] = dict()
        self.model[tuple(state)][action] = [list(next_state), reward]

    def feed_enemy (self, state, obsvt, action, next_state, next_obsvt, reward):
        obsvt = deepcopy(obsvt)
        next_obsvt = deepcopy(next_obsvt)
        if tuple(obsvt) not in self.model.keys():
            self.model[tuple(obsvt)] = dict()
        self.model[tuple(obsvt)][action] = [list(next_obsvt), reward]

    # randomly sample from previous experience
    def sample(self):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return list(state), action, list(next_state), reward


    def sample_enemy(self, maze, enemy):
        state_index = self.rand.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = self.rand.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        obsvt = obsvt_dyna(maze.grid_map, len(maze.obsvt_Manhattandistc)-1, np.array(state), np.array(enemy))
        next_state = deepcopy(next_state)
        next_obsvt = obsvt_dyna(maze.grid_map, len(maze.obsvt_Manhattandistc)-1, np.array(next_state), np.array(enemy))
        return list(state), list(obsvt), action, list(next_state), list(next_obsvt), reward

def obsvt_dyna(maze_map, len_obsvt_Manhattandistc, state, enemy):
    obsvt_theta = 0
    obsvt_Manhattandistc = 0

    #print(state, enemy)

    #if enemy.__sizeof__() == 1:
        # print(enemy)
    obsvt_Manhattandistc = abs(state[0] - enemy[0]) + abs(state[1] - enemy[1])
    if obsvt_Manhattandistc > len_obsvt_Manhattandistc:
        obsvt_theta = 8
        obsvt_Manhattandistc = 8  # to be max
    elif abs(state[0] - enemy[0]) == 0 and abs(state[1] - enemy[1]) == 0:
        obsvt_theta = 9
    elif enemy[0] > state[0] and enemy[1] == state[1]:
        obsvt_theta = 0
    elif enemy[0] > state[0] and enemy[1] < state[1]:
        obsvt_theta = 1
    elif enemy[0] == state[0] and enemy[1] < state[1]:
        obsvt_theta = 2
    elif enemy[0] < state[0] and enemy[1] < state[1]:
        obsvt_theta = 3
    elif enemy[0] < state[0] and enemy[1] == state[1]:
        obsvt_theta = 4
    elif enemy[0] < state[0] and enemy[1] > state[1]:
        obsvt_theta = 5
    elif enemy[0] == state[0] and enemy[1] > state[1]:
        obsvt_theta = 6
    elif enemy[0] > state[0] and enemy[1] > state[1]:
        obsvt_theta = 7

    else:
        part_num = max(abs(state[0] - enemy[0]), abs(state[1] - enemy[1]))
        for i in range(1, part_num):
            delta_x = abs(state[0] - enemy[0])
            delta_y = abs(state[1] - enemy[1])

            if delta_x == 0:
                a = state[0]
            else:
                a = (i / delta_x) * state[0] + (1 - (i / delta_x)) * enemy[0]
            if delta_y == 0:
                b = state[1]
            else:
                b = (i / delta_y) * state[1] + (1 - (i / delta_y)) * enemy[1]

            if maze_map[round(a), round(b)] == 1:
                obsvt_theta = 8   #被墙挡住了
                break

    return [obsvt_theta, obsvt_Manhattandistc]     #输出当前敌方的相对位置


def delta_d(agent0, agent1, enemy0, enemy1):
    d0 = abs(agent0[0] - enemy0[0]) + abs(agent0[1] - enemy0[1])
    d1 = abs(agent1[0] - enemy1[0]) + abs(agent1[1] - enemy1[1])
    d = d1 - d0
    return d

def game_matrix(maze, agent_state, enemy0):  # game_sum =0  reward_agent = -reward_enemy
    game_reward_matrix = np.zeros((4, 4))
    for i in range(0, 4):
        for j in range(0, 4):

            agent_state_new = deepcopy(agent_state)
            if i == 0:
                agent_state_new[0] = max(agent_state_new[0] - 1, 0)
            elif i == 1:
                agent_state_new[0] = min(agent_state_new[0] + 1, maze.WORLD_HEIGHT - 1)
            elif i == 2:
                agent_state_new[1] = max(agent_state_new[1] - 1, 0)
            elif i == 3:
                agent_state_new[1] = min(agent_state_new[1] + 1, maze.WORLD_WIDTH - 1)

            enemy0_new = deepcopy(enemy0)
            if j == 0:
                enemy0_new[0] = max(enemy0_new[0] - 1, 0)
            elif j == 1:
                enemy0_new[0] = min(enemy0_new[0] + 1, maze.WORLD_HEIGHT - 1)
            elif j == 2:
                enemy0_new[1] = max(enemy0_new[1] - 1, 0)
            elif j == 3:
                enemy0_new[1] = min(enemy0_new[1] + 1, maze.WORLD_WIDTH - 1)
            game_reward_matrix[i, j] = delta_d(agent_state, agent_state_new, enemy0, enemy0_new)

            if agent_state_new in maze.obstacles:
                game_reward_matrix[i, j] -= 3
            elif agent_state_new in maze.GOAL_STATES or agent_state_new in maze.Human_STATES:
                game_reward_matrix[i, j] += 4
            elif agent_state_new == enemy0_new:
                game_reward_matrix[i, j] -= 3

    return game_reward_matrix

# play for an episode for Dyna-Q algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @dyna_params: several params for the algorithm

def enemy_plan(self, state, enemy):
    x, y = enemy
   # print(state)
    reward = 0
    enemy_next = []
    if delta_d(state, state, enemy, (x-1, y)) < 0:
        enemy_next.extend([[x-1, y]])
    if delta_d(state, state, enemy, (x+1, y)) < 0:
        enemy_next.extend([[x+1, y]])
    if delta_d(state, state, enemy, (x, y-1)) < 0:
        enemy_next.extend([[x, y-1]])
    if delta_d(state, state, enemy, (x, y+1)) < 0:
        enemy_next.extend([[x, y+1]])

    enemy_position = random.choice(enemy_next)

    if enemy_position in self.obstacles or enemy_next.__len__() == 0:
        enemy_position = enemy

    return enemy_position


def dyna_q_enemy(q_value, q_value_enemy, model, maze, dyna_params, START_STATES, GOAL_STATES, enemy,enemy_path):  # 遇到了敌人，随后采取决策。
    state = START_STATES
    steps = 0
    Reward = 0
    path = [state]
    maze.human_num = 0
    str = copy.deepcopy(maze.Human_STATES)
    while str.__len__() != 0 or state not in GOAL_STATES:
        # track the steps
        steps += 1
        enemy = np.array(enemy)
        enemy0 = copy.deepcopy(enemy)
        n = enemy_path.__len__()
        num_list = enemy_path.index([enemy[0], enemy[1]])
        enemy = enemy_path[(num_list+1)-n*((num_list+1)//n)]    #取余数
        # print('enemy=', enemy, 'steps-n*(steps//n)',steps-n*(steps//n))
        [obsvt_theta, obsvt_Manhattandistc] = obsvt_dyna(maze.grid_map, len(maze.obsvt_Manhattandistc) - 1, state, enemy)
        obsvt = [obsvt_theta, obsvt_Manhattandistc]


        if obsvt_Manhattandistc <= 5 and obsvt_theta <= 7:
            p = np.array([1/3, 2/3])
            policy = np.random.choice([0, 1], p=p.ravel())

            if policy == 0:
                enemy = enemy_path[(num_list+1)-n*((num_list+1)//n)]

            else:
                enemy = enemy_plan(maze, state, enemy)
                if enemy !=  enemy_path[(num_list+1)-n*((num_list+1)//n)] :
                    enemy_path.insert((num_list+1)-n*((num_list+1)//n), enemy)
                    enemy_path.insert(((num_list-1)-n*((num_list+1)//n)+1), enemy)

        if obsvt_Manhattandistc > 7 or obsvt_theta > 7:
            action = choose_action(state, q_value, maze, dyna_params)
        else:
            p = np.array([1/3, 2/3])
            #p = np.array([0,1])
            policy = np.random.choice([0, 1], p=p.ravel())
            if policy == 0:
                action = choose_action(state, q_value, maze, dyna_params)
            else:
                action = choose_action_enemy(obsvt, q_value_enemy, maze, dyna_params)

        next_state, reward, action, str = maze.step_enemy(state, action, GOAL_STATES, enemy ,str)
        next_obsvt= obsvt_dyna(maze.grid_map, len(maze.obsvt_Manhattandistc) - 1, next_state, enemy)

        # while np.sqrt(np.sum(np.square(np.array(next_state) - np.array(enemy)))) <= 2 :
        # reward -= 0.1 #when action is decided, each action will make the reward changed(-1) ,could be less : 0.02

        Reward = Reward + reward

        # Q-Learning update
        q_value_enemy[(obsvt[0]), (obsvt[1]), action] += \
            dyna_params.alpha * (reward + dyna_params.gamma * np.max(
                q_value_enemy[(next_obsvt[0]), (next_obsvt[1]), :]) -
                                 q_value_enemy[(obsvt[0]), (obsvt[1]), action])

        q_value[state[0], state[1], action] += \
            dyna_params.alpha * (reward + dyna_params.gamma * np.max(q_value[next_state[0], next_state[1], :]) -
                                 q_value[state[0], state[1], action])

        # feed the model with experience
        model.feed(state, action, next_state, reward)

        # sample experience from the model
        for t in range(0, dyna_params.planning_steps):
            state_, obsvt_, action_, next_state_, next_obsvt_, reward_ = model.sample_enemy(maze, enemy)

            q_value_enemy[(obsvt_[0]), (obsvt_[1]), action_] += \
                dyna_params.alpha * (reward_ + dyna_params.gamma * np.max(
                    q_value_enemy[(next_obsvt_[0]), (next_obsvt_[1]), :]) -
                                     q_value_enemy[(obsvt_[0]), (obsvt_[1]), action_])

            q_value[state_[0], state_[1], action_] += \
                dyna_params.alpha * (reward_ + dyna_params.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) -
                                     q_value[state_[0], state_[1], action_])

        state = next_state

        # check whether it has exceeded the step limit
        if steps > maze.max_steps:
            break
        path.extend([next_state])  # show the path

    #print('enemy_path:', enemy_path)
    #print('path:', path)
    # return steps, Reward, enemy
    return steps, q_value, q_value_enemy, path, enemy, Reward, enemy_path


# Figure 8.2, DynaMaze, use 10 runs instead of 30 runs
def figure_main():
    # set up an instance for DynaMaze
    dyna_maze = Maze()
    dyna_params = DynaParams()
    runs = 1  #10
    episodes = 100  #100
    # planning_steps = [0, 5, 10]
    planning_steps = [5]
    steps_goal = np.zeros((len(planning_steps), episodes))
    steps_human1 = np.zeros((len(planning_steps), episodes))
    steps_human2 = np.zeros((len(planning_steps), episodes))
    steps_human3 = np.zeros((len(planning_steps), episodes))
    rewards1 = np.zeros((len(planning_steps), episodes))
    rewards2 = np.zeros((len(planning_steps), episodes))
    rewards3 = np.zeros((len(planning_steps), episodes))
    rewards4 = np.zeros((len(planning_steps), episodes))
    #test_save_data = []#save the result

    for i, planning_step in enumerate(planning_steps):

        dyna_params.planning_steps = planning_step
        q_value_goal = np.zeros(dyna_maze.q_size)
        q_value_human1 = np.zeros(dyna_maze.q_size)
        q_value_human2 = np.zeros(dyna_maze.q_size)
        q_value_human3 = np.zeros(dyna_maze.q_size)
        q_value_enemy_goal = np.zeros(dyna_maze.q_size)
        q_value_enemy_human1 = np.zeros(dyna_maze.q_size)
        q_value_enemy_human2 = np.zeros(dyna_maze.q_size)
        q_value_enemy_human3 = np.zeros(dyna_maze.q_size)

        # generate an instance of Dyna-Q model
        model_goal = TrivialModel()
        model_human1 = TrivialModel()
        model_human2 = TrivialModel()
        model_human3 = TrivialModel()
        enemy = [18, 18]
        for ep in range(episodes):
            print('planning step:', planning_step, 'episode:', ep)

            point_turn = []
            agent_path = []
            enemy_path0 = np.loadtxt('enemy_path.txt', dtype=int)
            enemy_path = copy.deepcopy(enemy_path0)
            enemy_path = enemy_path.tolist()

                #steps_human1_new, q_value_human1, agent_path1 = dyna_q(q_value_human1, model_human1, dyna_maze, dyna_params, dyna_maze.START_STATE, [dyna_maze.Human_STATES[0]])
                #steps_human2_new, q_value_human2, agent_path2 = dyna_q(q_value_human2, model_human2, dyna_maze, dyna_params, dyna_maze.Human_STATES[0], [dyna_maze.Human_STATES[1]])
                #steps_human3_new, q_value_human3, agent_path3 = dyna_q(q_value_human3, model_human3, dyna_maze, dyna_params, dyna_maze.Human_STATES[1], [dyna_maze.Human_STATES[2]])
                #steps_goal_new, q_value_goal, agent_path4 = dyna_q(q_value_goal, model_goal, dyna_maze, dyna_params, dyna_maze.Human_STATES[2], dyna_maze. GOAL_STATES)

            enemy_current = enemy# 敌人起点

            steps_human1_new, q_value_human1, q_value_enemy_human1, agent_path1, enemy_current, rewards1_new, enemy_path = dyna_q_enemy(q_value_human1, q_value_enemy_human1, model_human1, dyna_maze,dyna_params,dyna_maze.START_STATE,[dyna_maze.Human_STATES[0]], enemy_current, enemy_path)
            #steps_human2_new, q_value_human2, q_value_enemy_human2,agent_path2, enemy_current, rewards2_new, enemy_path = dyna_q_enemy(q_value_human2, q_value_enemy_human2, model_human2, dyna_maze,dyna_params,dyna_maze.Human_STATES[0],[dyna_maze.Human_STATES[1]], enemy_current, enemy_path)
            #steps_human3_new, q_value_human3, q_value_enemy_human3, agent_path3, enemy_current, rewards3_new, enemy_path = dyna_q_enemy(q_value_human3, q_value_enemy_human3, model_human3, dyna_maze,dyna_params,dyna_maze.Human_STATES[1],[dyna_maze.Human_STATES[2]], enemy_current, enemy_path)
            #steps_goal_new, q_value_goal, q_value_enemy_goal, agent_path4, enemy_current, rewards4_new, enemy_path = dyna_q_enemy(q_value_goal, q_value_enemy_goal, model_goal,dyna_maze, dyna_params,dyna_maze.Human_STATES[2],dyna_maze.GOAL_STATES, enemy_current, enemy_path)

            #steps_goal[i, ep] += steps_goal_new
            steps_human1[i, ep] += steps_human1_new
            #steps_human2[i, ep] += steps_human2_new
            #steps_human3[i, ep] += steps_human3_new

            rewards1[i, ep] += rewards1_new
            #rewards2[i, ep] += rewards2_new
            #rewards3[i, ep] += rewards3_new
            #rewards4[i, ep] += rewards4_new

            agent_path.extend(agent_path1)
            #agent_path.extend(agent_path2)
            #agent_path.extend(agent_path3)
            #agent_path.extend(agent_path4)

            if planning_step == planning_steps[0] and ep == episodes-1:
                print('last time')
                np.savetxt('/home/cy/pycharm_learn/learn/grid_rescue_mne_enemy_policy_2_with observation0322/images/nonash_0409/agent_path_0409_nonash.txt',agent_path)
                np.savetxt('/home/cy/pycharm_learn/learn/grid_rescue_mne_enemy_policy_2_with observation0322/images/nonash_0409/enemy_path_0409_nonash.txt',enemy_path)
                print(q_value_human1)
                print(q_value_human2)
                print(q_value_human3)
                print(q_value_goal)

    x = []
    y = []
    for ii in range(len(agent_path)):
        x.extend([agent_path[ii][0]])
        y.extend([agent_path[ii][1]])
    pl.plot(x, y)
    pl.show
    pl.savefig('/home/cy/pycharm_learn/learn/grid_rescue_mne_enemy_policy_2_with observation0322/images/nonash_0409/maze_map_0409_nonash'
               ''
               '.png')
    pl.close()

    for i in range(len(planning_steps)):
        plt.plot(steps_goal[i, :] + steps_human1[i, :] + steps_human2[i, :] + steps_human3[i, :],
                 label='%d planning steps' % (planning_steps[i]))
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()

    plt.savefig('/home/cy/pycharm_learn/learn/grid_rescue_mne_enemy_policy_2_with observation0322/images/nonash_0409/maze_steps_0409_nonash'
                ''
                '.png')
    pl.close()

    #np.savetxt('/home/cy/pycharm_learn/learn/grid_rescue_mne_enemy_policy_2_with observation0322/images/nonash_0409/rewards_nash_0409.txt',rewards1+rewards2+rewards3+rewards4)
    #np.savetxt('/home/cy/pycharm_learn/learn/grid_rescue_mne_enemy_policy_2_with observation0322/images/nonash_0409/steps_nash_0409.txt',steps_human1+steps_human2+steps_human3 + steps_goal)

    for i in range(len(planning_steps)):
        plt.plot(rewards1[i, :] + rewards2[i, :] + rewards3[i, :] + rewards4[i, :] ,
                 label='%d rewards' % (planning_steps[i]))
    plt.xlabel('episodes')
    plt.ylabel('rewards per episode')
    plt.legend()

    plt.savefig('/home/cy/pycharm_learn/learn/grid_rescue_mne_enemy_policy_2_with observation0322/images/nonash_0409/maze_rewards_0409_nonash'
                ''
                '.png')

    return (rewards1 + rewards2+ rewards3 + rewards4, steps_human1+steps_human2+steps_human3 + steps_goal)


if __name__ == '__main__':
    R0 = csv.reader(open('/home/cy/pycharm_learn/learn/grid_rescue_mne_enemy_policy_2_with observation0322/data_deal_reward.csv'))
    # csv_writer = csv.DictWriter(f, fieldnames=['run1', 'run2', 'run3', 'run4', 'run5', 'run6', 'run7', 'run8', 'run9',
    #                                          'run10'])
    R = list(R0)
    S0 = csv.reader(open('/home/cy/pycharm_learn/learn/grid_rescue_mne_enemy_policy_2_with observation0322/data_deal_step.csv'))
    S = list(S0)
    run = 30
    for i in tqdm(range(run)):
        [reward, steps] = figure_main()

        for j in range(reward.size):
            R[j][i] = reward[0][j]
            writer = csv.writer(open('/home/cy/pycharm_learn/learn/grid_rescue_mne_enemy_policy_2_with observation0322/data_deal_reward.csv', 'w'))
            writer.writerows(R)

            S[j][i] = steps[0][j]
            writer = csv.writer(open('/home/cy/pycharm_learn/learn/grid_rescue_mne_enemy_policy_2_with observation0322/data_deal_step.csv', 'w'))
            writer.writerows(S)
    print('DOWN')
