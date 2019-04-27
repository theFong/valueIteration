import numpy as np
from enum import Enum
from itertools import chain
DEBUG = False

class State(object):

    def __init__(self, col, row, grid, actions,
                reward=0., 
                correct_action_prob=.7, 
                incorrect_action_prob=.1):
        self.reward = reward
        self.prev_value = reward
        self.value = 0
        self.actions = actions
        self.pos = np.array([col, row])
        self.grid = grid
        self.correct_action_prob = correct_action_prob
        self.incorrect_action_prob = incorrect_action_prob
    
    def __iter__(self):
        def generator():
            for a in self.actions:
                yield self.move(a)

        return generator()

    def next(self):
        return next(iter(self))

    def move(self, action):
        # returns move with each (transition prob, reward, future value)
        params = []
        for a in self.actions:
            prob = .1
            if a[0] == action[0]:
                prob = .7
            state_prime = self.move_state(a)
            params.append((prob, state_prime.reward, state_prime, a))
            # print((prob, state_prime.reward, state_prime.value, state_prime.prev_value, a[2]))
        return params, self.move_state(action)
            
    def move_state(self, action):
        new_pos = self.pos + action[1]
        # print(self.pos, new_pos)

        if new_pos[0] < 0:
            new_pos[0] = 0
        if new_pos[1] < 0:
            new_pos[1] = 0
        if new_pos[0] >= len(self.grid):
            new_pos[0] = len(self.grid) - 1
        if new_pos[1] >= len(self.grid[0]):
            new_pos[1] = len(self.grid) - 1
        return self.grid[new_pos[0]][new_pos[1]]

    def __str__(self):
        return "{},{}".format(self.value, self.prev_value)

    def reset(self):
        self.prev_value = self.value

    def delta(self):
        return abs(self.value - self.prev_value)

class Environment(object):

    class Symbol(Enum):
        OBSTACLE = "o"
        DESTINATION = "."

    def __init__(self, file_name, 
                correct_action_prob=.7, 
                incorrect_action_prob=.1, 
                action_cost=-1.,
                actions=[("east", np.array([1,0]), ">"), ("west", np.array([-1,0]), "<"), ("north",np.array([0,-1]), "^"), ("south",np.array([0,1]), "v") ]):
        # will be overwritten
        self.grid_size = 0
        self.num_obstacles = 0
        self.obstacles = [(0,0)]
        self.destination = (0,0)
        self.grid = [[]]
        self.correct_action_prob = correct_action_prob
        self.incorrect_action_prob = incorrect_action_prob
        self.action_cost = action_cost
        self.actions = actions
        self.read(file_name)

    def read(self, file_name):
        with open(file_name, "r") as f:
            self.grid_size = int(next(f))
            self.num_obstacles = int(next(f))

            env_items= []
            for i in f:
                row= i.split(",")
                coord = (int(row[0]), int(row[1]))
                env_items.append(coord)

            self.obstacles = env_items[:-1]
            self.destination = env_items[-1]

    def init_board(self):
        self.grid = [ [ None for r in range(self.grid_size)] for c in range(self.grid_size) ]
        for c in range(self.grid_size):
            for r in range(self.grid_size):
                s = State(c, r, self.grid, self.actions, correct_action_prob=self.correct_action_prob, incorrect_action_prob=self.incorrect_action_prob, reward=self.action_cost)
                self.grid[c][r] = s

        for o in self.obstacles:
            self.grid[o[0]][o[1]].reward += -100
            self.grid[o[0]][o[1]].prev_value += -100
        
        self.grid[self.destination[0]][self.destination[1]].reward += 100
        self.grid[self.destination[0]][self.destination[1]].prev_value += 100
        if DEBUG:
            self.print_prev_value()
            # self.print_reward()

    def average_change(self):
        all_delta = [ s.delta() for s in chain.from_iterable(self.grid) ]
        sum_delta = sum(all_delta)
        avg = sum_delta / (self.grid_size **2)
        # print(avg)
        return avg

    # return a iterator over 2d array grid
    def __iter__(self):
        def generator():
            for c in self.grid:
                for r in c:
                    yield r
            # self.reset()
        return generator()

    def reset(self):
        # move value to prev value
        for r in self.grid:
            for c in r:
                c.reset()

    def print_reward(self):
        strung = [ [r.reward for r in c] for c in self.grid]
        np_array = np.array(strung).T
        print(np_array)

    def print_prev_value(self):
        # for pretty printing
        # coord tuples are (col, row) so take transpose
        strung = [ [r.prev_value for r in c] for c in self.grid]
        np_array = np.array(strung).T

        print(np_array)
    
    # its okay to cast and print because uncommon operation used for debugging
    def __str__(self):
        # for pretty printing
        # coord tuples are (col, row) so take transpose
        strung = [ [r.value for r in c] for c in self.grid]
        np_array = np.array(strung).T

        return str(np_array)


class ValueIterationPolicy(object):

    def __init__(self, environment, gamma=.9, epsilon=.1, max_iterations=100):
        self.environment = environment
        self.environment.init_board()
        self.grid = [ [ "" for _ in range(self.environment.grid_size)] for _ in range(self.environment.grid_size)]
        self.gamma= gamma
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def value_iterate(self):
        for i in range(self.max_iterations):
            # for all states
            for state in self.environment:
                # for all actions
                q_values = []
                if not (state.pos[0] == self.environment.destination[0] and state.pos[1] == self.environment.destination[1]):
                    for action, new_state in state:
                        # sum of all [ probability of action * (reward + prev_v) ]
                        q = sum([ prob * (self.gamma * state_prime.prev_value) for prob, _, state_prime, _ in action])
                        # print(q)
                        q_values.append(q)
                    # update
                    state.value = state.reward + max(q_values)
                else:
                    state.value = state.prev_value
            # print(self.environment)
            if self.environment.average_change() <= self.epsilon:
                print("Converged early in {} iterations".format(i))
                break
            self.environment.reset()
        if DEBUG:
            print(self.environment)


    def policy_extract(self):
        for state in self.environment:
            actions, _ = next(state)
            to_max = [ (state_prime.value, action[2]) for _, _, state_prime, action in actions if state != state_prime]
            max_opt = max(to_max, key= lambda a: a[0])
            self.grid[state.pos[0]][state.pos[1]] = max_opt[1]

        for obs in self.environment.obstacles:
            self.grid[obs[0]][obs[1]] = self.environment.Symbol.OBSTACLE.value

        self.grid[self.environment.destination[0]][self.environment.destination[1]] = self.environment.Symbol.DESTINATION.value

        if DEBUG:
            print(self)

    def write(self, file_name):
        t_grid = np.array(self.grid).T.tolist()
        out_array = [ "".join(r) for r in t_grid ]
        out_str = "\n".join(out_array) + "\n"

        with open(file_name, "w") as f:
            f.write(out_str)
            
    # its okay to cast and print because uncommon operation used for debugging
    def __str__(self):
        # for pretty printing
        # coord tuples are (col, row) so take transpose
        np_array = np.array(self.grid)
        return str(np_array.T)

def main():
    environment = Environment("input.txt")

    vip = ValueIterationPolicy(environment, max_iterations=100)
    vip.value_iterate()
    vip.policy_extract()
    
    vip.write("output.txt")

if __name__ == "__main__":
    main()