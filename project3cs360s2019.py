#!/usr/bin/python3
from typing import (Dict, 
List, 
Tuple)
import numpy as np
from enum import Enum

class Environment(object):

    def __init__(self, file_name: str):
        # will be overwritten
        self.grid_size: int = 0
        self.num_obstacles: int = 0
        self.obstacles: List[Tuple[int,int]] = []
        self.destination: Tuple[int,int] = (0,0)
        self.grid: List[List[int]] = []
        self.read(file_name)

    def read(self, file_name: str):
        with open(file_name, "r") as f:
            self.grid_size = int(next(f))
            self.num_obstacles = int(next(f))

            env_items: List[Tuple[int,int]] = []
            for i in f:
                row: List[str] = i.split(",")
                coord: Tuple[int,int] = (int(row[0]), int(row[1]))
                env_items.append(coord)

            self.obstacles = env_items[:-1]
            self.destination = env_items[-1]

    def zero_init_board(self):
        self.grid = [ [ 0 for j in range(self.grid_size)] for i in range(self.grid_size) ]

        for o in self.obstacles:
            self.grid[o[0]][o[1]] = -100
        
        self.grid[self.destination[0]][self.destination[1]] = 100
    
    # its okay to cast and print because uncommon operation used for debugging
    def __str__(self) -> str:
        # for pretty printing
        # coord tuples are (col, row) so take transpose
        np_array = np.array(self.grid)
        return str(np_array.T)


class ValueIterationPolicy(object):
    
    class PolicySymbol(Enum):
        OBSTACLE = "o"
        EAST = ">"
        WEST = "<"
        NORTH = "^"
        SOUTH = "v"
        DESTINATION = "."


    def __init__(self, environment: Environment, gamma: float=.9, iterations: int=10):
        self.environment: Environment = environment
        self.environment.zero_init_board()
        self.grid: List[List[str]] = []
        self.gamma: float = gamma
        self.iterations: int = iterations

    def solve(self):
        # for all states
        for i in range(self.iterations):
            pass



    def write(self, file_name: str):
        t_grid = np.array(self.grid).T.tolist()
        out_array = [ "".join(r) for r in t_grid ]
        out_str = "\n".join(out_array)

        with open(file_name, "w") as f:
            f.write(out_str)
            
    # its okay to cast and print because uncommon operation used for debugging
    def __str__(self) -> str:
        # for pretty printing
        # coord tuples are (col, row) so take transpose
        np_array = np.array(self.grid)
        return str(np_array.T)

def main():
    environment = Environment("input.txt")
    environment.zero_init_board()

    policy = ValueIterationPolicy(environment)
    policy.solve()

    policy.write("output.txt")

if __name__ == "__main__":
    main()