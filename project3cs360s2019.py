#!/usr/bin/python3
from typing import (Dict, 
List, 
Tuple)
import numpy as np

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

            self.grid = [ [ 0 for j in range(self.grid_size)] for i in range(self.grid_size) ]

            for o in self.obstacles:
                self.grid[o[0]][o[1]] = -100
            
            self.grid[self.destination[0]][self.destination[1]] = 100

    
    def __str__(self) -> str:
        # for pretty printing
        # coord tuples are (col, row) so take transpose
        np_array = np.array(self.grid)
        return str(np_array.T)
        

            
class PolicyExtractor(object):

    def __init__(self, environment: Environment):
        self.environment: Environment = environment

    def solve(self):
        pass

    def write(self, file_name: str):
        pass

def main():
    environment = Environment("input.txt")
    
    policy_extractor = PolicyExtractor(environment)
    policy_extractor.solve()

    policy_extractor.write("output.txt")

if __name__ == "__main__":
    main()