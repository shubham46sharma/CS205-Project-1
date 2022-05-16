#!/usr/bin/python

# Shubham Sharma
# CS205 Project 1 (8-Puzzle Problem)

import time
import math
import numpy as np
from heapq import heappush, heappop



#class to define the node and some helper helper functions 
class Node():
    def __init__(self, state, parent, cost, n):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.n = n
        self._left = None
        self._right = None
        self._up = None
        self._down = None


#These 4 functions check if the move is possible or not in their direction and returns the new state after making the movement
    def left(self):
        zero_index = np.where(self.state == 0)
        if zero_index[0][0] % self.n == 0:
            return False
        else:
            temp = self.state[zero_index[0][0] - 1]
            new_state = self.state.copy()
            new_state[zero_index] = temp
            new_state[zero_index[0][0] - 1] = 0
            return new_state, temp

    def right(self):
        zero_index = np.where(self.state == 0)
        if zero_index[0][0] % self.n >= self.n - 1:
            return False
        else:
            temp = self.state[zero_index[0][0] + 1]
            new_state = self.state.copy()
            new_state[zero_index] = temp
            new_state[zero_index[0][0] + 1] = 0
            return new_state, temp

    def up(self):
        zero_index = np.where(self.state == 0)
        if zero_index[0][0] <= self.n - 1:
            return False
        else:
            temp = self.state[zero_index[0][0] - self.n]
            new_state = self.state.copy()
            new_state[zero_index] = temp
            new_state[zero_index[0][0] - self.n] = 0
            return new_state, temp

    def down(self):
        zero_index = np.where(self.state == 0)
        if zero_index[0][0] >= self.n * 2:
            return False
        else:
            temp = self.state[zero_index[0][0] + self.n]
            new_state = self.state.copy()
            new_state[zero_index] = temp
            new_state[zero_index[0][0] + self.n] = 0
            return new_state, temp

    def __lt__(self, other):
        return self.state[0] < other.state[0]

    def print_path(self):
        state_trace = [self.state]
        # Adds the node information as going back up the tree.
        while self.parent:
            self = self.parent
            state_trace.append(self.state)

        # Prints the complete path with depth of the solution.
        depth = 0
        state_trace.pop()
        while state_trace:
            state_to_print = state_trace.pop()
            level = depth +1
            print(" \n Expanding node at depth " , level,"\n")
            array = [[state_to_print[j * self.n + i] for i in range(self.n)] for j in range(self.n)]
            for row in array:
                print(row)
            depth += 1
        print("\n Depth of the solution:",depth)

    def solve(self, goalState, res):
        start = time.time()
        n_value = self.n
        priority_queue = []  # Priority queue to store unvisited nodes wrt to path cost
        heappush(priority_queue, (0, self))
        nodes_processed = 0  # Counter for the number of nodes popped from the queue, to measure the performance of time
        max_size = 1  # maximum number of nodes in queue as an upper limit on the space used
        visited_states = set([])  # to remember which states have been visited
        while priority_queue:

            # updating maximum size of the queue
            if len(priority_queue) > max_size:
                max_size = len(priority_queue)

            tc, currentNode = heappop(priority_queue)  # This will give us the node with least cost.
            nodes_processed += 1
            visited_states.add(tuple(currentNode.state))  # avoid repeated states

            # when the goal state is found, trace back to the root node and print out the path
            if np.array_equal(currentNode.state, goalState):
                currentNode.print_path()
                print('\n Total number of nodes expanded:', str(nodes_processed))
                print('\n Maximum Number of nodes in queue (Space occupied):', str(max_size))
                end = time.time()
                time_elapsed = round(end - start, 3)
                print('\n Total Time spend to find solution:', time_elapsed,' seconds')
                return True

            else:
                # check if right move is valid
                if currentNode.right():
                    new_state, x = currentNode.right()
                    # check if the resulting node is already visited
                    if tuple(new_state) not in visited_states:
                        # create a new child node
                        if res == 1:
                            h_n = 0
                        elif res == 2:
                            h_n = tile_displacement(new_state, goalState)
                        else:
                            h_n = manhattan_distance(new_state, goalState)
                        currentNode._right = Node(state=new_state, parent=currentNode,
                                                          cost=tc + h_n + 1, n=n_value)
                        heappush(priority_queue, (tc + h_n+1, currentNode._right))

                # check if left move is valid
                if currentNode.left():
                    new_state, x = currentNode.left()
                    # check if the resulting node is already visited
                    if tuple(new_state) not in visited_states:
                        if res == 1:
                            h_n = 0
                        elif res == 2:
                            h_n = tile_displacement(new_state, goalState)
                        else:
                            h_n = manhattan_distance(new_state, goalState)
                        # create a new child node
                        currentNode._left = Node(state=new_state, parent=currentNode,
                                                         cost=tc + h_n + 1, n=n_value)
                        heappush(priority_queue, (tc + h_n+1, currentNode._left))

                # check if down move is valid
                if currentNode.down():
                    new_state, x = currentNode.down()
                    # check if the resulting node is already visited
                    if tuple(new_state) not in visited_states:
                        if res == 1:
                            h_n = 0
                        elif res == 3:
                            h_n = tile_displacement(new_state, goalState)
                        else:
                            h_n = manhattan_distance(new_state, goalState)
                        # create a new child node
                        currentNode._down = Node(state=new_state, parent=currentNode,
                                                         cost=tc + h_n + 1, n=n_value)
                        heappush(priority_queue, (tc + h_n+1, currentNode._down))

                # check if up move is valid
                if currentNode.up():
                    new_state, x = currentNode.up()
                    # check if the resulting node is already visited
                    if tuple(new_state) not in visited_states:
                        if res == 1:
                            h_n = 0
                        elif res == 3:
                            h_n = tile_displacement(new_state, goalState)
                        else:
                            h_n = manhattan_distance(new_state, goalState)
                        # create a new child node
                        currentNode._up = Node(state=new_state, parent=currentNode,
                                                       cost=tc + h_n + 1, n=n_value)
                        heappush(priority_queue, (tc + h_n+1, currentNode._up))





# returns h(n): count of total displaced tiles to reach the goal state
def tile_displacement(new_state, goal_state):
    h_n = np.sum(
        new_state != goal_state) - 1  # Remove 1 for the blank tile.
    if h_n > 0:
        return h_n
    else:
        return 0  # If its the goal state means all are at correct position


# returns h(n): sum of Manhattan distance to reach the goal state
def manhattan_distance(new_state, goal_state):
    distance = 0
    n = len(new_state)
    c = int(math.sqrt(n))
    for i in range(1, n + 1):
        first = np.where(new_state == 0)
        second = np.where(goal_state == 0)
        fx = first[0][0] % c
        fy = first[0][0] / c
        sx = second[0][0] % c
        sy = second[0][0] / c
        distance += abs(fx - sx) + abs(fy - sy)
    return distance


def Execute(input_state, goalState):
     print("\n Enter your choice of algorithm: \n"
     "1. Uniform Cost Search \n2. A* with Misplaced Tile heuristics \n"
     "3. A* with Manhattan distance heuristics \n ")
     res = int(input())
          
     if res == 1:
          print("\n Initial State: \n ")
          array = [[input_state[j * 3 + i] for i in range(3)] for j in range(3)]
          for row in array:
               print(row)
          root_node = Node(state=input_state, parent=None, cost=0, n=3)
          root_node.solve(goalState, 1)
     elif res == 2:
          print("\n Initial state: \n ")
          array = [[input_state[j * 3 + i] for i in range(3)] for j in range(3)]
          for row in array:
               print(row)
          root_node = Node(state=input_state, parent=None, cost=0, n=3)
          root_node.solve(goalState, 2)
     else:
          print("\n Initial state: \n ")
          array = [[input_state[j * 3 + i] for i in range(3)] for j in range(3)]
          for row in array:
               print(row)
          root_node = Node(state=input_state, parent=None, cost=0, n=3)
          root_node.solve(goalState, 3)

def print_state(state):
     array = [[state[j * 3 + i] for i in range(3)] for j in range(3)]
     for row in array:
         print(row)
    
def printMenu():

     print("\n \n Generic puzzle solver \n \n ")

     option = input("\n Enter 1 for creating your own 8 puzzle \n 2 for Default Puzzle \n")

     if option == "1":
          print("Enter your puzzle, use 0 for blank \n \n")
          first_row = input("Enter the first row \n")
          input_row_1 = [int(x) for x in first_row.split()]
          second_row = input("Enter the second row\n")
          input_row_2 = [int(x) for x in second_row.split()]
          third_row = input("Enter the third row\n")
          input_row_3 = [int(x) for x in third_row.split()]
          input_state = input_row_1 + input_row_2 + input_row_3
          input_state = np.array(input_state)
          print("\n Input state: \n ")
          print_state(input_state)
          goal_State = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0])
          print("\n Goal State: \n")
          print_state(goal_State)
          Execute(input_state, goal_State)

     else:
          input_state = np.array([1,2,3,4,5,6,0,7,8])
          print("\n Input state: \n ")
          print_state(input_state)
          goal_State = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0])
          print("\n Goal State: \n")
          print_state(goal_State)
          Execute(input_state, goal_State)

printMenu()