from model import think, value, reward
from dataclasses import dataclass
from typing import Union, List, Tuple, Optional
from enum import Enum
from collections import defaultdict
import re
import random
from copy import deepcopy

import csv
import os 
import random


THOUGHT = re.compile(r"([\d +\-*=/]+) \(left:((?: \d+)+)\)")
OP = re.compile(r"(\d+) ?([+\-*/]) ?(\d+)")

@dataclass
class State:
    problem: str
    # trajectory: List[Tuple[str, int, List[int]]]
    subproblem: List[int]
    operation: Optional[Tuple[str, int, List[int]]]
    prev: Optional["State"]

def serialize_state(s):
    return s.problem+"|"+(" ".join(str(i) for i in s.subproblem))

def increment(p):
    nxt = parse_thought(think(" ".join(str(i) for i in p.subproblem)))

    while not nxt:
        nxt = parse_thought(think(" ".join(str(i) for i in p.subproblem)))
        
    ns = deepcopy(p)

    ns.subproblem = nxt[2]
    ns.operation = nxt
    ns.prev = p

    return ns

def rollback(p):
    return p.prev

def get_traj(r):
    trajectory = [r.operation]

    while r.operation != None:
        r = r.prev
        trajectory.append(r.operation)

    return list(reversed(trajectory))[1:]

with open(os.path.join(os.path.abspath(os.path.join(__file__, os.pardir)),
                       "./24.csv")) as df:
    # leave out last 50
    DATA = [i for i in csv.reader(df)][1:][:-50]

def get_problem():
    sample = DATA[random.randint(0, len(DATA)-1)]
    return sample[1]

def new_problem():
    p = get_problem()
    def g(_):
        return State(
            problem=p,
            subproblem=[int(i) for i in p.split(" ")],
            operation=None,
            prev=None
        )

    return g

# utility to parse a thought
def parse_thought(thought):
    try:
        expr, remain = THOUGHT.findall(thought)[0]
    except IndexError:
        return None
    # split out remaining values
    remain = [int(i) for i in remain.strip().split(" ")]
    # split out the current expression
    operation, result = expr.split("=")
    operation = operation.strip()
    result = result.strip()

    return operation, int(result), remain

# parse_thought(think("8 5 2 2"))

# roll state out
def rollout_state(state):
    if state == None:
        return 0

    r = state

    while len(r.subproblem) != 1:
        r = increment(r)

    return reward(state.problem, parse_traj(get_traj(r))).item()

# generate a computation graph from steps
@dataclass
class Node:
    lhs: Union["Node", int]
    rhs: Union["Node", int]
    op: str
    
def graphify_traj(traj):

    available_nodes = {}

    for op, res, _ in traj:
        l,o,r = OP.findall(op)[0]

        n = Node(lhs=int(l),op=o,rhs=int(r))
        if available_nodes.get(n.lhs):
            orig = n.lhs
            n.lhs = available_nodes.get(n.lhs)
            del available_nodes[orig]
        if available_nodes.get(n.rhs):
            orig = n.rhs
            n.rhs = available_nodes.get(n.rhs)
            del available_nodes[orig]

        available_nodes[res] = n

    output = available_nodes[traj[-1][1]]

    return output

# and then stringify the result by dfs
def stringify_graph(graph):
    if isinstance(graph.lhs, Node):
        lhs = f"({stringify_graph(graph.lhs)})"
    else:
        lhs = graph.lhs
    if isinstance(graph.rhs, Node):
        rhs = f"({stringify_graph(graph.rhs)})"
    else:
        rhs = graph.rhs

    return f"{lhs} {graph.op} {rhs}"

# finally, we stringify the whole thing by generating intemediate graphs
def parse_traj(traj):
    if len(traj) == 0:
        # print("PARSE EMPTY TRAJ!!")
        return ""
    oup = traj[-1][1]

    for indx in reversed(range(len(traj))):
        intermediate = stringify_graph(graphify_traj(traj[indx:]))
        oup = f"{intermediate} = {oup}"

    return oup

def step_traj(traj):
    stp = []
    for l, r, _ in traj:
        stp.append(f"{l} = {r}")
    return stp
    

 # r = think("3 9 2 1")
# traj =  [('4 * 5', 20, [8, 9, 20]), ('8 * 9', 72, [20, 72])]
# step_traj(traj)

# graphify_traj(traj)

# r
# parse_thought(r[0])

# res = value("3 9 2 1", [parse_traj(traj)])
# SparseCat(["sure", "likely", "impossible"], res.tolist())

# [f"{i[0] for i in traj]
# parse_traj(traj)

# res, traj = rollout("3 9 2 1")
# traj
# import copy
# traj1 = copy.deepcopy(parse_traj(traj))
# traj2 = copy.deepcopy(parse_traj(traj))
# traj3 = copy.deepcopy(parse_traj(traj))
# traj1

# traj1
# traj2
# reward("3 9 2 1", traj2)
# reward("3 2 1 4", "((3 + 2) + 1) * 4 = (5 + 1) * 4 = 6 * 4 = 24")
        

# from model import value
# value("3 9 2 1", [traj1, traj2, traj3])

# for op, res, _ in reversed(traj):
#     result.append(l)
