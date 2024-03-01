from julia.api import Julia
# jl = Julia(compiled_modules=False)

from quickpomdps import QuickPOMDP

from julia.Main import Float64
from julia.POMDPs import solve, pdf, simulate
from julia.QMDP import QMDPSolver

from julia.POMDPTools import ImplicitDistribution, Deterministic, Uniform, SparseCat, HistoryRecorder
from julia.Random import MersenneTwister
from julia.BasicPOMCP import POMCPSolver, action_info
from julia.POMDPSimulators import stepthrough
from julia.SARSOP import SARSOPSolver
from julia.POMCPOW import POMCPOWSolver
from julia.POMDPModels import BabyPOMDP

import julia.Main as J

from copy import deepcopy

import sys
sys.path.append("/Users/houjun/Documents/Projects/sg/")

from trajectory import *
from model import *

# rng = MersenneTwister(1)
# rng()
# J.rand(rng)

# i = ImplicitDistribution(lambda x:5)
# J.rand(i)

# J.rand(ImplicitDistribution(new_problem))

N_THOUGHTS = 1


def transition(s, a):
    print("NEXT:", len(s.trajectory))
    # we are done, new problem
        # return ImplicitDistribution(new_problem)

    # otherwise:
    if a == "rollback":
        print("ROLLBACK!")
        ns = State(
            problem=s.problem,
            trajectory=s.trajectory[:-1]
        )
        return Deterministic(ns)
    elif a == "continue":
        print("CONTINUE!")
        thoughts = []
        problem = (" ".join([str(i) for i in s.trajectory[-1][-1]])
                   if len(s.trajectory) > 0 else s.problem)

        for i in range(N_THOUGHTS):
            sp = deepcopy(s)
            thought = parse_thought(think(problem))
            if thought: 
                sp.trajectory.append(thought)
                thoughts.append(sp)

        if len(thoughts) == 0:
            return s

        return Uniform(thoughts)

def observation(s, a, sp):
    # if you did nothing. uhh
    if len(sp.trajectory) == 0:
        return Uniform(["sure", "likely", "impossible"])

    parsed = parse_traj(sp.trajectory)
    res = value(sp.problem, [parsed])

    return SparseCat(["sure", "likely", "impossible"], res.tolist())

# because reward() is already our lm reward
def reward_(s, a, sp):
    # if you did nothing. uhh
    if len(sp.trajectory) == 0:
        return -5
    parsed = parse_traj(sp.trajectory)
    return reward(sp.problem, parsed)

# def estimate_value(o, p, op, s, sp, opp, spp, oppp, i):
#     breakpoint()

def stop(s):
    stopping = len(s.trajectory) > 0 and len(s.trajectory[-1][-1]) == 1

    return stopping

m = QuickPOMDP(
    actions = ["continue", "rollback"],
    observations = ["sure", "likely", "impossible"],
    discount = 0.5,
    isterminal = stop,
    transition = transition,
    observation = observation,
    reward = reward_,
    initialstate = ImplicitDistribution(new_problem),
)

# estimate_value(::PyObject,
#                ::QuickPOMDPs.QuickPOMDP{UUID("014127b5-0e53-412d-9305-454dade5f751"),
#                 PyObject,
#                 String,
#                 String,
#                 NamedTuple{(:isterminal, :actionindex, :transition, :reward, :actions, :observations, :discount, :initialstate, :obsindex, :observation), Tuple{PyCall.var"#fn#26"{PyCall.var"#fn#25#27"{PyObject}}, Dict{String, Int64}, PyCall.var"#fn#26"{PyCall.var"#fn#25#27"{PyObject}}, Main.PyQuickPOMDPs.var"#85#reward_pyfunc_closure#16"{PyObject}, Vector{String}, Vector{String}, Float64, POMDPTools.POMDPDistributions.ImplicitDistribution{PyObject, Tuple{}}, Dict{String, Int64}, Main.PyQuickPOMDPs.var"#53#observation_pyfunc_closure#10"{PyObject}}}},
#                ::PyObject,
#                ::BasicPOMCP.POMCPObsNode{String, String},
#                ::Int64)

# m

# m = BabyPOMDP()
# ImplicitDistribution(random_state)
# solver = SARSOPSolver()
solver = POMCPSolver(max_depth = 3, tree_queries = 6) #, estimate_value=estimate_value)
planner = solve(solver, m)

# a, info = action_info(planner, ImplicitDistribution(new_problem))
# breakpoint()

# history = HistoryRecorder(max_steps=10)
# hist = simulate(history, m, policy)
# breakpoint()
# b = 

for (s,a,o) in stepthrough(m, planner, "s,a,o"):
    print(s,a,o)
    breakpoint()

# r = stepthrough(m, policy, "s,a,r,sp,o")
# for i in r:
#     breakpoint()

# policy
# solver = QMDPSolver()
# policy = solve(solver, m)


# # J.rand(ImplicitDistribution(random_state))
