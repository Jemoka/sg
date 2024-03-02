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
from julia.NamedTupleTools import namedtuple

import julia.Main as J

# from collections import namedtuple
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

# N_THOUGHTS = 1


# # def transition(s, a):
# def observation(s, a, sp):
#     # if you did nothing. uhh
#     if len(sp.trajectory) == 0:
#         return Uniform(["sure", "likely", "impossible"])

#     parsed = parse_traj(sp.trajectory)
#     res = value(sp.problem, [parsed])

#     return SparseCat(["sure", "likely", "impossible"], res.tolist())

# # because reward() is already our lm reward
# def reward_(s, a, sp):
#     # if you did nothing. uhh
#     if len(sp.trajectory) == 0:
#         return -5
#     parsed = 
#     return reward(sp.problem, parse_traj(sp.trajectory))

# def estimate_value(o, p, op, s, sp, opp, spp, oppp, i):
#     breakpoint()

def stop(s):
    stopping = len(s.trajectory) > 0 and len(s.trajectory[-1][-1]) == 1

    return stopping

def generator(s,a,rng):
    print("NEXT:", len(s.trajectory))

    # calculate next state
    next_state = None
    rew = None
    obs = None

    # if we are rolling back, do so
    if a == "rollback":
        print("ROLLBACK!")
        ns = State(
            problem=s.problem,
            trajectory=s.trajectory[:-1]
        )
        next_state = ns
    # otherwise, sample a single thought
    elif a == "continue":
        print("CONTINUE!")
        problem = (" ".join([str(i) for i in s.trajectory[-1][-1]])
                   if len(s.trajectory) > 0 else s.problem)

        thought = None

        while not thought:
            sp = deepcopy(s)
            thought = parse_thought(think(problem))
            if thought: 
                sp.trajectory.append(thought)
                next_state = sp

    # calculate next trajectory
    next_traj = parse_traj(next_state.trajectory)

    # if next state has nothing, we are sad
    if len(next_state.trajectory) == 0:
        rew = -5
    else:
    # calculate reward
        rew = reward(next_state.problem, next_traj)

    # make an observation on our current state
    if len(next_state.trajectory) == 0:
        obs = J.rand(Uniform(["sure", "likely", "impossible"]))
    else:
        res = value(next_state.problem, [next_traj])
        obs = J.rand(SparseCat(["sure", "likely", "impossible"], res.tolist()))

    # g = generation()
    return namedtuple(["sp", "o", "r"], (next_state, obs, rew))

m = QuickPOMDP(
    actions = ["continue", "rollback"],
    observations = ["sure", "likely", "impossible"],
    discount = 0.5,
    isterminal = stop,
    # transition = transition,
    # observation = observation,
    # reward = reward_,
    initialstate = ImplicitDistribution(new_problem),
    gen = generator
    # gen = function (s, a, rng)
    #     x, v = s
    #     vp = v + a*0.001 + cos(3*x)*-0.0025 + 0.0002*randn(rng)
    #     vp = clamp(vp, -0.07, 0.07)
    #     xp = x + vp
    #     if xp > 0.5
    #         r = 100.0
    #     else
    #         r = -1.0
    #     end
    #     o = xp + 0.15*randn(rng)
    #     return (sp=(xp, vp), o=o, r=r)
    # end,
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
