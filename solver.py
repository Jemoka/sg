from julia.api import Julia
# jl = Julia(compiled_modules=False)

from quickpomdps import QuickPOMDP

from julia.Main import Float64
from julia.POMDPs import solve, pdf, simulate, initialstate
from julia.QMDP import QMDPSolver

from julia.POMDPTools import ImplicitDistribution, Deterministic, Uniform, SparseCat, HistoryRecorder
from julia.Random import MersenneTwister
from julia.BasicPOMCP import POMCPSolver, action_info
from julia.POMDPSimulators import stepthrough
from julia.SARSOP import SARSOPSolver
from julia.POMCPOW import POMCPOWSolver
from julia.POMDPModels import BabyPOMDP
from julia.NamedTupleTools import namedtuple
from julia.ParticleFilters import BootstrapFilter
from julia.D3Trees import inbrowser, D3Tree

import julia.Main as J

jl = Julia()

# from collections import namedtuple
from copy import deepcopy

import sys
PATH = "/Users/houjun/Documents/Projects/sg/"
sys.path.append(PATH)

from trajectory import *
from model import *
import torch

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

    if s == None:
        # print("GOODYBE!")
        return True

    else:
        return False
    # stopping = len(s.trajectory) > 0 and len(s.trajectory[-1][-1]) == 1

    # return stopping

def generator_weight(s, a, sp, o):
    if s == None or sp == None:
        return 0
    o = get_reasoning(o)
    # get a trajectory
    next_traj = parse_traj(get_traj(sp))
    # make an observation on our current state
    if len(next_traj) == 0:
        return 1/3
    else:
        res = value(sp.problem, next_traj)
        return res.tolist()[["sure", "likely", "impossible"].index(o)]

counter = 0

def generator(s,a,rng):
    global counter
    counter += 1

    # if counter % 10 == 0:
        # print(counter)
    # print("NEXT:", len(s.trajectory))

    # calculate next state
    next_state = None
    obs = None
    # non-submit actions have a default penalty of -1
    rew = -1.0

    # print(a)
    # if we are rolling back, do so
    if a == "rollback":
        # if we try to roll back 
        if len(s.subproblem) == 4:
            next_state = s
        else:
            next_state = rollback(s)
    # otherwise, sample a single thought
    elif a == "continue":
        # if we are out of states, evaluate
        if len(s.subproblem) == 1:
            next_state = None
            traj = parse_traj(get_traj(s))
            rew = reward(s.problem, traj).item()
        else:
            next_state = increment(s)

    # calculate next trajectory
    if next_state and len(next_state.subproblem) != 4:
        # get a possible next trajectory
        next_traj = parse_traj(get_traj(next_state))
        # make an observation on our current state
        res = torch.argmax(value(next_state.problem, next_traj)).item()
        if res == 0:
            obs = seralize_obs(next_state.subproblem, "sure")
        elif res == 1:
            obs = seralize_obs(next_state.subproblem, "likely")
        elif res == 2:
            obs = seralize_obs(next_state.subproblem, "impossible")
    # if we have no trajectory, we have a random observation
    elif next_state:
        next_traj = []
        obs = J.rand(Uniform([seralize_obs(next_state.subproblem, "sure"),
                              seralize_obs(next_state.subproblem, "likely"),
                              seralize_obs(next_state.subproblem, "impossible")]))

    else:
        next_traj = []
        obs = J.rand(Uniform([seralize_obs([], "sure"),
                              seralize_obs([], "likely"),
                              seralize_obs([], "impossible")]))

    # return result. non-submit actions have a penalty of -1
    return namedtuple(["sp", "o", "r"], (next_state, obs, rew))

roll_jl_bridge = jl.eval(f"""
@pyimport importlib.machinery as machinery
loader = machinery.SourceFileLoader("trajectory","{PATH}/trajectory.py")
ro = loader[:load_module]("trajectory").rollout_state

function rollout(pomdp, s, h, steps)
    value = ro(s)
    return value
end

rollout""")


m = QuickPOMDP(
    actions = ["continue", "rollback"],
    # observations = ["sure", "likely", "impossible"],
    obstype = J.String,
    discount = 1.0,
    isterminal = lambda x : x == None,
    # transition = transition,
    # observation = observation,
    # reward = reward_,
    initialstate = ImplicitDistribution(new_problem()),
    gen = generator,
    obs_weight = generator_weight,
    default_action = "rollback"
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
filter = BootstrapFilter(m, 10)
solver = POMCPSolver(max_depth = 15, tree_queries = 500,
                     estimate_value=roll_jl_bridge) #, estimate_value=estimate_value)
# solver = POMCPOWSolver()
planner = solve(solver, m)

# a, info = action_info(planner, ImplicitDistribution(new_problem))
# breakpoint()

# history = HistoryRecorder(max_steps=10)
# hist = simulate(history, m, planner, filter)
# breakpoint()
# b = 

    # # print(a)
    # # if we are rolling back, do so
    # if a == "rollback":
    #     # if we try to roll back 
    #     if len(s.subproblem) == 4:
    #         next_state = s
    #     else:
    #         next_state = rollback(s)
    # # otherwise, sample a single thought
    # elif a == "continue":
    #     # if we are out of states, evaluate
    #     if len(s.subproblem) == 1
    #         next_state = None
    #         traj = parse_traj(get_traj(s))
    #         rew = reward(s.problem, traj).item()
    #     else:
    #         next_state = increment(s)


# r = new_problem()(0)
# while len(r.subproblem) > 1:
#     a, info = action_info(planner, Deterministic(r), tree_in_info=True)
#     if a == "rollback":
#         if len(r.subproblem) == 4:
#             next_state = r
#         else:
#             next_state = rollback(r)
#     elif a == "continue":
#         if len(r.subproblem) == 1:
#             breakpoint()
#         else:
#             r = increment(r)
#     s2 = " | ".join(step_traj(get_traj(r))) if r != None else ""
#     print(f"DID: {a}")
#     if s2 != "":
#         # breakpoint()
#         print(f"GOT: {s2}")

#     counter = 0
#     inbrowser(D3Tree(info["tree"]), "firefox")
#     breakpoint()

# breakpoint()

    # inbrowser(D3Tree(info[:tree], init_expand=3))

while True:
    for (s,sp, a,o,r) in stepthrough(m, planner, filter, "s,sp,a,o,r"):
        # s1 = parse_traj(s.trajectory) if s != None else ""
        s2 = " | ".join(step_traj(get_traj(sp))) if sp != None else ""
        print(f"DID: {a}")
        if s2 != "":
            # breakpoint()
            print(f"GOT: {s2} <{o}>")
            # print(sp.trajectory)
        # print(s,a,o)
    print(parse_traj(get_traj(s)), "|", r)
    breakpoint()

# r = stepthrough(m, policy, "s,a,r,sp,o")
# for i in r:
#     breakpoint()

# policy
# solver = QMDPSolver()
# policy = solve(solver, m)


# # J.rand(ImplicitDistribution(random_state))
