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
from julia.ParticleFilters import BootstrapFilter

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
    # get a trajectory
    next_traj = step_traj(sp.trajectory)
    # make an observation on our current state
    if len(sp.trajectory) == 0:
        return 1/3
    else:
        res = value(sp.problem, next_traj)
        return res.tolist()[["sure", "likely", "impossible"].index(o)]

def generator(s,a,rng):
    # print("NEXT:", len(s.trajectory))

    # calculate next state
    next_state = None
    rew = 0
    obs = None

    # print(a)
    # if we are rolling back, do so
    if a == "rollback":
        # if we try to roll back 
        ns = State(
            problem=s.problem,
            trajectory=s.trajectory[:-1]
        )
        next_state = ns
    # otherwise, sample a single thought
    elif a == "continue":
        problem = (" ".join([str(i) for i in s.trajectory[-1][-1]])
                   if len(s.trajectory) > 0 else s.problem)

        thought = None

        while not thought:
            sp = deepcopy(s)
            thought = parse_thought(think(problem))
            if thought: 
                sp.trajectory.append(thought)
                next_state = sp
    # if we are submitting, evaluate and return
    elif a == "submit":
        sp = None
        # if we are at a good stopping point
        is_stopping = len(s.trajectory) > 0 and len(s.trajectory[-1][-1]) == 1
        if not is_stopping:
            traj = step_traj(s.trajectory)
            # res = value(s.problem, traj)
            rew += reward(s.problem, traj).item()

            # mode said sure
            # if torch.argmax(res).item() == 0:
            #     rew = 10
            # # mode said impossible
            # elif torch.argmax(res).item() == 2:
            #     rew = -10
        else:
            traj = ""
            res = 0
            rew = 0
        # otherwise, punish model
        return namedtuple(["sp", "o", "r"], (None,
                                             J.rand(Uniform(["sure", "likely", "impossible"])),
                                             -100 if not is_stopping else rew*10))

    # calculate next trajectory
    if len(next_state.trajectory) != 0:
        next_traj = step_traj(next_state.trajectory)
    else:
        next_traj = []

    # # if next state has nothing, we are sad
    if len(next_state.trajectory) == 0:
        rew -= 1
    # else:
    # # calculate reward
    #     rew = reward(next_state.problem, next_traj)

    # make an observation on our current state
    if len(next_state.trajectory) == 0:
        obs = J.rand(Uniform(["sure", "likely", "impossible"]))
    else:
        res = value(next_state.problem, next_traj)
        obs = J.rand(SparseCat(["sure", "likely", "impossible"], res.tolist()))

        # mode said sure
        # if torch.argmax(res).item() == 0:
        #     rew += 1
        # elif torch.argmax(res).item() == 2:
        #     rew -= 1

    # g = generation()
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
    actions = ["continue", "rollback", "submit"],
    observations = ["sure", "likely", "impossible"],
    discount = 0.5,
    isterminal = lambda x : x == None,
    # transition = transition,
    # observation = observation,
    # reward = reward_,
    initialstate = ImplicitDistribution(new_problem()),
    gen = generator,
    obs_weight = generator_weight,
    default_action = "submit"
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
solver = POMCPSolver(max_depth = 3, tree_queries = 3, estimate_value=roll_jl_bridge) #, estimate_value=estimate_value)
# solver = POMCPOWSolver()
planner = solve(solver, m)

# a, info = action_info(planner, ImplicitDistribution(new_problem))
# breakpoint()

# history = HistoryRecorder(max_steps=10)
# hist = simulate(history, m, planner, filter)
# breakpoint()
# b = 

while True:
    for (s,sp, a,o,r) in stepthrough(m, planner, filter, "s,sp,a,o,r"):
        # s1 = parse_traj(s.trajectory) if s != None else ""
        s2 = " | ".join(step_traj(sp.trajectory)) if sp != None else ""
        print(f"DID: {a}")
        if s2 != "":
            # breakpoint()
            print(f"GOT: {s2} <{o}>")
            # print(sp.trajectory)
        # print(s,a,o)
    print(parse_traj(s.trajectory), r)
    breakpoint()

# r = stepthrough(m, policy, "s,a,r,sp,o")
# for i in r:
#     breakpoint()

# policy
# solver = QMDPSolver()
# policy = solve(solver, m)


# # J.rand(ImplicitDistribution(random_state))
