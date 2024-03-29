from transformers import AutoTokenizer, LlamaForCausalLM

import math
import torch
import torch.nn.functional as F

from prompts import propose as P
from prompts import value as V
from prompts import reward as R
from oai import prompt_for_completion, prompt_for_next

from collections import defaultdict

import re
import random
from functools import cache

# model = LlamaForCausalLM.from_pretrained("/juice5/scr5/nlp/llama-2-hf-latest/Llama-2-7b-chat-hf", use_cache=True, low_cpu_mem_usage=True, device_map = 'cuda')
# tokenizer = AutoTokenizer.from_pretrained("/juice5/scr5/nlp/llama-2-hf-latest/Llama-2-7b-chat-hf")

THOUGHT = re.compile(r"([\d +\-*=/]+) \(left:((?: -?\d+)+)\)")
THOUGHT_CACHE = defaultdict(list)
VALUE_CACHE = {}
REWARD_CACHE = {}

def list_to_tuple(function):
    def wrapper(*args):
        args = [tuple(x) if isinstance(x, list) else x for x in args]
        result = function(*args)
        result = tuple(result) if isinstance(result, list) else result
        return result
    return wrapper

NUM_THOUGHTS = 5

# generate thoughts
def think(task, prev):
    taskhash = tuple(sorted([int(i) for i in task.split(" ")]))
    existing = [i for i in THOUGHT_CACHE[taskhash] if i not in prev]

    # if we have an unused thought, give it to them
    if len(existing) > 0:
        return existing[0]

    # we need to think get some more
    output = []

    while len(output) == 0:
        # fill in value prompt template
        prompt = P(task, existing, len(existing)+NUM_THOUGHTS)

        output = [i.strip() for i in 
                  prompt_for_completion(prompt).split("\n")
                if i.strip() != ""]

    # for each valid thought, append
    for i in output:
        if THOUGHT.match(i):
            THOUGHT_CACHE[taskhash].append(i)

    # recursively return
    return think(task, prev)


# generate a value 
@list_to_tuple
@cache
def value(task):
    # fill in value prompt template
    cachestring = task

    if not VALUE_CACHE.get(cachestring):
        # get values for task
        thoughts = []
        if not task.isnumeric():
            for i in range(NUM_THOUGHTS):
                thoughts.append(think(task, thoughts))

        parts = []
        for t in thoughts:
            parts.append(t.split("(")[0].strip())
        # try to get cache, otherwise, eval
        # print("EVALUATING", task, "|", steps[-1])
        prompt = V(task, parts)

        # sample output distribution
        # inputs = tokenizer(prompt.strip(), return_tensors="pt")
        output = prompt_for_next(prompt)
        VALUE_CACHE[cachestring] = output

    output = VALUE_CACHE[cachestring]
        
    print(task, output)

    # calculate probablitiies
    dist_sure = 0.001
    dist_likely = 0.001
    dist_impossible = 0.001

    for i,j in output.items():
        if i == "sure" or i == "'sure":
            dist_sure = math.exp(j)
        # im is apparently a token
        elif i == "im" or i == "'im":
            dist_impossible = math.exp(j)
        elif i == "likely" or i == "'likely":
            dist_likely = math.exp(j)


    # grab the distributions for each class and rescale
    try:
        probs = F.softmax(torch.tensor([dist_sure, dist_likely, dist_impossible]),
                          dim=0)
    except RuntimeError:
        breakpoint()

    return probs

# perform reward scoring
@list_to_tuple
@cache
def reward(task, solution):
    ## TODO TODO HACK EXOGENOUS REWARD SANITY CHECK REMOVE ME ##
    # if int(solution.split("=")[-1].strip()) == 24:
    #     return torch.tensor(30.0)
    # else:
    #     return torch.tensor(-30.0)
    ## ### ####

    cachestring = task+solution

    if not REWARD_CACHE.get(cachestring):
        # try to get cache, otherwise, eval
        # print("REWARDING", task, "|", scolution)
        spl = solution.split("=")
        prompt = R(task, f"{spl[0].strip()} = {spl[-1].strip()}")
        output = prompt_for_next(prompt)

        # sample output distribution
        # inputs = tokenizer(prompt.strip(), return_tensors="pt")
        REWARD_CACHE[cachestring] = output

    output = REWARD_CACHE[cachestring]
        

    # fill in value prompt template

    # sample output distribution
    # inputs = tokenizer(prompt.strip(), return_tensors="pt")

    # calculate probablitiies
    dist_sure = 0.001
    dist_impossible = 0.001

    for i,j in output.items():
        if i == "sure":
            dist_sure = math.exp(j)
        # im is apparently a token
        elif i == "im":
            dist_impossible = math.exp(j)

    # # grab the distributions for each class and rescale
    # probs = F.softmax(torch.tensor([
    #     dist_sure, dist_impossible
    #     ]), dim=0)

    return torch.tensor(150.0) if dist_sure  > dist_impossible  else torch.tensor(-30.0)

