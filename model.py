from transformers import AutoTokenizer, LlamaForCausalLM

import math
import torch
import torch.nn.functional as F

from prompts import propose as P
from prompts import value as V
from prompts import reward as R
from oai import prompt_for_completion, prompt_for_next

# model = LlamaForCausalLM.from_pretrained("/juice5/scr5/nlp/llama-2-hf-latest/Llama-2-7b-chat-hf", use_cache=True, low_cpu_mem_usage=True, device_map = 'cuda')
# tokenizer = AutoTokenizer.from_pretrained("/juice5/scr5/nlp/llama-2-hf-latest/Llama-2-7b-chat-hf")

# generate thoughts
def think(task):
    # fill in value prompt template
    prompt = P(task)

    output = [i.strip() for i in 
              prompt_for_completion(prompt).choices[0].message.content.split("\n")]

    return output[0]

# generate a value 
def value(task, steps):
    # fill in value prompt template
    prompt = V(task, steps)

    # sample output distribution
    # inputs = tokenizer(prompt.strip(), return_tensors="pt")
    output = prompt_for_next(prompt).choices[0].logprobs.content[0].top_logprobs

    # calculate probablitiies
    dist_sure = 0
    dist_likely = 0
    dist_impossible = 0

    for i in output:
        if i.token == "sure":
            dist_sure = math.exp(i.logprob)
        # im is apparently a token
        elif i.token == "im":
            dist_impossible = math.exp(i.logprob)
        elif i.token == "likely":
            dist_likely = math.exp(i.logprob)

    # grab the distributions for each class and rescale
    try:
        probs = F.softmax(torch.tensor([
            dist_sure, dist_likely, dist_impossible
            ]), dim=0)
    except RuntimeError:
        breakpoint()

    return probs

# perform reward scoring
def reward(task, solution):
    # fill in value prompt template
    prompt = R(task, solution)

    # sample output distribution
    # inputs = tokenizer(prompt.strip(), return_tensors="pt")
    output = prompt_for_next(prompt).choices[0].logprobs.content[0].top_logprobs

    # calculate probablitiies
    dist_sure = 0
    dist_impossible = 0

    for i in output:
        if i.token == "sure":
            dist_sure = math.exp(i.logprob)
        # im is apparently a token
        elif i.token == "im":
            dist_impossible = math.exp(i.logprob)

    # grab the distributions for each class and rescale
    probs = F.softmax(torch.tensor([
        dist_sure, dist_impossible
        ]), dim=0)

    return probs[0]-probs[1]

