from transformers import AutoTokenizer, LlamaForCausalLM

import torch
import torch.nn.functional as F

from prompts import value

model = LlamaForCausalLM.from_pretrained("/juice5/scr5/nlp/llama-2-hf-latest/Llama-2-7b-chat-hf", use_cache=True, low_cpu_mem_usage=True, device_map = 'cuda')
tokenizer = AutoTokenizer.from_pretrained("/juice5/scr5/nlp/llama-2-hf-latest/Llama-2-7b-chat-hf")

# grab ids for "sure", "likely", and "impossible"
TOK_SURE = tokenizer.encode("sure")[-1]
TOK_LIKELY = tokenizer.encode("likely")[-1]
TOK_IMPOSSIBLE = tokenizer.encode("impossible")[-1]

TOK_ONE = tokenizer.encode("C")[-1]
TOK_TWO = tokenizer.encode("K")[-1]


while True:
    # premise = input("(premise) > ").strip()
    # statement = input("(statement) > ").strip()
    # judgement = input("(sure/likely/impossible) > ").strip()
    task = input("(task) > ").strip()
    step = "-"
    steps = []
    while step != "":
        step = input("(step) > ").strip()
        if step != "":
            steps.append(step)

    # premise = "I'm very athletic."
    # statement = "I'm very athletic."
    # judgement = "impossible"
    # hypothesis = "I'm going to climb to the top of sparkle montain."
    
    # fill in value prompt template
    prompt = value(task, steps)

    # sample output distribution
    inputs = tokenizer(prompt.strip(), return_tensors="pt")
    output = model(inputs.input_ids.cuda())

    # grab the last one
    distr = output["logits"][0][-1]

    # grab the distributions for each class
    probs = F.softmax(torch.tensor([
        distr[TOK_SURE], distr[TOK_LIKELY], distr[TOK_IMPOSSIBLE]
        ]), dim=0)

    print("(prompting) > ", ["sure", "likely", "impossible"][torch.argmax(probs).item()])
    # print("(desired) > ", tokenizer.decode(torch.argmax(distr).item()))
    print(probs)

    # res = tokenizer.batch_decode(model.generate(inputs.input_ids.cuda(), max_new_tokens=50, do_sample=False, temperature=0))[0] 
            # skip_special_tokens=true, clean_up_tokenization_spaces=false)[0][len(prompt)-1:]
    # print(res)
    breakpoint()

