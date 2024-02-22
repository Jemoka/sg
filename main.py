from transformers import AutoTokenizer, LlamaForCausalLM

import torch
import torch.nn.functional as F

model = LlamaForCausalLM.from_pretrained("/juice5/scr5/nlp/llama-2-hf-latest/Llama-2-7b-chat-hf", use_cache=True, low_cpu_mem_usage=True, device_map = 'cuda')
tokenizer = AutoTokenizer.from_pretrained("/juice5/scr5/nlp/llama-2-hf-latest/Llama-2-7b-chat-hf")

# grab ids for "sure", "likely", and "impossible"
TOK_SURE = tokenizer.encode("sure")[-1]
TOK_LIKELY = tokenizer.encode("possible")[-1]
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
    steps = "\n".join(steps)

    # premise = "I'm very athletic."
    # statement = "I'm very athletic."
    # judgement = "impossible"
    # hypothesis = "I'm going to climb to the top of sparkle montain."
    
    prompt = f"""
Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: sure
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: sure
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: sure
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: impossible
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: impossible
Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: impossible
Input: {task}
Answer: {steps}
Judge: """

    # sample output distribution
    inputs = tokenizer(prompt.strip(), return_tensors="pt")
    output = model(inputs.input_ids.cuda())

    # print(tokenizer.batch_decode(output, 
        # skip_special_tokens=True, 
        # clean_up_tokenization_spaces=True)[0])
    # breakpoint()

    # grab the last one
    distr = output["logits"][0][-1]

    # grab the distributions for each class
    probs = F.softmax(torch.tensor([
        distr[TOK_SURE], distr[TOK_IMPOSSIBLE]
        ]), dim=0)

    print("(prompting) > ", ["one", "two"][torch.argmax(probs).item()])
    # print("(desired) > ", tokenizer.decode(torch.argmax(distr).item()))
    print(probs)

    # res = tokenizer.batch_decode(model.generate(inputs.input_ids.cuda(), max_new_tokens=50, do_sample=False, temperature=0))[0] 
            # skip_special_tokens=true, clean_up_tokenization_spaces=false)[0][len(prompt)-1:]
    # print(res)
    breakpoint()


