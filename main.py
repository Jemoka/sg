from transformers import AutoTokenizer, LlamaForCausalLM

import torch
import torch.nn.functional as F

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
    steps = "\n".join(steps)

    # premise = "I'm very athletic."
    # statement = "I'm very athletic."
    # judgement = "impossible"
    # hypothesis = "I'm going to climb to the top of sparkle montain."
    
    prompt = f"""
Evaluate if given numbers can reach 24 (sure/likely/impossible)
10 14
10 + 14 = 24
Judge: sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
Judge: impossible
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
Judge: sure
4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure
5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
Judge: likely
5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
Judge: likely
10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
Judge: impossible
1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
Judge: impossible
{task}
{steps}
Judge: """

    # sample output distribution
    inputs = tokenizer(prompt.strip(), return_tensors="pt")
    output = model(inputs.input_ids.cuda())

    # sure likely and impossible prompts
    nli_sure = tokenizer(prompt.strip()+"sure", return_tensors="pt").input_ids.cuda()
    nli_likely = tokenizer(prompt.strip()+"likely", return_tensors="pt").input_ids.cuda()
    nli_impossible = tokenizer(prompt.strip()+"impossible", return_tensors="pt").input_ids.cuda()

    output_sure = model(nli_sure, labels=nli_sure).loss
    output_likely = model(nli_likely, labels=nli_likely).loss
    output_impossible = model(nli_impossible, labels=nli_impossible).loss

    # grab the last one
    distr = output["logits"][0][-1]

    # grab the distributions for each class
    probs = F.softmax(torch.tensor([
        distr[TOK_SURE], distr[TOK_LIKELY], distr[TOK_IMPOSSIBLE]
        ]), dim=0)
    probs_nll = F.softmax(torch.tensor([
        output_sure, output_likely, output_impossible
        ]), dim=0)

    print("(prompting) > ", ["sure", "likely", "impossible"][torch.argmax(probs).item()])
    print("(nll) > ", ["sure", "likely", "impossible"][torch.argmax(probs_nll).item()])
    # print("(desired) > ", tokenizer.decode(torch.argmax(distr).item()))
    print(probs)
    print(probs_nll)

    # res = tokenizer.batch_decode(model.generate(inputs.input_ids.cuda(), max_new_tokens=50, do_sample=False, temperature=0))[0] 
            # skip_special_tokens=true, clean_up_tokenization_spaces=false)[0][len(prompt)-1:]
    # print(res)
    breakpoint()


