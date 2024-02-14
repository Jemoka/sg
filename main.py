from transformers import AutoTokenizer, LlamaForCausalLM

import torch
import torch.nn.functional as F

model = LlamaForCausalLM.from_pretrained("/juice5/scr5/nlp/llama-2-hf-latest/Llama-2-7b-chat-hf", load_in_4bit=True, use_cache=True, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("/juice5/scr5/nlp/llama-2-hf-latest/Llama-2-7b-chat-hf")

# grab ids for "sure", "likely", and "impossible"
TOK_SURE = tokenizer.encode("definitely")[-1]
TOK_LIKELY = tokenizer.encode("possible")[-1]
TOK_IMPOSSIBLE = tokenizer.encode("impossible")[-1]

TOK_ONE = tokenizer.encode("Q")[-1]
TOK_TWO = tokenizer.encode("P")[-1]


while True:
    # premise = input("(premise) > ").strip()
    # statement = input("(statement) > ").strip()
    # judgement = input("(sure/likely/impossible) > ").strip()
    premise = input("(premise) > ").strip()
    hypothesis1 = input("(hypothesis1) > ").strip()
    hypothesis2 = input("(hypothesis2) > ").strip()

    # premise = "I'm very athletic."
    # statement = "I'm very athletic."
    # judgement = "impossible"
    # hypothesis = "I'm going to climb to the top of sparkle montain."
    
    prompt = f"""
Judge which of the two statements logically make sense next to each other. 

premise: "I'm very smart"
response A: "I just failed my math test"
response B: "I just aced my math test"

Judge:
response B, because a smart person will ace their math test.

premise: "I love humanities"
response S: "I am an author"
response K: "I am a scientist"

Judge:
response S, because a love of humanities means you will write a lot.

premise: "{premise}"
response Q: "{hypothesis1}"
response P: "{hypothesis2}"

Judge:
response """

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
        distr[TOK_ONE], distr[TOK_TWO]
        ]), dim=0)

    print("> ", ["one", "two"][torch.argmax(probs).item()])
    print("(desired) > ", tokenizer.decode(torch.argmax(distr).item()))
    print(probs)

    # res = tokenizer.batch_decode(generate_ids.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # print(res)
    breakpoint()


