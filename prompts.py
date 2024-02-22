def propose(task):
    return f"""
Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {task}
Possible next steps:
""".strip() 

def value(task, steps):
    steps = "\n".join(steps)
    return f"""
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
1 5 5
1 * 5 * 5 = 24
Judge: impossible
{task}
{steps}
Judge: """.strip()+" "

# added the last one as an example of hallucination
