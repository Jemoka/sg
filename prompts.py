# 8 - 2 = 6 (left: 6 8 14)
# 14 + 2 = 16 (left: 8 8 16)
# 14 - 2 = 12 (left: 8 8 12)
# 14 - 8 = 6 (left: 2 6 8)
# 14 / 2 = 7 (left: 7 8 8)
# 16 - 8 = 8 (left: 2 8 14)
# 8 + 14 = 22 (left: 2 8 22)

# 14 - 8 = 6 (left: 2 6)
# 14 * 2 = 28 (left: 28 8)
# 14 - 2 = 12 (left: 8 12)
# 2 + 8 = 10 (left: 10 14)
# 14 / 2 = 7 (left: 7 8)
# 16 - 8 = 8 (left: 2 8)
# 14 + 2 = 16 (left: 16 8)
# 2 * 14 = 28 (left: 28 8)
# 8 + 14 = 22 (left: 2 22)

def propose(task, previous, n):
    steps = "\n".join(previous)
    if len(steps) > 0:
        steps = "\n"+steps
    return f"""
Provide at exactly {str(n)} possible next steps for a given input. Do not divide indivisible numbers. Do *NOT* use negative numbers. Follow the output format exactly. Do not say extra words. 

Input: 2 8 8 14
Possible next steps:
2 * 8 = 16 (left: 8 14 16)
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
[...repeat until {str(n)} responses]

Input: 2 8 14
Possible next steps:
2 * 8 = 16 (left: 14 16)
[...repeat until {str(n)} responses]


Input: {task}
Possible next steps:{steps}
""".strip() 

def value(task, steps):
    steps = "\n".join(steps)
    return f"""
Evaluate if the given numbers can be combined to reach 24 (sure/likely/impossible) using each number exactly once. Reply with ONLY the word sure, likely, or impossible.

problem: 10 14
10 + 14 = 24
judge: 'sure'
problem: 5 7 8
8 - 5 = 3
3 * 7 = 21
judge: 'likely'
problem: 11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
judge: 'impossible'
problem: 4 4 10
10 - 4 = 6
6 * 4 = 24
judge: 'sure'
problem: 4 9 11
9 + 11 = 20
20 + 4 = 24
judge: 'sure'
problem: 5 6 6
6 - 5 = 1
1 * 6 = 6
judge: 'likely'
problem: 10 10 11
11 - 10 = 1
1 * 10 = 10
judge: 'impossible'
problem: 1 3 3
1 + 3 = 4
4 * 3 = 12
judge: 'impossible'
problem: 1 5 5
1 * 5 = 5
5 * 5 = 24
judge: 'impossible'
problem: 23
judge: 'impossible'
problem: 24
judge: 'sure'
problem: {task}
{steps}
judge: '""".strip()

# added the last one as an example of hallucination


def reward(task, solution):
    return f"""Check if the expression is valid AND result in the number 24. Reply with 'sure' or 'impossible'.

input: 4 4 6 8
answer: (4 + 8) * (6 - 4) = 24
judge: 'sure'
input: 2 9 10 12
answer: 2 * 12 * (10 - 9) = 24
judge: 'sure'
input: 4 9 10 13
answer: (13 - 9) * (10 - 4) = 24
judge: 'sure'
input: 4 4 6 8
answer: (4 + 8) * (6 - 4) + 1 = 25
judge: 'impossible'
input: 2 9 10 12
answer: 2 * (12 - 10) = 24
judge: 'impossible'
input: 12 12 5 9
answer: (12 + 12) * 5 + 9 = 129
judge: 'impossible'
input: 4 9 10 13
answer: (13 - 4) * (10 - 9) = 24
judge: 'impossible'
input: {task}
answer: {solution}
judge: '""".strip()
