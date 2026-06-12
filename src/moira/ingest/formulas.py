from __future__ import annotations

import re
from collections import defaultdict

_TOKEN = re.compile(
    r"""
    (?P<elem>[A-Z][a-z]?) |
    (?P<num>\d+)          |
    (?P<lpar>\()          |
    (?P<rpar>\))          |
    (?P<other>.)
""",
    re.VERBOSE,
)


def formula_to_composition(formula: str) -> dict[str, int]:
    s = formula.strip()
    s = s.replace("*", "")
    s = re.sub(r"[\s\-_]", "", s)

    stack: list[defaultdict[str, int]] = [defaultdict(int)]
    tokens = list(_TOKEN.finditer(s))
    i = 0
    while i < len(tokens):
        token = tokens[i]
        kind = token.lastgroup
        value = token.group()

        if kind == "elem":
            i += 1
            count = 1
            if i < len(tokens) and tokens[i].lastgroup == "num":
                count = int(tokens[i].group())
                i += 1
            stack[-1][value] += count
            continue

        if kind == "lpar":
            stack.append(defaultdict(int))
            i += 1
            continue

        if kind == "rpar":
            if len(stack) == 1:
                raise ValueError(f"Unbalanced parentheses in formula '{formula}'")
            group = stack.pop()
            i += 1
            multiplier = 1
            if i < len(tokens) and tokens[i].lastgroup == "num":
                multiplier = int(tokens[i].group())
                i += 1
            for element, count in group.items():
                stack[-1][element] += count * multiplier
            continue

        if kind == "num":
            raise ValueError(f"Unexpected number '{value}' in formula '{formula}'")

        i += 1

    if len(stack) != 1:
        raise ValueError(f"Unbalanced parentheses in formula '{formula}'")

    return dict(stack[0])
