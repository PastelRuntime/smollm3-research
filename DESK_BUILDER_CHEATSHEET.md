# Desk Builder Cheatsheet

The ONLY chart that exists. You are the author of both columns.

## The translation table (the whole secret)

| Your sentence says... | You write in the slot |
|---|---|
| "I care a lot about X" | `+2` |
| "I care a bit about X" | `+1` |
| "I dislike X" | `-1` |
| "I hate X" | `-2` |
| "I don't care about X" | `0` |

That's it. That's the entire ruleset. The numbers are just your opinion,
translated. Nothing else determines them.

## The contract (the slots)

Within ONE setup, the slot order is fixed and you decide it:

| Slot | 0 | 1 | 2 | 3 |
|------|---|---|---|---|
| Box | female | verb | pronoun | sarah-ness |

Every notepad, every desk, every output obeys the same order. The labels
are arbitrary (a different toy could call slot 0 "color") but the ORDER
is a contract: slot 0 always pairs with slot 0.

## The 3-step process (always works)

1. **Say the sentence.** "I care about [box A], I dislike [box B], I don't care about [box C]."
2. **Translate each box's opinion** using the table above.
3. **Place each number in its box's slot.**

## Worked example

Sentence: *"I love Sarah-ness, I dislike verbs, I don't care about female or pronoun."*

| Box | Opinion | Translation | Slot |
|-----|---------|-------------|------|
| female | don't care | `0` | 0 |
| verb | dislike | `-1` | 1 |
| pronoun | don't care | `0` | 2 |
| sarah-ness | love | `+2` | 3 |

→ `("sarah fan", [0.0, -1.0, 0.0, 2.0])`

## Rule of thumb

Number of non-zero slots = number of opinions the desk holds.
Zeros are "no opinion," not "empty."