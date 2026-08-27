# PRESENCE.md — The visibility engine

> Written 2026-08-26. This file governs the *posting* half of the studio.
> Rule it inherits from NORTH_STAR.md: **one verified claim per post.**

## Positioning (the one sentence)

*Independent researcher doing pre-registered experiments on small models,
free compute only, everything public including the failures.*

That last clause is the moat. Most AI-poster accounts cherry-pick wins.
Pre-registration + negative results + $0 budget is a combination nobody
else is running consistently at this level. Do not dilute it with hot takes
on the news cycle.

## The three content lanes

| Lane | Cadence | Source | Example |
|---|---|---|---|
| **Experiment receipts** | ~2 weeks (whenever an exp closes) | EXPERIMENT_N.md closeouts | Exp1/Exp2 threads |
| **Process grit** | weekly filler | session logs, kernel failures | "8 kernel versions, every failure was the environment" |
| **Reply-guy depth** | as inspired, 3–5x/week target | other people's threads | technical replies on NoPE/SLWA/MoE/Kaggle-quota threads |

The reply lane is how accounts actually grow before they're big. Receipts
give strangers a reason to follow once they click the profile; replies are
how they find the profile at all.

## Cadence mechanics (ADHD-safe)

- One post per experiment closeout. Not zero, not five.
- Session ends → append to SESSION_LOG.md → if anything funny/honest happened,
  drop one process-grit post same day (5 min, no editing).
- Thread > longpost. First line must survive out of context.
- Post the boring numbers. tok/s, GB, quota burned. Specificity reads as credibility.

## Landing page rule

Every experiment thread ends with the repo link. Therefore:

1. [ ] Create real repo (github.com/<user>/smollm3-research), push this project
2. [ ] Pin README = 30-second tour: NORTH_STAR ladder table + result receipts
3. [ ] Only then post the Experiment 2 layman thread

## Queue

- NEXT POST: Experiment 2 layman thread (drafted in sessions/X_PROGRESS_POST_ALWAYS-UPDATE.md,
  sibling X_LAYMAN_OVERVIEW.md) — blocked on the repo link being real
- THEN: kernel-environment horror mini-post (process grit, already written in exp1 draft footnote)
- THEN: Exp 3 pre-registration announcement ("I'm teaching a 3B to interrogate — preregistered, here's what would falsify me")

## Metrics that matter (check monthly, not daily)

Ignore likes. Track: profile visits → follows conversion, DM quality, and
whether anyone cites/runs your kernels. One collaborator beats 1000 likes.
