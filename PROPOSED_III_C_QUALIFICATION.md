# Proposed qualification for arXiv:2610.37712v1, Section III-C

This note is not an arXiv replacement and has not been submitted anywhere.

## Claim as published

Equation (7) sets the mean-preserving gate to

s = r / mean(r),

so the mean multiplier on a tensor is 1. Section III-C then says that this form does not inflate the cumulative quadratic step-size term because that average multiplier is one.

## What is wrong with that sentence

Mean 1 does not preserve the squared Euclidean norm. Take

r = (1/5, 1), s = r / mean(r) = (1/3, 5/3), u = (1, 1).

Then mean(s) = 1, ||u||^2 = 2, and ||s ⊙ u||^2 = 26/9, which is 13/9 times ||u||^2.

If every |u_i| is equal and s is not constant, the squared norm strictly increases: mean(s^2) > mean(s)^2 = 1, so ||s ⊙ u||^2 = mean(s^2) ||u||^2 > ||u||^2.

## Bound that does preserve the rate

For the shrink gate, s_i is in [r_min, 1], so ||s ⊙ u||^2 ≤ ||u||^2.

For the mean-preserving gate, if r_i is in [r_min, 1] and r_min > 0, then

s_i is in [r_min, 1/r_min],

and

||s ⊙ u||^2 ≤ r_min^{-2} ||u||^2.

A uniform bound of that form keeps an O(sqrt(T)) regret template, with a constant set by the largest gate entry. The constant is not fixed by the mean being one.

The published defaults include r_min = 0.2. The two-coordinate example above is inside that range (r_min = 1/5) and already inflates the squared step by 13/9.

## Suggested replacement

Keep the statement that the mean-preserving gate has mean 1 and that a bounded gate does not change the O(sqrt(T)) order. Delete the claim that mean 1 by itself prevents inflation of the quadratic term. State the inflation factor through ||s||_∞, which is at most 1/r_min when r_i ∈ [r_min, 1].

The accuracy tables are a separate question. This note does not change any reported accuracy.
