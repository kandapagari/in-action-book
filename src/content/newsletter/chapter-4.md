---
chapter: 4
subject: "The classical planners still running inside modern robots"
---
The action models you've met so far emit numbers. Symbolic ones emit a name and a list of arguments instead, `pick(block_a)` or `pour(cup_1, bowl_2)`, drawn from a finite alphabet the engineer wrote down ahead of time. The field started here, and it's the family every other chapter measures itself against.

Chapter 4 opens with STRIPS, the planner behind Shakey, and shows how its handful of pieces, predicates, action schemas, and the planning problem itself, still underlie basically every symbolic planner since, right up to the PDDL syntax the competitions standardized. Heuristic search is what makes this fast: millions of ground actions solved in seconds, which is why symbolic task layers stay load-bearing even in stacks whose low-level motion comes from a learned policy. The chapter is blunt about the limits too. An action schema is not a controller, and it drags along the frame and grounding problems that motivate much of the book. Knowing what a symbolic action can and can't promise is what lets you read SayCan or task-and-motion planning as engineering rather than magic. The full treatment is on the site.
