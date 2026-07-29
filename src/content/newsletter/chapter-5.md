---
chapter: 5
subject: "Reward, the training signal classical planners never had"
---
A STRIPS planner knows what its actions do only because someone typed the effects in by hand, and it has no way to notice that one plan turned out better than another. That's an empty slot in the book's anatomy of an action model. Chapter 5 fills it with reward, and the object that lets you reason about reward cleanly is the Markov decision process.

Most of the opening pages go to what those five letters actually mean, states, actions, transitions, reward, and a discount factor, because the rest of the chapter and a good chunk of Chapter 7 stay unreadable until you can take in the tuple without translating it. The Markov property, policies, returns, and value functions all get built up on a gridworld small enough to sketch on a napkin. Then comes the harder point: an MDP is a modeling choice, not a fact about the world. That gap between a tidy MDP on paper and a messy robot in a lab is, the chapter argues, responsible for more wasted GPU-hours than any other decision in the field. You'll leave knowing what the formalism buys you and exactly where it charges you for it. Read §5.1 onward on the site.
