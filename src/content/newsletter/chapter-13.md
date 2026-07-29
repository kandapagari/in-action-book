---
chapter: 13
subject: "π₀ and the case for continuous action heads"
---
Discrete action tokens let a transformer treat control as next-token prediction and reuse a web-pretrained backbone at no cost. All true, and it's exactly why RT-1, RT-2, and OpenVLA are built the way they are. Chapter 13 is about the tasks where that same choice quits paying and starts costing, and about π₀, the model that answered by tossing the token head and regressing continuous actions under a flow-matching objective.

The chapter is exact about the ceiling it's after. Tokenization carries three costs, and only the first, a resolution ceiling, is obvious. The other two cut deeper: sequence length that explodes on high-frequency, long-horizon tasks until the policy can't learn at all, and a serial-decode latency that's structural, since autoregression won't parallelize away. Worse, all three bite at once on precisely the fast, dexterous, multi-stage jobs a useful home or factory robot has to do, with shirt-folding as the running example. You can't regress your way out naively either, because that marches right back into the averaging trap. Wanting an action head that's expressive, multimodal, continuous, and still fast is what drives π₀ to flow matching, and the chapter takes the architecture apart end to end. Read it on the site.
