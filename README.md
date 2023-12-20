# Programming Assignment 3
(TA : Harsh Shah and Shrey Modi)  
This repository contains code for PA3 of CS747 - Foundations of Learning Agents (Autumn 2023). Clone/download this repository and read instructions over [here](https://www.cse.iitb.ac.in/~shivaram/teaching/cs747-a2023/index.html).

(simulator taken from [here](https://github.com/max-kov/pool))  

##################################################################################################################

By: Archit Swamy

In this assignment, we attempt to build an RL agent that learns how to optimally play pool. 
Disclaimer: The simulations consist entirely of solid balls (only for now, hopefully), and no spin can be applied to the cue ball.

## Preliminaries

An agent performs actions as (v, theta), where v is the speed of the cue ball, and theta is the angle at which the cue ball is shot.
First we ask the question, what makes a good pool-playing agent? Consider the situation below: 

![Alt text](/img/situation_1.png)

Let us initially ignore complicated actions such as rebounds, and potting a solid ball off of a solid ball or "chaining" balls. 
Which solid ball should the cue ball try to pot? Given a solid ball, which hole should the agent try to pot it in? A solid first instinct (hehe) might be to first pot ball 3 in hole 6. Then we ask, what makes ball 3 so much more of an attractive shot that balls 1 and 2? Consider the image below, depicting possible trajectories for the balls:

![Alt text](/img/situation_1_marked.png)

It is easily observed that the angle made by the pair of lines cue to ball 3, and ball 3 to hole, is the greatest of any other angle made by the cue-solid-hole pairs. We can reason that such a positioning makes for a better shot because the shot is **"straighter"** than any other. Also, note that if the angle between the cue-solid-hole lines is lesser than or equal to 90 degrees, then the shot cannot be made.

In order to pot a solid in a hole, it will also be necessary that there is no other ball between the solid and the hole.

In this pool simulator, a normal noise is added to the angle at which the cue ball is shot based on the speed it is being shot at. Therefore, faster and thus longer shots are bound to be less accurate. The agent should also be able to optimise the speed at which the cue ball is shot in order to maximise accuracy.