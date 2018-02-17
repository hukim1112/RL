

# Reinforcement learning

## 1. Introduction

![marshmallow-girl-large](images/marshmallow-girl-large.jpg)

​									         **marshmallow test**

Reinforcement Learning is seemed like ["*marshmallow test*"](https://en.wikipedia.org/wiki/Stanford_marshmallow_experiment) conducted by Walter Mischel, then a professor at Stanford university. In this experiment, he insisted that the children to wait for larger later reward, will have better outcome at SAT test, educational achievements, Body mass index and so on. Like this, We want our **intelligent agent** to activate in consideration of long term reward. That is the purpose of Reinforcement Learning.

Reinforcement Learning (RL) refers to a kind of Machine Learning method that how software agents ought to take *actions* in an *environment* so as to maximize some notion of cumulative *reward*.

It was mostly used in games (e.g. Atari, Mario), with performance on par with or even exceeding humans. Recently, as the algorithm evolves with the combination of Neural Networks to solve more complex problems.

![RL_tutorial](images/RL_tutorial.png)

Q. How can we model decision making process of agents for a environment which have specific characteristics? That's called **Markov decision process**

## 2. Markov decision process

![Markov_Decision_Process.svg](images/Markov_Decision_Process.svg.png)

---

We need to understand a useful property called **[Markov property](https://en.wikipedia.org/wiki/Markov_property)** . Markov property refers to the memoryless property of a stochastic process.
$$
P[S_{t+1}|S_{t},S_{t-1},S_{t-2},S_{t-3}...S_{2},S_{1}] = P[S_{t+1}|S_{t}]\\
\text{if and only if this case is called Markov}
$$
A stochastic process has the Markov property if the [conditional probability distribution](https://en.wikipedia.org/wiki/Conditional_probability_distribution) of future states of the process (conditional on both past and present values) depends only upon the present state.
$$
\text{suppose that there is a Bayesian chain like below, }\\
P[S_{t+1},S_{t},S_{t-1},S_{t-2},S_{t-3}...S_{2},S_{1}] = P[S_{t+1}|S_{t}, S_{t-1}, ...,S_1]P[S_t|S_{t-1},S_{t-2},...,S_1]...P[S_2|S_1]P[S_1] \\
\text{if this stochastic process is markov, We can represent Markov chain from Bayesian chain like this.} \\ P[S_{t+1}|S_{t}]P[S_{t}|S_{t-1}]...P[S_2|S_1]P[S_1]
$$
**The future state is independent of the past states**, because the present state contains all relevant information from the history. It is used very efficiently in many problems where we can model its environment as Markovian process.

---

**Markov decision processes** (**MDPs**) provide a mathematical framework for modeling [decision making](https://en.wikipedia.org/wiki/Decision_making) in situations where outcomes are partly [random](https://en.wikipedia.org/wiki/Randomness#In_mathematics) and partly under the control of a decision maker. MDPs are useful for studying a wide range of [optimization problems](https://en.wikipedia.org/wiki/Optimization_problem) solved via [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) and [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning). (ref : wikipedia)

![markov decision process](images/markov decision process.png)

#### Definition

1. Action (A): All the possible moves that the agent can take

2. State (S): Current situation returned by the environment. $$S^a_t = f(S^e_t)$$

3. State transition probability matrix. Because our stochastic process is Markovian, probability distribution of future states depends on only a present state.

   $$\large P^{a}_{ss'} = P[S_{t+1} = s' | S_t = s, A_t = a]$$

   ![State transition probability matrix](images/State transition probability matrix.png)

   ​

4. Reward (R): An immediate return send back from the environment to evaluate the last action.

    $$R^{a}_{s} = E[R_{t+1}|S_t = s , A_t = a]$$ 


5. state value function (V): The expected long-term return with discount, as opposed to the short-term reward R. $$V\pi(s)$$ is defined as the expected long-term return of the current state sunder policy π.

   ![state-value-function1](images/state-value-function1.png)

   ![state-value-function2](images/state-value-function2.png)

   **Discounting factor** ( $$0\le \gamma \le 1 $$) : to avoid infinite returns, to represent uncertainty of future reward, to show the preference of immediate reward.

   ![Discounting](images/Discounting.png)

   ​

   To show simple example, Markov reward process(Action is not considered.) is presented below.

   ![Marcov_3](images/Marcov_3.png)

   ![return](images/return.png)

   ![ex1_state-value](images/ex1_state-value.png)

   ![ex2_state-value](images/ex2_state-value.png)

   ​

   But What we use for Reinforcement Learning is Markov decision process. MDP has difference that it's reward function given state depends on agent's action in contrary to MRP.

   ![Student_MDP](images/Student_MDP.png)

   ![Student_MDP2](images/Student_MDP2.png)


5. Q-value or action-value function (Q): Q-value is similar to Value, except that it takes an extra parameter, the current action *a*. $$Q\pi(s,a)$$ refers to the long-term return of the current state *s*, taking action *a* under policy π. **this case in contrast to state-value function, do not need MDP and knowledge(state to state) to know how to get to next state which achieve the optimal reward.** Therefore we use this.

   ![Action-value-function](images/Action-value-function.png)

6. Policy (π): The strategy that the agent employs to determine next action based on th e current state. A policy fully defines the behaviour of an agent. MDP policies depend on the current state(note the history)![policy_definition](images/policy_definition.png)

#### 3. Bellman Equation


A Bellman equation, named after its discoverer, Richard Bellman, also known as dynamic programming equation. This breaks a dynamic optimization problem into simpler subproblems.

1. A dynamic decision problem

   the current payoff from taking $$a$$ in state $$x$$ is $$F(x, a)$$

   ![A dynamic decision problem](images/A dynamic decision problem.png)

2. Bellman's Principle of optimality

   **Principle of Optimality :** An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision. 

   ![Bellman's principle of optimality](images/Bellman's principle of optimality.png)

   Choose the action to be considered with not only current payoff, but also rewards of the following states.  

3. The Bellman equation

   ![Bellman equation](images/Bellman equation.png)

4. state value function (V) with the **Bellman equation**: 

   ![state-value-function1](images/state-value-function1.png)

   ![state-value-function2](images/state-value-function2.png)

   ![Markov_4](images/Markov_4.png)
   $$
    G_{t} = R_{t+1} + \gamma R_{t+2}+\gamma^2 R_{t+3}+... \\
       = R_{t+1} + \gamma\{ R_{t+2}+\gamma R_{t+3}+\gamma^2 R_{t+4}+...\}  \\
       \;\\
       \therefore v(s) = E[G_t|S_t = s]  \\= E[R_{t+1} + \gamma\{ R_{t+2}+\gamma R_{t+3}+\gamma^2 R_{t+4}+..\} | S_t = s] \\
       = E[R_{t+1} + \gamma G_{t+1} | S_t = s] = E[R_{t+1} + \gamma v(S_{t+1})|S_t = s]
   $$
   ![ex3_state-value](images/ex3_state-value.png)

   ![Bellman_equation_matrice](images/Bellman_equation_matrice.png)

   ![solving the Bellman Eqaution](images/solving the Bellman Eqaution.png)



5. Q-value or action-value function (Q): Q-value is similar to Value, except that it takes an extra parameter, the current action *a*. $$Q\pi(s,a)$$ refers to the long-term return of the current state *s*, taking action *a* under policy π. **this case in contrast to state-value function, do not need MDP and knowledge(state to state) to know how to get to next state which achieve the optimal reward.** Therefore we use this.

   ![Action-value-function](images/Action-value-function.png)

6. Policy (π): The strategy that the agent employs to determine next action based on th e current state.   ![policy_definition](images/policy_definition.png)

7. Bellman expectation equation : Combination of Bellman equation and value function


   ![state-value-function2](images/state-value-function2.png)

   ![state-value_tree](images/state-value_tree.png)

   ##### optimal state-value function

$$
G_{t} = R_{t+1} + \gamma R_{t+2}+\gamma^2 R_{t+3}+... \\
   = R_{t+1} + \gamma\{ R_{t+2}+\gamma R_{t+3}+\gamma^2 R_{t+4}+...\}  \\
   \;\\
   \therefore v(s) = E[G_t|S_t = s]  \\= E[R_{t+1} + \gamma\{ R_{t+2}+\gamma R_{t+3}+\gamma^2 R_{t+4}+..\} | S_t = s] \\
   = E[R_{t+1} + \gamma G_{t+1} | S_t = s] = E[R_{t+1} + \gamma v(S_{t+1})|S_t = s]
$$
   			![ex3_state-value](images/ex3_state-value.png)



5. Q-value or action-value function (Q): Q-value is similar to Value, except that it takes an extra parameter, the current action *a*. $$Q\pi(s,a)$$ refers to the long-term return of the current state *s*, taking action *a* under policy π. **this case in contrast to state-value function, do not need MDP and knowledge(state to state) to know how to get to next state which achieve the optimal reward.** Therefore we use this.

   ![Action-value-function](images/Action-value-function.png)

6. Policy (π): The strategy that the agent employs to determine next action based on th e current state.   ![policy_definition](images/policy_definition.png)



$$
\huge\; v_{\pi*}(x) = \max_{\pi} v_\pi(s)
$$

##### rewrite as Q function(action-value function)

![action-value-function_tree](images/action-value-function_tree.png)

##### 	optimal state-value function

$$
\huge q_{\pi*}(s, a) = \max_{\pi}q_\pi (s,a)
$$










#### 2-1. Bellman Expectation equation






































## 2. Q-Learning intro

We are going to explore Q-learning algorithms. Q-learning algorithms are a family of Reinforcement learning. These are a little different with policy-based algorithm. These use look-up table to solve problems.

We will combine Q-learning approaches and policy gradient method to build the state of the art RL agent. This tutorial may be easy and basic but enough to start for going to complex and difficult problems.

both approaches target the goal to choose intelligent action given a situation.

---

**Policy gradient** methods learn a mapping function from an observation to an action( **Observation** $$\rightarrow$$ **Action** ) 

**Q-learning algorithms** learn the value of given state to each action using **look-up tables or networks**.

---

#### 2-1. Dynamic programming

**Dynamic programming breaks a multi-period planning problem into simpler steps at different points in time.** Therefore, it requires keeping track of how the decision situation is evolving over time. **The information about the current situation which is needed to make a correct decision is called the "state"**.

The variables chosen at any given point in time are often called the ***control variables*.**

The **dynamic programming approach describes the optimal plan** by finding a rule that tells what the controls should be, given any possible value of the state. You can see the example of dynamic programming in the below figure.

![dynamic programming](images/dynamic programming.png)

Each cell of the table was filled with a score from a scoring system and pre-states. There were two cases when you scoring from pre-states to a present state. first, optimal case to get the highest score among pre-states. second, a case to take lower score than other case. Of course, you must choose first one.

Consequently, present decision should be done to make optimal as considering both present payoff and the following states.

#### 2-2. Bellman equation



#### 2-3. policy iteration

![state-value_tree](images/state-value_tree.png)

To solve dynamic programming, reward and state transition probability information must be known. At the first, a agent can not know policy function. Therefore It start with random policy. We make the agent learn environment and get the true value function. this process is called **policy evaluation**. Next, we need to update our policy to be better. Then the policy gradually become optimal. This is **policy improvement**.
$$
\huge v_{k+1}(s) = \sum_{a \in A}\pi(a|s)(R^a_s + \gamma\sum_{s' \in S}P^a_{ss'}v_k(s'))
$$
implementing  iterations of policy evaluation and policy improvement to get optimal policy is called **policy iteration**.

![policy_iteration](images/policy_iteration.png)



#### 2-4. value iteration

The different thing with Value iteration to Policy iteration is using Bellman optimality equations, instead Bellman expectation equation. 

![value_iteration](images/value_iteration.png)

이 equation을 사용하여 다음과 같이 update를 하는데 policy iteration처럼 중간 중간에 policy를 update하지않고 마지막에 구해진 optimal q-function의 값을 보고 greedy하게 policy를 정해버립니다. Policy Iteration때와 마찬가지로 Bellman optimality equation과 다른 점은 k라는 iteration숫자가 달리면서 한 스텝 한 스텝 업데이트를 한다는 것입니다.

![value_iteration_2](images/value_iteration_2.png)









Reference : https://en.wikipedia.org/wiki/Bellman_equation

#### 2-3. "The FrozenLake" game introduction 

![frozen-lake](images/frozen-lake.png)

The FrozenLake game consists of a 4x4 grid of block, each one being either the starting block, a safe frozen block, the goal block and a dangerous hole block. The objective is to train an agent which can move from the starting block to the goal block, not to fall into any hole. The reward at every step is 0, except for entering the goal, which provides a reward of 1.The agent can move either up, down, left, or right at any time. However, sometimes there is a wind to blow the agent onto a space It did not choose. 

Therefore perfect performance every time is impossible, but we want to design an agent to take **long-term expected rewards**. This is exactly what Q-Learning is designed to provide.



This equation states that the long term expected reward given a action is equal to **combination of the immediate reward from the current action and the expected reward from the best future action taken at following state.**
$$
Q(s, a) = r + ymax( Q(s', a') )
$$
This says that the Q-value for a given state (s) and action (a) should represent the current reward (r) plus the maximum discounted (γ) future reward expected according to our own table for the next state (s’) we would end up in.

![Q_learning_with table](images/Q_learning_with table.gif)

​										Q-Learning with a table

![Q_learning_with networks](images/Q_learning_with networks.jpg)

​										Q-Learning with a network





## 3. Policy gradient

![agent](images/agent.jpg)

![policy_value_based](images/policy_value_based.png)



![policy_based](images/policy_based.png)





































