# Deepseek R1: Explaining the Science Behind the Miracle

This will be an in-depth, yet intuitive blog on Deepseek R1, covering all the important talking points about it from an AI researcher/engineer’s perspective. This blog is ideal for beginners, students as well as practioners and hobbyists. So strap in!

## Why Deepseek R1 is SO Important?

- It is the first time an open source model has beaten/matched the performance of State-of-the-Art (SOTA) LLMs.
- It was *allegedly* trained for just $5.5 million, which is peanuts compared to the training costs of other SOTA LLMs. This also released soon after OpenAI and the US government announced the $500 billion Stargate project, so discussions about the cost of training AI models were already trending. Also how they were able to train the model despite serious export restrictions on Nvidia GPUs to China has been a hot topic of discussion
- The company behind the model, Deepseek, is Chinese, and their ability to produce such an advanced model came as a shock to a lot of experts as well as the general public, who believed that USA has a decent advantage in AI as compared to China.
- The model introduces and uses several very interesting techniques to train the model so efficiently, including a new reinforcement learning algorithm called GRPO (more on that later in the blog), which piqued the interest of AI researchers.

Now that we’ve set the background of R1, let’s move on to the technical aspects of R1, starting with an introduction to LLMs and Reinforcement Learning.

# LLMs and Reinforcement Learning (RL):

## What is RL?

- Reinforcement Learning (RL) is a way for computers to learn by trying things out and getting rewards or penalties. The computer (agent) keeps testing actions and observing their outcomes (states), learning which ones lead to better rewards. Over time, it gets better at making decisions, just like learning from experience.
- **Agent:** The language model itself
- **State:** the prompt (input tokens)
- **Action:** which token is selected as the next token
- **Reward Model:** Rewarded for generating good response, no reward for generating bad response. While negative reward can be given for bad responses, it is sometimes avoided as negative rewards can destabilize the training model and complicates optimization.
    - **Policy:** Policy is basically what tells the model what to do. For LLMs, the policy is the language model itself as it models the probability of the action space given the current state of the agent. Basically, as the LLM itself produces a probability distribution of the next token, LLM is the policy itself.m

![[https://github.com/hkproj/rlhf-ppo/blob/main/Slides.pdf](https://github.com/hkproj/rlhf-ppo/blob/main/Slides.pdf)](Deepseek R1 Blog/image.png)

[https://github.com/hkproj/rlhf-ppo/blob/main/Slides.pdf](https://github.com/hkproj/rlhf-ppo/blob/main/Slides.pdf)

### Reinforcement Learning from Human Feedback (RLHF):

- In the pre-training stage, we dump a lot of info at the LLM, e.g. the entire web. But when we ask a LLM something, we want the answer to be structured and formatted in a certain way instead of it just throwing info at us from it’s training. So, in post-training, alignment training is required.
- E.g. we need the model to follow our instructions, always greet the user, avoid curse words, etc. We use reinforcement learning from human feedback here.
- So, a dataset is created of question, and we ask the LLM to generate multiple answers for the input query. Then, professional annotators mark (reward) the answers which are better, incentivizing the LLM to generate answers which are preferred by the annotators (humans).
- But up until now, pure RL in post-training has been insufficient. Most of the post-training heavy lifting has been done by supervised fine-tuning while RLHF has been used to provide the ‘finishing touches’ to the models.

## Supervised Fine Tuning (SFT):

- Before RLHF, the model undergoes Supervised Fine Tuning (SFT) to learn how to write ideal responses.
- Basically a large dataset is created of queries and the ideal/expected answer.
- E.g. one entry in the dataset can be:
    - **Query:** What is the capital of France?
    - **Ideal Answer:** ‘The capital of France is Paris, a city also known for it’s fashion and food. Is there anything else that you would like to know about Paris?’
- The model is trained on this dataset to tune the model to answer the questions as in the dataset. This dataset is mostly used to teach the model how to reply considering it already has learnt the info in pre-training.
- Though new information is taught to the model using SFT in post-training if the model is finetuned to be better in a particular domain, e.g. an LLM being finetuned to be better at law.
- SFT along with RLHF is the key reason why different LLMs ‘sound’ different. SFT and RLHF is also used to train the model to reject answering certain queries (e.g. related to politics, making a bomb, etc.).
- While SFT teaches the model how to reply, it may not yet be completely deployable. The model may still exhibit some quirks, which are polished out by humans in RLHF.
- So, SFT gives a strong foundation of how to reply to LLMs, and RLHF fine-tunes it based on human feedback for better user experience.

# Deepseek R1

- Deepseek R1 is an open-source thinking model released by the Chinese firm Deepseek, which they claim was trained for just $5.5 million (OpenAI’s comparable model o1 whose training cost allegedly range between hundreds of millions to a few billion dollars).
- Deepseek R1 ditches (or more accurately, significantly minimizes) the SFT phase and only performs RL on the base LLM model Deepseek V3.
- The RL optimization technique R1 uses is the **Group Relative Policy Optimization (GRPO)**, which is the key innovation here.

## Group Relative Policy Optimization (GRPO):

- Now let’s get a bit technical and jump directly into the gigantic GRPO formula. While it may scare you at first, trust me, by the end of the blog, you’ll have a complete intuitive understanding of the equation and why is it the way it is!
- The equation for the GRPO algorithm is as follows:

$$
\mathbb{E}_{q \sim P(q)} \mathbb{E}_{\{(o_i)\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^G \min \left( \frac{\pi_{\theta}(o_i | q)}{\pi_{\theta_{\text{old}}}(o_i | q)} A_i, \, \text{clip} \left( \frac{\pi_{\theta}(o_i | q)}{\pi_{\theta_{\text{old}}}(o_i | q)}, 1 - \varepsilon, 1 + \varepsilon \right) A_i \right) \right] - \beta \, \mathbb{D}_{KL} (\pi_\theta || \pi_{ref})
$$

- While it looks scary and complicated we’ll break it down to make it simpler.
- Let’s start with our goal. Our goal is to optimize the policy. We want to train the policy $\pi_{\theta}$ to maximize our objective (the equation above).
- **REMINDER:** In context of LLMs, policy is what decides how the next token in an LLM will be chosen, here it depends on the attention mechanism (but we don’t need to dive too much into it here).
- But what exactly is our objective here? What is the big equation implying?
- Let’s say we have a list of questions that belong to a database of questions
    - q ~ P(q)
- and we sample some outputs from our current policy using these questions. NOTE that we’re **sampling multiple outputs for multiple questions**.
    - $\{(o_i)\}_{i=1}^G \sim {\pi_{\theta_{old}}} (O | q)$
- Based on the reward the outputs get, we train the LLM to give more weight to (make it more likely to take) those actions that result in good reward and to give less weight to actions that result in bad rewards.
- Firstly, let’s understand log probabilities:
- For the query: **Where is Shanghai?**
- Consider the answers:
    - Shanghai is in China.
    - The sky is blue.
    - I am very smart
- There’s log probabilities associated with each word (e.g. Shanghai, The, great, etc. for the query ‘Where is Shanghai?’
- For ‘Where is Shanghai’, say we have log probabilities for the words:
    - Shanghai, The and I.
- Then, for ‘Where is Shanghai? Shanghai’, there’s log probabilities for the words:
    - is, and, was, etc.
- and so on for all the words…
- $\pi_{\theta_{old}} (o_i | q)$ is the old (previous) policy and $\pi_{\theta} (o_i | q)$ is the new (current) policy.
- $\pi_{\theta} (o_i | q)$ is the product of all the log probabilities generated at each step $O_i$. Each log probability is weighted by Advantage $A_i$.
- Advantage $A_i$ basically tells how much better is choosing a particular token given a particular input over all possible tokens.
    - For ‘Where is Shanghai?’, Shanghai would have a better Advantage value than, say, sky or pizza’.
- Let’s sum up what we’ve learnt about the formula till now:
    - For a group of questions, and multiple answers for each question, and for each answer we have log probabilities associated with this answer ( which is basically equals to the product of all the probabilities of choosing that particular token given that particular input). We weigh each of the log probability with the Advantage term ( which tells us how good is choosing this particular token as compared token over all the other that are available for this particular input).
    - It may seem a bit overwhelming initially so you may want to read it twice or thrice to ingest the info completely before moving forward.
- The ratio $\frac{\pi_\theta(O_i|q)}{\pi_{\theta_{old}}(O_i|q)}$ represents the *relative change* in the probability of taking that output $o_i$between the old and current policies.
    - If the ratio is greater than 1, it means the current policy is more likely to generate that output.
    - If the ratio is less than 1, it means the old policy is more likely to generate that output.
    - A ratio close to 1 suggests the policies are similar for that output.
- Now moving on to the ‘clip’ part of the equation:  $\text{clip} \left(\frac{\pi_\theta(O_i|q)}{\pi_{\theta_{old}}(O_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i$.
- The aim of ‘clip’ is to make training more stable and ensure that the policy doesn’t change too much in each iteration.
- As the name suggests, it basically ‘clips’ or limits the value of how much better or worse the next token is in this policy as compared to the previous policy, in turn, restricting how much the policy changes in each iteration.
- Let ratio = $\frac{\pi_\theta(O_i|q)}{\pi_{\theta_{old}}(O_i|q)}$, then, the ‘clip’ part of the equation becomes:
    - $\text{clip} (ratio, 1-\epsilon, 1+\epsilon)\;A_i$
- Clip is a function which outputs:
    - $1 + \epsilon$ if ratio > $1 + \epsilon$
    - $1-\epsilon$ if ratio < $1-\epsilon$
    - ratio if ratio is between $1-\epsilon$ and $1+\epsilon$
- Here, $\epsilon$ is a small hyperparameter, like 0.1 or 0.2.
- What this does is that it limits the ratio to be between $1-\epsilon$ and $1+\epsilon$. So even if a particular output token is significantly better than the rest, it will be clipped at a max value of $1+\epsilon$ and even if an output token is significantly worse, it is clipped at a minimum value of $1-\epsilon$.
- This value is then multiplied by the Advantage $A_i$ to form the new clipped value.
- The reason it does this is to enable more stable and conservative training, avoiding aggressive policy updates which may wash out the model’s learnings and lead to slower generalization and more bias.
- Now we can understand the key chunk of the equation: $\min \left( \frac{\pi_{\theta}(o_i | q)}{\pi_{\theta_{\text{old}}}(o_i | q)} A_i, \, \text{clip} \left( \frac{\pi_{\theta}(o_i | q)}{\pi_{\theta_{\text{old}}}(o_i | q)}, 1 - \varepsilon, 1 + \varepsilon \right) A_i \right)$
- To sum up, we take the minimum value between $ratio * Advantage$ and the clipped value multiplied by the Advantage for more stable training and less aggressive updates (to decrease bias and help generalize quicker).
- Now turning our focus onto another key part of the equation the KL-divergence: $\beta \; \mathbb{D}_{KL} (\pi_\theta || \pi_{ref})$. It tells us how different 2 distributions are, or how far they are from each other.
- The aim of the KL-Divergence here is to reduce the difference between the current policy and the previous policy, to ensure that the training is stable and the changes are not too drastic.
- Just like some houses have 2 doors with 2 locks for extra protection, this equation has 2 locks (clip and KL-divergence) to ensure that the changes are not too drastic.
- $\beta$ is a positive hyperparameter which controls the strength of the KL-penalty. Higher $\beta$ means higher higher KL-penalty.
- As we’re subtracting the KL-divergence, we’re incentivizing the model to minimize the KL-divergence between the current policy and a reference policy (not much detail is provided in the paper about the reference policy and how often it is updated).
- We’re essentially performing policy regularization here by incentivizing the model to not perform policy upgrades that are too aggressive (as it may wash out a lot of progress made). This results in more stable training updates.
- Increasing $\beta$ will incentivize the model to perform **less aggressive** policy updates while decreasing $\beta$ will allow the model to perform more aggressive policy updates.
- Now we have covered the main equation.
- To sum it up, we take the minimum value between the ratio * advantage and the clipped value * advantage, and subtract it with the KL-divergence weighted with the hyperparameter $\beta$.

### GRPO vs PPO:

- First let’s understand the key problems with Proximal Policy Optimization (PPO):
    - Computationally and memory-intensive as it has a complex policy neural network as well as a complex value neural network.
    - Training is sometimes instable resulting in sub-optimal results occasionally.
    - Slower, requiring a lot of iterations.
- Now let’s understand how GRPO improves upon them:
    - GRPO has only 1 neural network (policy neural network), and doesn’t use the value neural network, reducing the memory load as well as compute required.
    - Training is more stable in GRPO as compared to PPO due to additional measures taken.
    - Significantly faster than PPO, requires fewer iterations (as per the anecdotal evidence provided my multiple labs and research groups).
- The key difference between them is that PPO required a critic model, so we needed to train another neural network which is computationally intensive as well as time consuming.
- In GRPO, the advantage terms $A_i$ are calculated using the formula: 

$$\frac{r_i - \text{mean}(r_1, r_2, ..., r_G)}{\text{standard\_deviation}(r_1, r_2, ..., r_G)}$$ 

where $r_i$ is the reward assigned to the output i and G is the total number of outputs generated. This is a very simple method to calculate Advantage as compared to what PPO uses (a critic neural network), which takes up more time, compute and memory.
- GRPO ranks the candidate solutions that are generated, indulging in relative ranking of the candidate group.
- So GRPO basically sacks the critic neural network in favour of the simple Advantage calculation
- GRPO also introduces KL-divergence directly into the loss function which stabilizes training.

- Now that we’ve understood the equation, let’s checkout an astonishing result, and one of the other major talking point of the paper, the ‘aha’ moment.

## The ‘aha’ moment in R1-Zero:

![image.png](Deepseek R1 Blog/image%202.png)

- In an intermediate version of R1-Zero, the model was given a math problem, and in between it’s thinking process, the model has an ‘aha’ moment, optimizes it’s approach and solves the problem correctly.
- The thought process of the model here is very human-like, and the model has it’s ‘aha’ moment without being explicitly trained to think in a certain way or solve a math problem in a certain way.
- This is an ‘aha’ moment for the model as well as the researchers as it proved how effective RL is that just by providing the right incentives it autonomously developed advanced problem-solving techniques.
- This ‘aha’ moment is reminiscent of Move 37 by AlphaGo and an instant classic in the world of AI research!

# R1 and R1-Zero:

- An important detail I’ve left until now is that Deepseek has trained 2 reasoning models: R1 and R1-Zero, and they’ve elaborated on both the models in the paper.
- R1-Zero is trained solely using RL without SFT and using only GRPO technique for policy optimization.
- R1-Zero was trained first and it resulted in a lot of the breakthroughs which are the main highlights of the paper, but to make the model better and truly an o1 competitor, they took the learnings from R1-Zero and trained a better model R1.
- The key difference between R1 and R1-Zero is that in R1, there is a small SFT phase before RL where high-quality datasets are used for initial SFT to make the thinking process of the model more coherent and readable (e.g. R1-Zero had issues like losing coherence or changing languages mid-thought, which R1 improves upon).
- There’s a second RL stage in R1 which aligns the model with human preferences for helpfulness and safety.
- Basically R1 is the more ‘polished’, production-grade version of R1-Zero while R1-Zero is the core model in which novel techniques like GRPO were initially used.

# Reward Modelling:

- R1-Zero follows a rule-based reward system that mainly consists of 2 types of rewards:
    - **Accuracy reward:** Where the model is rewarded for the correct answer.
    - **Format reward:** Rewards and enforces the model to put it’s thinking between <think></think> tags.
- They design a straightforward template requiring R1-Zero to produce a reasoning process first followed by the final answer. They consciously avoid adding any content-specific instructions (e.g. promoting a particular problem solving strategy or reflective reasoning) to assess the model’s natural progress during the RL process. This is in contrast to other thinking model where the models are encouraged to follow a thinking strategy.
- The template for Deepseek R1-Zero is as follows:
    - A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
    The assistant first thinks about the reasoning process in the mind and then provides the user
    with the answer. The reasoning process and answer are enclosed within <think> </think> and
    <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
    <answer> answer here </answer>. User: prompt. Assistant:

![image.png](Deepseek R1 Blog/image%203.png)

- A really intriguing thing that the model learnt during the RL phase in R1-Zero is generating longer responses to solve reasoning tasks, allocating more time to thinking. It learnt to prioritize long term rewards, generate longer sequences of thinking for logical tasks without being explicitly prompted to do it!

## Why is the Advantage Term the way it is? (skip if you don’t want to get too math-ey)

- I’ve already outlined that the key upgrade from PPO to GRPO is the ditching PPO’s value network for a simpler Advantage calculation, let’s dive deeper into the Advantage calculation.
- The advantage term is calculated as follows: $A_i = \frac{r_1-mean(r_1, r_2, ..., r_G)}{std(r_1, r_2, ..., r_G)}$, but why is it calculated the way it is? Why is this simple formula outperforming a complex neural network (as present in PPO)?
- Let’s first understand what exactly is the formula doing:
    - for every reward $r_i$, we subtract it from the mean reward of that group.
    - Then, we divide that value by the standard deviation in the rewards of that group.
- Why do we do this?
    - We subtract the reward by the mean reward to center the advantage function around 0. So if a reward is better than average, the value will be positive. If the reward is worse than the average reward, the value will be negative.
    - So, the advantage function is not determined by the absolute reward but by the relative reward. So even if all the samples in the group achieve fairly high rewards, the model will be optimized according to the samples that perform better than the other samples, which already were performing fairly well.
    - Then, we divide it by standard deviation of the rewards obtained in that group. The standard deviation tells us how much the data usually diverge from the mean. If the values are all similar, they will be closer to the mean, so the standard deviation will be smaller. If the values are distributed, they will usually be far from the mean, resulting in a higher standard deviation.
    - By dividing the advantage function by the standard deviation, we’re scaling the advantage function based on how much the rewards usually vary in the group.
        - So if the standard deviation (variation) is high, the advantage function will be scaled down quite a bit, as the rewards are noisy (highly varied), so it is ideal to make smaller updates for stable and more generalizable training. Making big adjustments based on 1 or a few samples when the rewards are noisy may introduce bias and result in the model getting stuck in a local minima/suboptimal policy.
        - On the other hand, when the standard deviation is low, the rewards are stable, the advantage function value will be higher as it’s getting divided by a smaller number. This is desirable as we’ve achieved fairly stable rewards (as standard deviation is low) here and even small improvements are desirable and should be reflected in the model.
        - With the standard deviation, we’re normalizing the advantage function so that the importance of a significantly better reward in a noisy reward group is not overestimated, and at the same time, the importance of a slightly better reward in a stable reward group isn’t underestimated.
    - With this we robust, effective and efficient RL-training for the model.

# R1-Zero → R1:

- R1-Zero produced excellent results solely through RL but it faced some problems like poor readability and language mixing.
- Highly encouraged by the results of R1-Zero, the research team set out to use the learnings from R1-Zero and optimize to create a higher quality, production-grade model. They designed a new pipeline to train R1, the model that’s released to public now.
- The R1 pipeline can be divided into 4 phases:

### Phase 1 - SFT for Cold Start:

- To prevent the early unstable cold start phase of R1 and speed up convergence they performed Supervised Fine Tuning (SFT) on a small dataset.
- They collected some high-quality Chain-of-Thought (COT) data and finetuned the base model on it. This also helped in improving the readability of the model’s thinking process.
- The SFT finetuning dataset was miniscule as compared to the SFT datasets using in current SOTA models like o1, gpt-4o, etc. so this didn’t raise the training cost by a lot.

### Phase 2 - Reasoning RL:

- During the RL phase in R1, they also introduced a reward for language consistency to mitigate language mixing.
- This phase is the same as in R1-Zero.

### Phase 3 - Rejection Sampling and SFT:

- After the model converges, a lot of samples are generated, including multiple samples for the same prompts to create a dataset.
- This dataset undergoes a rigorous filtering process which includes 2 filters:
    1. **Rule based Checks:** Specific rules to ensure correct answer and formatted in a certain preferred way.
    2. **Semantic Filtering:** Filtering out less coherent/relevant samples.
- After this only the most coherent, relevant samples with the correct answers in the desired format are retained.
- R1 is then finetuned on this along with some other data to further improve the quality of the output and improve the performance at general tasks underrepresented in its training until now, like writing, role-playing, etc.

### Phase 4 - Diverse Reinforcement Learning:

- Rule-based and human/AI feedback-based methods are introduced in this final phase to align the model as per human preferences.
- This is a standard practice in all SOTA/production LLMs to ensure a good, consistent user experience.

# Distilling R1 into other LLMs:

- As this is not a key innovation or talking point of the paper, I will only briefly go through this part.
- Knowledge distillation is mainly used to train smaller models on the output of bigger, smarter models. This is one of the key reasons why small language models have become more effective in the past year. Here too, their motivation of distillation is to ‘distill’ or pass on the knowledge of a large Deepseek-V3 model (670B parameters) into smaller (1B, 8B, 32B, etc.) LLMs.
- Knowledge distillation can be understood as a teacher LLM teaching it’s capability to a student LLM. This process is far cheaper than training the model from scratch.
- The Deepseek team distilled the ‘thinking’ knowledge from Deepseek V3 Base to a range of open source Meta Llama and Alibaba Qwen models.
- They distilled the knowledge into LLMs of different sizes including smaller models like Qwen 1.5B and Llama 8B, making them ideal for cheap or on-device local usage.
- Even the smaller, distilled models outperform many SOTA models according to the results in the paper (and recently verified by independent sources).

To summarize, Deepseek R1 shook the world of AI using new methodologies, neat tricks to deliver an outstanding model at a fraction of the cost. I’ve tried to intuitively explain all the key concepts to help you understand how they achieved this feat including GRPO, 4-phase R1 pipeline, etc.

With this, we come to an end to a slightly long, but hopefully well-explained blog post on Deepseek R1. I hope this was a valuable read and would love to hear your feedback/questions.

For any questions, or just to drop an Hi, email me at [vanshkharidia7@gmail.com](mailto:vanshkharidia7@gmail.com) or DM me on [X](https://x.com/VanshKharidia).
