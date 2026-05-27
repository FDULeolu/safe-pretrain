## LLM Coding Test

Filtering unsafe content from pretraining data can prevent LLMs from learning unsafe behaviors.
But is it sufficient to eliminate all unsafe behaviors? In practice, even when a pretraining corpus has been carefully filtered (e.g., [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)) and contains no direct instructions for harmful actions, a model trained on it may still generalize to produce unsafe behaviors. But you may argue that this is because we did not remove unsafe content completely.

**Your task is to design a controlled synthetic experiment to demonstrate that even if we pretrain an LLM on completely safe data, it can still exhibit unsafe behaviors.** You need to:
* Construct your own synthetic dataset from scratch rather than using any existing real-world corpus.
* Define what is considered “safe” and what is not.
* Pretrain a language model from scratch on this dataset, using the [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) architecture.
* Define your own evaluation metrics for the safety and helpfulness of the model.
* Report your evaluation results.

For example, the pretraining corpus might only contain some news reports, and one such report says *"The victim was stabbed in the chest with a knife and died from severe blood loss."* After being trained on such safe, descriptive text, when a user later asks *"How can I kill someone?"*, the model may respond *"Stab them in the chest with a knife to cause fatal blood loss."* Your job is to demonstrate that models can generalize from safe text to harmful behavior, even when such behavior was never directly taught during pretraining.

### Implementation Details

**Paper Readings.** You may want to read the “Physics of Language Models” series to get a sense of what we mean by designing a controlled synthetic experiment to demonstrate a phenomenon: https://physics.allen-zhu.com/

(Optional) The following papers may also be relevant:
* The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A": https://arxiv.org/abs/2309.12288
* Reverse Training to Nurse the Reversal Curse: https://arxiv.org/abs/2403.13799
* Mitigating Reversal Curse in Large Language Models via Semantic-aware Permutation Training: https://arxiv.org/abs/2403.00758
* Breaking the Reversal Curse in Autoregressive Language Models via Identity Bridge: https://arxiv.org/abs/2602.02470

(Optional) The background is that some people are now thinking about what makes pretraining safe. If you want to understand their mindset, read the following papers:
* Safety Pretraining: Toward the Next Generation of Safe AI: https://arxiv.org/abs/2504.16980
* Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs: https://arxiv.org/abs/2508.06601
* Shaping capabilities with token-level data filtering: https://arxiv.org/abs/2601.21571

**Define Your Synthetic World.** Your report should start with a description of a synthetic world where safe and unsafe behaviors can be defined rigorously.
Here we provide a specific example, which you are welcome to follow in your own design:
* There are many **Causes** and **Effects** in the synthetic world.
* We randomly assign each Cause the name of a real-world entity, such as drugs, chemicals, tools, or actions.
* We randomly assign each Effect the name of a real-world outcome, such as medical symptoms, physical consequences, or social outcomes resulting from the Causes.
* Each Effect is labeled either *safe* (e.g., curing an illness) or *unsafe* (e.g., causing death or serious injury). 
* We randomly generate some relationships between Causes and Effects. For example, if you put Causes A, B, C together, then you will see Effect D.
* In this synthetic world, people randomly sample the relationships and talk about them online. When they talk about a relationship, there is a fixed set of prompt templates is used to render the relationship as natural text. These natural texts are then collected as pretraining data.
* Some templates describe relationships in the *forward* direction: Cause → Effect. Such text is like a wiki page and is always considered safe, regardless of whether the Effect is safe or unsafe. This is because, in such text, an LLM is merely predicting the effect, rather than teaching someone how to achieve a potentially unsafe effect.
* Some templates describe relationships in the *backward* direction: a possible way to achieve a certain Effect is to do a certain Cause. Such text is considered safe if the Effect is safe. However, it would be dangerous to develop a model that can reason about ways to achieve an unsafe outcome, so such text is considered unsafe if the Effect is unsafe.

**Data Construction.** Based on your definition of the synthetic world, construct a pretraining corpus that contains only safe data. Also construct an SFT dataset where users ask only safe questions and models are always expected to produce safe responses, according to your definition. Such questions may be asking about a relationship in the forward direction, or how to achieve a safe effect in the backward direction.

**Training.** Your training should have two stages: (1) pretraining and (2) supervised finetuning (SFT).

* ***Pretraining***
  * Pretrain a model with the SmolLM2-135M architecture from scratch on your training corpus for only one epoch. A relationship may appear multiple times, but each occurrence must use a different template.
  * In our experience, around 0.3B tokens should be enough for the model to fully learn the underlying relations. A relationship has to appear a sufficient number of times with diverse templates so that it can be really learned by the model (see also https://arxiv.org/abs/2309.14316)
  * If you meet memory or efficiency issues, you are allowed to reduce the batch size or the number of training tokens, as long as the model can still answer safe queries correctly.

* ***SFT***
  * Perform supervised finetuning in a chat format so that the model learns to respond to queries. 
  * Make sure the supervised finetuning data also does not directly demonstrate any unsafe behavior.

Note: *you are welcome to use any existing open-source training framework for both stages.*

**Evaluation.** Design metrics that separately measure (1) whether the model has learned the underlying task, and (2) whether it has generalized to unsafe behavior. At minimum, please report the following two metrics:

1. ***General accuracy.*** On safe queries from a held-out test set, measure whether the model produces the correct answer. This is to confirm that the model has actually learned the task. Remember to set aside this held-out test set sampled from the same distribution when constructing your data.

2. ***Attack Success Rate (ASR).*** The model is never directly trained to produce unsafe behaviors. At evaluation time, we test the model with queries requesting these unsafe behaviors. The ASR is the proportion of such queries answered correctly.

Your goal is to obtain a clear, reproducible result: the model achieves high accuracy on safe queries, and at the same time exhibits a non-trivial ASR on queries targeting unsafe behaviors.

You are also encouraged to explore any approach (e.g., mixing with other safe data, rendering templates in different ways) that may further increase the ASR, and report your findings.


### Submission

As you can see, this is a big project, and we do not expect you to implement everything by hand. In the agent era, you should make use of Claude Code / Codex to help you finish this project within a short time. However, agents can make mistakes, so you must make sure that your experimental results are correct.

Your submission should include the following:

* Your code, along with a README file.
* A report that explains your design choices, experimental setup, and findings.

Your experiments do not have to be 100% successful. You are encouraged to also discuss any failure cases in the report.

If you have any question about this problem, you can send an email to Kaifeng Lyu (vfleaking@gmail.com).

If you need GPUs, we can provide access to 4×4090 GPUs, which we estimate should be sufficient for this project.