# eyeballvul experiments

Code and data accompanying the paper [eyeballvul: a future-proof benchmark for vulnerability detection in the wild](https://arxiv.org/abs/2407.08708).

See also: [eyeballvul](https://github.com/timothee-chauvin/eyeballvul).

⚠️️️️️️️️⚠️⚠️

Due to a [bug in LiteLLM](https://github.com/BerriAI/litellm/commit/2452753e084e8134c0c484b32c63fb5f2950c5ba), the Gemini 1.5 Pro cost was underestimated in the original version of the paper. A revised version has been submitted to arxiv and should be available on Monday, July 15th.

Until then, [this](data/plots/costs.pdf) is the correct version of Figure 7. Both Claude 3.5 Sonnet and Gemini 1.5 Pro stand out from the other models in terms of low inference cost and low ratio of false positives to true positives.

⚠️⚠️⚠️
