# 🤖 Unlocking the Non-Native Language Context Limitation: Native Language Prompting Facilitates Knowledge Elicitation💡

📢This repository is maintained for the complete set of code for ***Nat**ive **Lan**guage Prompting (**NatLan**)*.

🥳In addition to NatLan, we also provide rapid implementations of related methods such as Self-Translation and Google-MT for comparative analysis.

## Native Language Prompting (NatLan)

Run `main.py` to execute NatLan and related methods, with the following parameter configuration:

| Parameter             | Explanation                                         |
| --------------------- | --------------------------------------------------- |
| `model_pth`           | Speaker LLMs path                                   |
| `translate_model_pth` | Translator LLMs path (for NatLan)                   |
| `model_name`          | result file identifier                              |
| `data_pth`            | dataset path                                        |
| `result_pth`          | result saving path                                  |
| `prompts_pth`         | translation prompts (for NatLan & Self-Translation) |
| `nshot`               | n-shot prompting                                    |
| `use_natlan`          | whether to use NatLan                               |
| `use_google`          | whether to use Google-MT                            |
| `use_self`            | whether to use Self-Translation                     |
| `target_lang`         | the original langauge of the questions              |
| `native_lang`         | the native (dominant) language of Speaker LLMs      |
| `google_api_key`      | google translate API (for Google-MT)                |

P.S. Only one of the variables `use_natlan`, `use_google`, or `use_self` can be set to `True` at most. If all three are set to False, the default behavior is to have Speaker LLMs directly answer questions in the target language (`Original` in our paper).

## Detailed Prompts

The proposed NatLan method involves two types of prompts: **translate prompts** and **Q&A prompts**.

1. **Translate prompts**

```markdown
<System Prompts>
You are a professional Chinese-English translator. Translation rules: Proper nouns in English or Chinese need to be retained without translation, retain the original meaning to the greatest extent, and follow the original format in the translation process.

<Original Question Prompts>
Now help me translate the following sentence into English, only return the translated sentence, the original sentence is:
Question:
{original example['question']}
Choices:
A. {original example['choice A']}
B. {original example['choice B']}
C. {original example['choice C']}
D. {original example['choice D']}
Answer:

<Translated Question Prompts>
Question:
{translated example['question']}
Choices:
A. {translated example['choice A']}
B. {translated example['choice B']}
C. {translated example['choice C']}
D. {translated example['choice D']}
Answer:
```

2) **Q&A prompts**

```markdown
<System Prompts>
You are a professional {subject name} expert, and you are currently answering a multiple-choice question about {subject name}, you need to provide only one option as the answer based on the question, and you only need to return one single capital character as the answer.

<Question Prompts>
Question:
{translated example['question']}
Choices:
A. {translated example['choice A']}
B. {translated example['choice B']}
C. {translated example['choice C']}
D. {translated example['choice D']}
Answer:

<Answer Prompts>
{example['answer']}
```

P.S. The translated examples are derived from results pre-translated using `gpt-4o-2024-05-13` on questions from the C-Eval dev set, stored in the `.json` file that is stored in the  `prompts_pth`.

## Other Hyperparameters

To reduce randomness introduced by sampling, the LLMs in this experiment use **Greedy Decoding** for all generation processes.

## Evaluation

This experiment is evaluated on [the C-Eval benchmark](https://github.com/hkust-nlp/ceval). You can directly submit the `.json` file generated after running `main.py` to [the official C-Eval platform](https://cevalbenchmark.com/index.html) for performance assessment on the test set.

## Performance

We report the performance of NatLan on the test set of the C-Eval benchmark as follows:

| Model             | Language | Avg.     | Avg. (Hard) |
| ----------------- | -------- | -------- | ----------- |
| Phi-3-mini (3.8B) | zh       | 41.2     | 36.3        |
| +Self-Translation | en       | 43.8     | 37.7        |
| +Google-MT        | en       | 50.9     | 40.4        |
| **+NatLan**       | **en**   | **51.3** | **41.3**    |
|                   |          |          |             |
| Phi-3-small (7B)  | zh       | 49.0     | 41.6        |
| +Self-Translation | en       | 52.0     | 42.1        |
| +Google-MT        | en       | 55.7     | 42.7        |
| **+NatLan**       | **en**   | **55.9** | **44.7**    |
|                   |          |          |             |
| Gemma-1.1 (7B)    | zh       | 44.4     | 36.3        |
| +Self-Translation | en       | 41.9     | 33.9        |
| +Google-MT        | en       | 46.7     | 38.2        |
| **+NatLan**       | **en**   | **47.7** | **38.6**    |
|                   |          |          |             |
| Mistral-0.3 (7B)  | zh       | 42.8     | 32.6        |
| +Self-Translation | en       | 34.8     | 30.9        |
| +Google-MT        | en       | 48.0     | 33.3        |
| **+NatLan**       | **en**   | **47.7** | **38.6**    |
|                   |          |          |             |
| Llama-2 (7B)      | zh       | 21.3     | 14.7        |
| +Self-Translation | en       | 9.6      | 10.3        |
| +Google-MT        | en       | 25.4     | 15.1        |
| **+NatLan**       | **en**   | **27.6** | **18.6**    |

P.S. In this experiment, all models involved are the Instruct/Chat versions. For more detailed performance results, please refer to our paper.

## Citation

🚀Under Review and Coming Soon

