# 🤖 NatLan: Native Language Prompting Facilitates Knowledge Elicitation Through Language Trigger Provision and Domain Trigger Retention💡

📢This repository is maintained for the complete set of code for ***Nat**ive **Lan**guage Prompting (**NatLan**)*.

🥳In addition to NatLan, we also provide rapid implementations of related methods such as Self-Translation and Google-MT for comparative analysis.

## Detailed Prompts

The proposed NatLan method involves two types of prompts: **translate prompts** and **QA prompts**.

**1. Trans. prompts**

```markdown
<System Prompts>
You are a professional {non-native language name}-English translator. Translation rules: Proper nouns in English or {} need to be translated according to the {discipline name} domain-specific terms, retain the original meaning to the greatest extent, and follow the original format in the translation process.

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

**2. QA prompts**

```markdown
<System Prompts>
You are a professional {discipline name} expert, and you are currently answering a multiple-choice question about {discipline name}, you need to provide only one option as the answer based on the question, and you only need to return one single capital character as the answer.

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

## Other Hyperparameters

To reduce randomness introduced by sampling, the LLMs in this experiment use **Greedy Decoding** for all generation processes.

## Evaluation: MMMLU Benchmark

The relevant code is contained within the `./MMMLU` directory.

During our evaluation phase, to facilitate data reuse and enhance computational efficiency, we implemented a segmented computation and storage process for the translate-then-answer methods, divided into the following three steps:

**1. Get Translation:**

```bash
python self_trans.py
python google_trans.py
python gpt_trans.py
```

**2. Inference:**

```bash
bash inference.sh
```

**3. Evaluation and Further Analysis:**

```
python evaluate.py
```

Further analysis was constructed based on experiments conducted within the paper...

## Evaluation: C-Eval Benchmark

The relevant code is contained within the `./C-Eval` directory.

Run `main.py` to execute NatLan and related methods, with the following parameter configuration:

| Parameter        | Explanation                                         |
| ---------------- | --------------------------------------------------- |
| `model_pth`      | Speaker LLMs path                                   |
| `translator_pth` | Translator LLMs path (for NatLan)                   |
| `model_name`     | result file identifier                              |
| `data_pth`       | dataset path                                        |
| `result_pth`     | result saving path                                  |
| `prompts_pth`    | translation prompts (for NatLan & Self-Translation) |
| `nshot`          | n-shot prompting                                    |
| `use_natlan`     | whether to use NatLan                               |
| `use_google`     | whether to use Google-MT                            |
| `use_self`       | whether to use Self-Translation                     |
| `target_lang`    | the original langauge of the questions              |
| `native_lang`    | the native (dominant) language of Speaker LLMs      |
| `google_api_key` | google translate API (for Google-MT)                |

P.S. Only one of the variables `use_natlan`, `use_google`, or `use_self` can be set to `True` at most. If all three are set to False, the default behavior is to have Speaker LLMs directly answer questions in the target language (`Original` in our paper).

This experiment is evaluated on [the C-Eval benchmark](https://github.com/hkust-nlp/ceval). You can directly submit the `.json` file generated after running `main.py` to [the official C-Eval platform](https://cevalbenchmark.com/index.html) for performance assessment on the test set.

P.P.S. In this experiment, all models involved should be the Instruct/Chat versions that can properly follow user instructions. For more detailed performance results, please refer to our paper.

## Citation

🚀 Under Review and Coming Soon...

