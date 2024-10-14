import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_from_disk
from torch.utils.data import DataLoader
import tqdm
import json
import argparse


if __name__ == '__main__':
    torch.random.manual_seed(42)
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--translator", type=str)
    args = parser.parse_args()
    
    translator = args.translator
    assert translator in ['human', 'none']

    model_pth = '<SPEAKER-LLM-PTH>'
    model_name = '<SPEAKER-LLM-NAME>'

    src_lang = {
        'Arabic': 'AR_XY',
        'Chinese': 'ZH_CN',
        'French': 'FR_FR',
        'German': 'DE_DE',
        'Japanese': 'JA_JP'
    }
    if translator == 'human':
        src_lang = {
            'Arabic': 'AR_XY',
        }
    courses = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

    prompt_pth = '<MMLU-DATASET-PTH>'

    M3LU_pth = '<MMMLU-DATASET-PTH>'
    M2LU_pth = '<MMLU-DATASET-PTH>'

    tokenizer = AutoTokenizer.from_pretrained(model_pth, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_pth, device_map="cuda", torch_dtype="auto", trust_remote_code=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    generation_args = {
        "max_new_tokens": 2,
        "return_full_text": False,
        "do_sample": False,
        "top_p": None,
    }

    bar_format = "{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

    for lang_nl, lang_mark in src_lang.items():
        if translator == 'none':
            dataset = load_from_disk(M3LU_pth + lang_mark)

        result_pth = f'./answers/{model_name}_{translator}_{lang_nl}_answer.json'
        
        predicted_answer = {}
        for c in courses:
            print(f"Processing {model_name}-{translator}-{lang_nl}-{c}")
            tmp_answer = {}

            if translator == 'human':
                tmp_dataset = load_from_disk(M2LU_pth + c)['test']
            else:
                tmp_dataset = dataset.filter(lambda x: x['Subject'] == c)

            prompts = load_from_disk(prompt_pth + c)['dev']
            messages = [
                {"role": "system", "content": f"You are a professional {c} expert, and you are currently answering a multiple-choice question about {c}, you need to provide only one option as the answer based on the question, and you only need to return one single capital character as the answer."},
            ]
            for i in range(5):
                content = f"Question:\n{prompts[i]['question']}\nChoices:\nA.{prompts[i]['choices'][0]}\nB.{prompts[i]['choices'][1]}\nC.{prompts[i]['choices'][2]}\nD.{prompts[i]['choices'][3]}\nAnswer:"
                messages.append({"role": "user", "content": content})
                a = ['A', 'B', 'C', 'D']
                messages.append({"role": "assistant", "content": a[prompts[i]['answer']]})

            for i, data in tqdm.tqdm(enumerate(tmp_dataset), total=len(tmp_dataset), bar_format=bar_format):
                if translator == 'none':
                    messages.append({"role": "user", "content": f"Question:\n{data['Question']}\nChoices:\nA.{data['A']}\nB.{data['B']}\nC.{data['C']}\nD.{data['D']}\nAnswer:"})
                else: 
                    messages.append({"role": "user", "content": f"Question:\n{data['question']}\nChoices:\nA.{data['choices'][0]}\nB.{data['choices'][1]}\nC.{data['choices'][2]}\nD.{data['choices'][3]}\nAnswer:"})

                # print(messages)

                response = pipe(messages, **generation_args)[0]['generated_text']
                response = response.replace(' ', '')
                if len(response) > 1:
                    response = None
                tmp_answer[i] = response

                # print(response)

                messages = messages[:-1]
                # break
            
            predicted_answer[c] = tmp_answer
        
        f = open(result_pth, 'w')
        predicted_answer = json.dumps(predicted_answer)
        f.write(predicted_answer)
        f.close()