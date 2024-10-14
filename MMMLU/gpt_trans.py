from openai import OpenAI
from datasets import load_from_disk
import json
import tqdm
import time
from torch.utils.data import DataLoader


if __name__ == '__main__':
    gpt_api_key = '<YOUR-API-KEY>'
    model_name = 'gpt-4o-mini'
    src_lang = {
        'Arabic': 'AR_XY',
        'Chinese': 'ZH_CN',
        'French': 'FR_FR',
        'German': 'DE_DE',
        'Japanese': 'JA_JP'
    }
    courses = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    
    prompts_pth = '<TRANSLATE-PROMPT-PTH>'
    with open(prompts_pth, 'r') as file:
        trans_prompts = json.load(file)

    GPT = OpenAI(api_key=gpt_api_key)

    dataset_pth = '<MMMLU-DATASET-PTH>'
    for lang_nl, lang_mark in src_lang.items():
        test_trans = {}

        for c in courses:
            print(f"processing {lang_nl}-{c} ...")
            course_trans = []

            messages = [
                {"role": "system", "content": f"You are a professional {lang_nl}-English translator. Translation rules: Proper nouns in English or {lang_nl} need to be translated according to the {c} domain-specific terms, retain the original meaning to the greatest extent, and follow the original format in the translation process."},
            ]
            for i in range(2):
                messages.append({"role": "user", "content": f"Now help me translate the following sentence into English, only return the translated sentence, the original sentence is: {trans_prompts[lang_mark][c][i][0]}"})
                messages.append({"role": "assistant", "content": trans_prompts[lang_mark][c][i][1]})
            

            dataset = load_from_disk(dataset_pth + lang_mark).filter(lambda x: x['Subject'] == c)
            data_loader = DataLoader(dataset, shuffle=False, batch_size=1)
            data_iter = tqdm.tqdm(enumerate(data_loader), desc=f"{lang_nl}-{c}", total=len(data_loader), bar_format="{l_bar}{r_bar}")

            for i, data in data_iter:
                content = f"Question:\n{data['Question'][0]}\nChoices:\nA.{data['A'][0]}\nB.{data['B'][0]}\nC.{data['C'][0]}\nD.{data['D'][0]}\nAnswer:"
                messages.append({"role": "user", "content": f"Now help me translate the following sentence into English, only return the translated sentence, the original sentence is: {content}"})

                # print(messages)

                response = GPT.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    top_p=1,
                ).choices[0].message.content
            
                # print(response)

                course_trans.append(response)

                messages = messages[:-1]

            test_trans[c] = course_trans

        f = open(f'./trans/gpt_{lang_nl}_test_trans.json', 'w')
        test_trans = json.dumps(test_trans)
        f.write(test_trans)
        f.close()