from openai import OpenAI
from datasets import load_from_disk
import json
import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    gpt_api_key = '<YOUR-API-KEY>'
    model_name = 'gpt-4o-mini'

    comp_pair = ['google', 'phi3-small']  # and ['gpt', 'google']

    src_lang = {
        'Arabic': 'AR_XY',
        'Chinese': 'ZH_CN',
        'French': 'FR_FR',
        'German': 'DE_DE',
        'Japanese': 'JA_JP'
    }

    courses = ['anatomy', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_medicine', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'virology']

    GPT = OpenAI(api_key=gpt_api_key)
    bar_format = "{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

    for lang_nl, lang_mark in src_lang.items():
        result_pth = f'./DT/{model_name}_{comp_pair[0]}-{comp_pair[1]}_{lang_nl}_DT.json'

        with open(f'./trans/{comp_pair[0]}_{lang_nl}_test_trans.json', 'r') as f:
            trans_0 = json.load(f)
        with open(f'./trans/{comp_pair[1]}_{lang_nl}_test_trans.json', 'r') as f:
            trans_1 = json.load(f)

        comp_result = {}
        for c in courses:
            win_0 = 0
            win_1 = 0
            none = 0
            tmp_result = {}

            messages = [
                {"role": "system", "content": f"As a professional quality assessor in the field of {c}, you are required to determine which of the two provided questions contains a greater number of domain-specific terms or phrases conducive to eliciting domain knowledge. Return the id of the question with higher quality, either 0 or 1. If the quality of the two translations is equivalent, return: none."},
                {"role": "user", "content": "0: Question:\nAmong the following substances, which one, when completely combusted, produces products other than carbon dioxide and water?____.\nChoices:\nA. Methane\nB. Ethylene\nC. Vinyl chloride\nD. Ethanol\nAnswer:\n1: Question:\nWhen the following substances are completely burned, the products include carbon dioxide and water, and other substances____.\nChoices:\nA. Methane\nB. Ethylene\nC. Vinyl chloride\nD. Ethanol\nAnswer:"},
                {"role": "assistant", "content": "0"},
                {"role": "user", "content": "0: Question:\nWhen the following substances are completely burned, the products include carbon dioxide and water, and other substances____.\nChoices:\nA. Methane\nB. Ethylene\nC. Vinyl chloride\nD. Ethanol\nAnswer:\n1: Question:\nAmong the following substances, which one, when completely combusted, produces products other than carbon dioxide and water?____.\nChoices:\nA. Methane\nB. Ethylene\nC. Vinyl chloride\nD. Ethanol\nAnswer:"},
                {"role": "assistant", "content": "1"},
                {"role": "user", "content": "0: Question:\nAmong the following substances, which one, when completely combusted, produces products other than carbon dioxide and water?____.\nChoices:\nA. Methane\nB. Ethylene\nC. Vinyl chloride\nD. Ethanol\nAnswer:\n1: Question:\nAmong the following substances, which one, when completely combusted, produces products other than carbon dioxide and water?____.\nChoices:\nA. Methane\nB. Ethylene\nC. Vinyl chloride\nD. Ethanol\nAnswer:"},
                {"role": "assistant", "content": "none"},
            ]

            tmp_trans_0 = trans_0[c]
            tmp_trans_1 = trans_1[c]

            for i, _ in tqdm.tqdm(enumerate(tmp_trans_0), total=len(tmp_trans_0), bar_format=bar_format):
                content = {"role": "user", "content": f"0: {tmp_trans_0[i]}\n1: {tmp_trans_1[i]}"}
                messages.append(content)

                response = GPT.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    top_p=1,
                    max_tokens=5,
                ).choices[0].message.content

                # print(response)

                if response == '1':
                    win_1 += 1
                elif response == '0':
                    win_0 += 1
                else:
                    none += 1

                tmp_result[i] = response
                messgaes = messages[:-1]
            
            print(f"{comp_pair[0]}-{comp_pair[1]}, {lang_nl}-{c}, {win_0}:{none}:{win_1}")
            comp_result[c] = tmp_result

        f = open(result_pth, 'w')
        comp_result = json.dumps(comp_result)
        f.write(comp_result)
        f.close()