from openai import OpenAI
from datasets import load_from_disk
import json
import tqdm
from torch.utils.data import DataLoader
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import time


def translate_text_api_key(text, source, api_key="AIzaSyATOSHLJ7ihPuCnbYptpUmpRGfVsvdh6T4", target="en"):
    url = "https://translation.googleapis.com/language/translate/v2"
    params = {
        'q': text,
        'source': source,
        'target': target,
        'format': 'text',
        'key': api_key
    }
    retries = 100
    # Configure retry strategy
    retry_strategy = Retry(
        total=retries,  # Total number of retries to allow
        read=retries,  # Retry on read errors
        connect=retries,  # Retry on connection errors
        status_forcelist=[500, 502, 503, 504],  # Status codes to retry
        backoff_factor=0.3  # Time between retries
    )
    
    # Mount it with a session
    adapter = HTTPAdapter(max_retries=retry_strategy)
    with requests.Session() as session:
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        try:
            # Make the request
            response = session.get(url, params=params)
            # Ensure we handle the response appropriately
            response.raise_for_status()  # Will raise an exception for HTTP error codes
            return response.json()['data']['translations'][0]['translatedText']

        except requests.exceptions.HTTPError as err:
            # Check for 400 client error
            if response.status_code == 400:
                return "[ERROR]"  # Replace with your special string
            else:
                raise  # Re-raise for other HTTP errors


if __name__ == '__main__':
    src_lang = {
        'Arabic': 'AR_XY',
        'Chinese': 'ZH_CN',
        'French': 'FR_FR',
        'German': 'DE_DE',
        'Japanese': 'JA_JP'
    }
    google_src = {
        'Arabic': 'ar',
        'Chinese': 'zh_CN',
        'French': 'fr',
        'German': 'de',
        'Japanese': 'ja'
    }
    courses = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    dataset_pth = '<MMMLU-DATASET-PTH>'

    for lang_nl, lang_mark in src_lang.items():
        test_trans = {}

        for c in courses:
            print(f"processing {lang_nl}-{c} ...")
            course_trans = []

            dataset = load_from_disk(dataset_pth + lang_mark).filter(lambda x: x['Subject'] == c)
            data_loader = DataLoader(dataset, shuffle=False, batch_size=1)
            data_iter = tqdm.tqdm(enumerate(data_loader), desc=f"{lang_nl}-{c}", total=len(data_loader), bar_format="{l_bar}{r_bar}")

            for i, data in data_iter:
                content = f"Question:\n{data['Question'][0]}\nChoices:\nA.{data['A'][0]}\nB.{data['B'][0]}\nC.{data['C'][0]}\nD.{data['D'][0]}\nAnswer:"
                
                # print(content)

                response = translate_text_api_key(text=content, source=google_src[lang_nl])

                # print(response)

                course_trans.append(response)

            test_trans[c] = course_trans

        f = open(f'./trans/google_{lang_nl}_test_trans.json', 'w')
        test_trans = json.dumps(test_trans)
        f.write(test_trans)
        f.close()