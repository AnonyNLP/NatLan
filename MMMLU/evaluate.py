import json
from datasets import load_from_disk


if __name__ == '__main__':
    model_name = '<SPEAKER-LLM-NAME>'
    translator = '<TRANSLATOR-NAME>'  # 'gpt' for NatLan, 'google' for Google-MT, 'self' for Self-Translation, 'none' for direct answering non-native questions, 'human' for answering human-constructed English questions (Gold)

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
    ground_truth_pth = '<MMLU-DATASET-PTH>'

    for lang_nl, lang_mark in src_lang.items():
        total_q = 0
        acc_q = 0

        predicted_pth = f'./answers/{model_name}_{translator}_{lang_nl}_answer.json'
        with open(predicted_pth, 'r') as file:
            predict = json.load(file)

        for c in courses:
            ground_truth = load_from_disk(ground_truth_pth + c)['test']

            c_total = 0
            c_true = 0

            tmp_predict = predict[c]

            for i in tmp_predict:
                total_q += 1
                c_total += 1

                if tmp_predict[i] == {"0": "A", "1": "B", "2": "C", "3": "D"}[str(ground_truth['answer'][int(i)])]:
                    acc_q += 1
                    c_true += 1

            c_acc = c_true / c_total

        acc = acc_q / total_q

        print(f"{lang_nl}-Acc: {acc*100}%")