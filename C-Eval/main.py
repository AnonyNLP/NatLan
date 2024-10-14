import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_from_disk
from torch.utils.data import DataLoader
import json
import tqdm

from natlan import NatLan
from google_trans import google_translate
from self_trans import Self_Translate
from utils import generate_natlan_prompts, generate_prompts


def main():
    torch.random.manual_seed(42)
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    # -----------------------------------------------------------
    model_pth = '<SPEAKER-PATH>'
    translaor_pth = '<TRANSLATOR-PATH>'
    model_name = '<SPEAKER-NAME>' + '<TRANSLATOR-NAME>'
    google_api_key = '<GOOGLE-API-KEY>'

    data_type = 'C-Eval'
    data_pth = '<DATA-PATH>'
    result_pth = '<RESULT-PATH>'
    prompts_pth = '<QA-PROMPTS-PATH>'
    nshot = 5

    use_natlan = True
    use_google = False
    use_self = False
    assert (use_natlan + use_google + use_self) <= 1  # at most one can be True

    target_lang = 'Chinese'
    native_lang = 'English'
    subject_name = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics', 'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics', 'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology', 'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine', 'college_economics', 'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification', 'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography', 'modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history', 'civil_servant', 'sports_science', 'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']
    # -----------------------------------------------------------
    result_pth = result_pth + f"{model_name}-{str(nshot)}shot-"
    result_pth = result_pth + f"NatLan-{data_type}.json" if use_natlan else result_pth + f"None-{data_type}.json"

    with open(prompts_pth, 'r') as file:
        translated_prompts = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(model_pth, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_pth, device_map="cuda", torch_dtype="auto", trust_remote_code=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    generation_args = {
        "max_new_tokens": 512,
        "return_full_text": False,
        "do_sample": False,
        "top_p": None,
    }
    if use_natlan:
        natlan = NatLan(translaor_pth=translaor_pth, device=device, translated_prompts=translated_prompts, target_lang=target_lang, native_lang=native_lang)
    elif use_self:
        self_translate = Self_Translate(pipe=pipe, generation_args=generation_args, device=device, translated_prompts=translated_prompts, target_lang=target_lang, native_lang=native_lang)

    answer = {}
    for cname in subject_name:
        dataset = load_from_disk(data_pth + cname)
        cname_prompts = translated_prompts[cname]
        if (use_natlan + use_google + use_self) == 1:
            messages = generate_natlan_prompts(translated_prompts=cname_prompts, examples=dataset['dev'], subject_name=cname, nshot=nshot)  # max to 5-shot
        else:
            messages = generate_prompts(examples=dataset['dev'], subject_name=cname, nshot=nshot)
        
        dataset = dataset['test']
        data_loader = DataLoader(dataset, shuffle=False, batch_size=1)
        data_iter = tqdm.tqdm(enumerate(data_loader), desc=f"Evaluating on {data_type}-{cname}", total=len(data_loader), bar_format="{l_bar}{r_bar}")
        
        ans = {}
        for i, data in data_iter:
            user_input = f"Question:\n{data['question'][0]}\nChoices:\nA.{data['A'][0]}\nB.{data['B'][0]}\nC.{data['C'][0]}\nD.{data['D'][0]}\nAnswer:"

            if use_natlan:  # NatLan
                msg, _ = natlan.target2native(content=user_input, cname=cname)
                messages.append({'role': 'user', 'content': msg})
            elif use_google:  # Google-MT
                msg = google_translate(text=user_input, api_key=google_api_key)
                messages.append({'role': 'user', 'content': msg})
            elif use_self:  # Self-Translation
                msg, _ = self_translate.target2native(content=user_input, cname=cname)
                messages.append({'role': 'user', 'content': msg})
            else:
                messages.append({'role': 'user', 'content': user_input})

            response = pipe(messages, **generation_args)[0]['generated_text']
            messages = messages[:-1]
            response = response.replace(' ', '')
            if len(response) > 1:
                response = None
            ans[str(i)] = response

        answer[cname] = ans
    
    # save the answers by the C-Eval test format
    answer = json.dumps(answer)
    f = open(result_pth, 'w')
    f.write(answer)
    f.close()


if __name__ == '__main__':
    main()