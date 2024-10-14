import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class NatLan:
    '''
    Native Language Prompting
    '''
    def __init__(self, translator_pth, device, translated_prompts, target_lang, native_lang):
        self.tokenizer = AutoTokenizer.from_pretrained(translator_pth, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(translator_pth, device_map="cuda", torch_dtype="auto", trust_remote_code=True)
        self.native_lang = native_lang
        self.target_lang = target_lang
        self.translated_prompts = translated_prompts
        self.device = device

    def target2native(self, content, cname, nshot=5):
        messages = [
            {"role": "system", "content": "You are a professional {}-{} translator. Translation rules: Proper nouns in English or Chinese need to be translated according to the {} domain-specific terms, retain the original meaning to the greatest extent, and follow the original format in the translation process.".format(self.target_lang, self.native_lang, cname)},
        ]
        translated_prompts = self.translated_prompts[cname]
        for i in range(nshot):
            messages.append({"role": "user", "content": f"Now help me translate the following sentence into {self.native_lang}, only return the translated sentence, the original sentence is: {translated_prompts[i][0]}"})
            messages.append({"role": "assistant", "content": translated_prompts[i][1]})
        messages.append({"role": "user", "content": f"Now help me translate the following sentence into {self.native_lang}, only return the translated sentence, the original sentence is: {content}"})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=False,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        messages.append({"role": "assistant", "content": response})
        history = messages

        return response, history