
class Self_Translate:
    '''
    Related Work: Self-Translation
    '''
    def __init__(self, pipe, generation_args, device, translated_prompts, target_lang, native_lang):
        self.pipe = pipe
        self.generation_args = generation_args
        self.native_lang = native_lang
        self.target_lang = target_lang
        self.translated_prompts = translated_prompts
        self.device = device

    def target2native(self, content, cname, nshot=5):
        messages = [
            {"role": "system", "content": "You are a professional {}-{} translator. Translation rules: Proper nouns in English or Chinese need to be retained without translation, retain the original meaning to the greatest extent, and follow the original format in the translation process.".format(self.target_lang, self.native_lang)},
        ]
        translated_prompts = self.translated_prompts[cname]
        for i in range(nshot):
            messages.append({"role": "user", "content": f"Now help me translate the following sentence into {self.native_lang}, only return the translated sentence, the original sentence is: {translated_prompts[i][0]}"})
            messages.append({"role": "assistant", "content": translated_prompts[i][1]})
        messages.append({"role": "user", "content": f"Now help me translate the following sentence into {self.native_lang}, only return the translated sentence, the original sentence is: {content}"})

        response = self.pipe(messages, **self.generation_args)[0]['generated_text']
        messages.append({"role": "assistant", "content": response})
        history = messages

        return response, history