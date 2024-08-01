
def generate_natlan_prompts(translated_prompts, examples, subject_name, nshot):
    '''
    Q&A Prompts of NatLan
    '''
    messages = [
        {"role": "system", "content": f"You are a professional {subject_name} expert, and you are currently answering a multiple-choice question about {subject_name}, you need to provide only one option as the answer based on the question, and you only need to return one single capital character as the answer."},
    ]
    if nshot == 0:
        return messages
    elif nshot > 5:
        raise ValueError('Too many examples to show, you need to set the nshot parameter to be <= 5.')
    else:
        for i in range(nshot):
            msg = {"role": "user", "content": translated_prompts[i][1]}
            messages.append(msg)

            msg = {"role": "assistant", "content": examples[i]['answer']}
            messages.append(msg)

        return messages
        

def generate_prompts(examples, subject_name, nshot):
    '''
    Original Q&A Prompts w/o translation
    '''
    messages = [
        {"role": "system", "content": f"You are a professional {subject_name} expert, and you are currently answering a multiple-choice question about {subject_name}, you need to provide only one option as the answer based on the question, and you only need to return one single capital character as the answer."},
    ]
    if nshot == 0:
        return messages
    elif nshot > 5:
        raise ValueError('Too many examples to show, you need to set the nshot parameter to be <= 5.')
    else:
        for i in range(nshot):
            msg = {"role": "user", "content": f"Question:\n{examples[i]['question']}\nChoices:\nA.{examples[i]['A']}\nB.{examples[i]['B']}\nC.{examples[i]['C']}\nD.{examples[i]['D']}\nAnswer:"}
            messages.append(msg)

            msg = {"role": "assistant", "content": examples[i]['answer']}
            messages.append(msg)

        return messages