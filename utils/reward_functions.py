import re

def citation_reward_function(completions, **kwargs):
    pattern =  r'^<cited_answer>\s*<answer>.*?</answer>\s*<citations>\s*<source_id>.*?</source_id>\s*</citations>\s*</cited_answer>$'
    contents = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, content) for content in contents]
    return [1.0 if match else 0.0 for match in matches]