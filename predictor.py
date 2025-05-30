import hydra
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prompts import test_prompt

def set_up(base_model_name, lora_config):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2")
    lora_model = PeftModel.from_pretrained(base_model, lora_config)
    return tokenizer, base_model, lora_model


def messages_initialize(input):
    messages = [
    {"role":"system", "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."},
    {"role":"user", "content":test_prompt.format(input)}]
    return messages

def answer_generate(tokenizer, model, text):
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
    **inputs,
    max_new_tokens=512
    )
    output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return content

@hydra.main(version_base="1.3", config_path="./configs", config_name="config")
def main(cfg):
    import pandas as pd
    df_test = pd.read_csv(cfg.evaluation.evaluation_dataset)
    tokenizer, model, lora_model = set_up(cfg.evaluation.base_model, cfg.evaluation.lora_config)
    answers = {"Answer": []}
    for _, row in df_test.iterrows():
        input = row["Question"]
        messages = messages_initialize(input)
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
            )
        answer = answer_generate(tokenizer, model, tokenized)
        answers["Answer"].append(answer)
        print("The summarization of the medical record: {}".format(answer))
    eval_df = pd.DataFrame(answers)
    eval_df.to_csv("base_model_{}_eval_results.csv".format("surgery"),index=False)

if __name__=="__main__":
    main()