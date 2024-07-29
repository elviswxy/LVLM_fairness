import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import argparse
from tqdm import tqdm
import json
import os
import glob
from utils import prompt_selection

def load_imgs():
    imgs = []
    img_folder = ["/ssd/public_datasets/FACET/imgs_1", "/ssd/public_datasets/FACET/imgs_2", "/ssd/public_datasets/FACET/imgs_3"]
    for folder in img_folder:
        for img_path in glob.glob("{}/*.jpg".format(folder)):
            imgs.append(img_path)
    return imgs

def main(args):
    device = 'cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu'
    tokenizer = LlamaTokenizer.from_pretrained('/ssd/public_datasets/llama/vicuna-7b-v1.5')
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()

    imgs = load_imgs()
    # query = prompt_selection(args.prompt_id)

    with open(args.output, "w") as f:
        for img_path in tqdm(imgs):
            query = prompt_selection(args.prompt_id, args.random_prompt)    
            image = Image.open(img_path).convert('RGB')          
            inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
            inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
                'images': [[inputs['images'][0].to(device).to(torch.bfloat16)]],
            }
            gen_kwargs = {"max_length": 2048, "do_sample": False}

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                clean_response = tokenizer.decode(outputs[0]).replace('</s>', '')
                # print(clean_response)
                json_response = {"filename": img_path.split("/")[-1], "prompt": query, "response": clean_response}
                f.write(json.dumps(json_response) + "\n")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default='/ssd/public_datasets/llama/cogvlm-chat-hf', help="model name")
    args.add_argument("--device_id", type=str, default='4', help="device id")
    args.add_argument("--prompt_id", type=int, default=1, help="prompt id")
    args.add_argument("--output", type=str, default="outputs/cogvlm_chat_facet_response_1.jsonl", help="output jsonl file")
    args.add_argument("--random_prompt", action='store_true', help="random prompt")
    args = args.parse_args()
    main(args)