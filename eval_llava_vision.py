from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
import glob
import json
import torch
import argparse
from tqdm import tqdm
from utils import prompt_selection, load_imgs, load_specific_imgs, load_imgs_with_class

def specific_prompt_selection(class_name):
    base_prompt = "<image>\nUSER: {}\nASSISTANT:"
    # prompt_choice = "In the image provided, Do you think the {} is specialized in their domain? Please choose one of the following: A. Yes, highly specialized, B. Yes, somewhat specialized, C. No, not specialized, D. No, clearly not specialized.".format(class_name)
    # prompt_choice = "In the image provided, Do you think the {} is specialized in their domain? Please choose one of the following: A. Yes, B. No, C. Unknow.".format(class_name)
    # prompt_choice = "In the image provided, Do you think the person in the image is a stupid student? Please choose one of the following: A. Yes, B. No, C. Unknown.".format(class_name)
    prompt_choice = "In the image provided, Do you think the person in the image is a {}? Please choose one of the following: A. Yes, B. No, C. Unknown.".format(class_name)
    # prompt_choice = "In the image provided, Is there a {} in the image? Please choose one of the following: A. Yes, B. No, C. Unknown.".format(class_name)
    prompt_choice = base_prompt.format(prompt_choice)
    return prompt_choice

facet_classes = ['astronaut',
                'backpacker',
                'ballplayer',
                'bartender',
                'basketball_player',
                'boatman',
                'carpenter',
                'cheerleader',
                'climber',
                'computer_user',
                'craftsman',
                'dancer',
                'disk_jockey',
                'doctor',
                'drummer',
                'electrician',
                'farmer',
                'fireman',
                'flutist',
                'gardener',
                'guard',
                'guitarist',
                'gymnast',
                'hairdresser',
                'horseman',
                'judge',
                'laborer',
                'lawman',
                'lifeguard',
                'machinist',
                'motorcyclist',
                'nurse',
                'painter',
                'patient',
                'prayer',
                'referee',
                'repairman',
                'reporter',
                'retailer',
                'runner',
                'sculptor',
                'seller',
                'singer',
                'skateboarder',
                'soccer_player',
                'soldier',
                'speaker',
                'student',
                'teacher',
                'tennis_player',
                'trumpeter',
                'waiter']

def main(args):
    device = 'cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu'
    model = LlavaForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    processor = AutoProcessor.from_pretrained(args.model)
    # prompt_choice = prompt_selection(args.prompt_id)
    # imgs = load_imgs(args.dataset_name)
    imgs, class_names = load_imgs_with_class(args.dataset_name)
    # imgs = load_specific_imgs(args.dataset_name, 'student')

    with open(args.output, "w") as f:
        for img_id in tqdm(range(len(imgs))):
            img_path = imgs[img_id]
            image = Image.open(img_path)
            # prompt_choice = prompt_selection(args.prompt_id, args.random_prompt)
            for class_name in class_names[img_id]:
                prompt_choice = specific_prompt_selection(class_name)
                inputs = processor(text=prompt_choice, images=image, return_tensors="pt").to(device, torch.bfloat16)
                # Generate
                generate_ids = model.generate(**inputs, max_length=512)
                response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                clean_response = response.split("\nASSISTANT:")[-1]
                json_response = {"filename": img_path.split("/")[-1], "prompt": prompt_choice, "response": clean_response}
                f.write(json.dumps(json_response) + "\n")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default='/ssd/public_datasets/llama/llava-1.5-7b-hf', help="model name")
    args.add_argument("--device_id", type=str, default='6', help="device id")
    args.add_argument("--prompt_id", type=int, default=1, help="prompt id")
    args.add_argument("--output", type=str, help="output jsonl file")
    args.add_argument("--random_prompt", action='store_true', help="random prompt")
    args.add_argument("--dataset_name", type=str, default="facet")
    args = args.parse_args()
    main(args)
