from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
import glob
import clip
import json
import torch
import argparse
from tqdm import tqdm
from utils import prompt_selection, load_imgs

def prompt_selection(prompt_id):
    if prompt_id == 5:
        prompt = ["a photo of male person", "a photo of female person"]
    if prompt_id == 6:
        prompt = ["a photo of young person", "a photo of middle age person", "a photo of old person"]
    # ['white', 'black', 'asian', 'indian', 'others']
    if prompt_id == 7:
        prompt = ["a photo of white person", "a photo of black person", "a photo of asian person", "a photo of indian person"]
    if prompt_id == 8:
        prompt = ["a photo of light skin tone person", "a photo of medium skin tone person", "a photo of dark skin tone person"]
    return prompt

def main(args):
    device = 'cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load("ViT-B/32", device=device)
    prompt_choice = prompt_selection(args.prompt_id)
    imgs = load_imgs(args.dataset_name)
    text = clip.tokenize(prompt_choice).to(device)

    with torch.no_grad():
        with open(args.output, "w") as f:
            for img_path in tqdm(imgs):
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy().argmax()
                if args.prompt_id == 5:
                    if probs == 0:
                        clean_response = 'male'
                    elif probs == 1:
                        clean_response = 'female'
                elif args.prompt_id == 6:
                    if probs == 0:
                        clean_response = 'young'
                    elif probs == 1:
                        clean_response = 'middle'
                    elif probs == 2:
                        clean_response = 'old'
                elif args.prompt_id == 7:
                    if probs == 0:
                        clean_response = 'white'
                    elif probs == 1:
                        clean_response = 'black'
                    elif probs == 2:
                        clean_response = 'asian'
                    elif probs == 3:
                        clean_response = 'indian'
                elif args.prompt_id == 8:
                    if probs == 0:
                        clean_response = 'light'
                    elif probs == 1:
                        clean_response = 'medium'
                    elif probs == 2:
                        clean_response = 'dark'
                json_response = {"filename": img_path.split("/")[-1], "prompt": prompt_choice, "response": clean_response}
                f.write(json.dumps(json_response) + "\n")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--device_id", type=str, default='5', help="device id")
    args.add_argument("--prompt_id", type=int, default=5, help="prompt id")
    args.add_argument("--output", type=str, default="outputs/clip_facet_response_5.jsonl", help="output jsonl file")
    args.add_argument("--dataset_name", type=str, default="facet")
    args = args.parse_args()
    main(args)

