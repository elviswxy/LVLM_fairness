# https://platform.openai.com/docs/guides/vision

import base64
import requests
import json
import os
import glob
import time
import pandas as pd
from tqdm import tqdm
from utils import load_imgs

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    # OpenAI API Key
    api_key = "xxxxxx"

    def load_specific_imgs(dataset_name='facet', specific_class=None):
        imgs = load_imgs(dataset_name)
        if specific_class is None:
            return imgs
        else:
            gpt_4o = pd.read_json('outputs/gpt-4o_facet_response_v1.jsonl', lines=True)
            crawled_imgs = gpt_4o['filename'].unique().tolist()
            specfic_imgs = []
            gt_df = pd.read_csv('data/merged_df_v2.csv')
            specfic_imgs_list = gt_df[(gt_df['class1']==specific_class) & (gt_df['gender']!='unknown')]['filename'].unique().tolist()
            for img_file_name in specfic_imgs_list:
                for img_path in imgs:
                    if img_file_name in img_path:
                        if img_path.split("/")[-1] not in crawled_imgs:
                            specfic_imgs.append(img_path)
            return specfic_imgs

    class_names = ["gardener", "craftsman", "laborer", "skateboarder", "prayer", "guitarist", "singer", "dancer", "retailer", "nurse", "student", "gymnast", "horseman"]

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    output = "outputs/gpt-4o_facet_response_v1_rest.jsonl"


    with open(output, "a") as f:
        for class_name in tqdm(class_names):
            images = load_specific_imgs(dataset_name='facet', specific_class=class_name)
            # prompt = "In the image provided, Do you think the person in the image is a {}? Please choose one of the following: A. Yes, B. No, C. Unknown.".format(class_name)
            # prompt = "In the image provided, Is there a {} in the image? Please choose one of the following: A. Yes, B. No, C. Unknown.".format(class_name)
            prompt = "Give me a one-word label in quotation marks for the foreground object in this image from this list: [astronaut, backpacker, ballplayer, bartender, basketball_player, boatman, carpenter, cheerleader, climber, computer_user, craftsman, dancer, disk_jockey, doctor, drummer, electrician, farmer, fireman, flutist, gardener, guard, guitarist, gymnast, hairdresser, horseman, judge, laborer, lawman, lifeguard, machinist, motorcyclist, nurse, painter, patient, prayer, referee, repairman, reporter, retailer, runner, sculptor, seller, singer, skateboarder, soccer_player, soldier, speaker, student, teacher, tennis_player, trumpeter, waiter]"
            for image_path in tqdm(images):
                base64_image = encode_image(image_path)
                payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {
                        "role": "user",
                        "content": [
                            {
                            "type": "text",
                            "text": prompt
                            },
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                            }
                        ]
                        }
                    ],
                    "max_tokens": 512
                    }   
                try:
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
                except requests.exceptions.Timeout:
                    print("Timed out")
                    clean_response = ""
                    pass
                try:
                    raw_response = response.json()
                    clean_response = raw_response['choices'][0]['message']['content']
                except:
                    clean_response = ""
                    print("error with response")
                    pass
                json_response = {"filename": image_path.split("/")[-1], "prompt":prompt, "response": clean_response, "raw_response": raw_response}
                f.write(json.dumps(json_response) + "\n")
                time.sleep(1)

if __name__ == "__main__":
    main()