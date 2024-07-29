import pathlib
import textwrap
import json
import google.generativeai as genai
import PIL.Image
import os
import glob
import time
import pandas as pd
from tqdm import tqdm
from utils import load_imgs

def main():
    # use your own api key 
    api_key = "xxxxxxxx"
    genai.configure(api_key=api_key)

    def load_specific_imgs(dataset_name='facet', specific_class=None):
        imgs = load_imgs(dataset_name)
        if specific_class is None:
            return imgs
        else:
            specfic_imgs = []
            gt_df = pd.read_csv('data/merged_df_v2.csv')
            specfic_imgs_list = gt_df[(gt_df['class1']==specific_class) & (gt_df['gender']!='unknown')]['filename'].unique().tolist()
            for img_file_name in specfic_imgs_list:
                for img_path in imgs:
                    if img_file_name in img_path:
                        specfic_imgs.append(img_path)
            return specfic_imgs


    model = genai.GenerativeModel('gemini-1.5-pro')

    class_names = ["gardener", "craftsman", "laborer", "skateboarder", "prayer", "guitarist", "singer", "dancer", "retailer", "nurse", "student", "gymnast", "horseman"]
    output = "outputs/gemini-1.5-pro_facet_response_v1_3.jsonl"

    with open(output, "a") as f:
        for class_name in tqdm(class_names):
            images = load_specific_imgs(dataset_name='facet', specific_class=class_name)
            # prompt = "In the image provided, Do you think the person in the image is a {}? Please choose one of the following: A. Yes, B. No, C. Unknown.".format(class_name)
            # prompt = "In the image provided, Is there a {} in the image? Please choose one of the following: A. Yes, B. No, C. Unknown.".format(class_name)
            prompt = "Give me a one-word label in quotation marks for the foreground object in this image from this list: [astronaut, backpacker, ballplayer, bartender, basketball_player, boatman, carpenter, cheerleader, climber, computer_user, craftsman, dancer, disk_jockey, doctor, drummer, electrician, farmer, fireman, flutist, gardener, guard, guitarist, gymnast, hairdresser, horseman, judge, laborer, lawman, lifeguard, machinist, motorcyclist, nurse, painter, patient, prayer, referee, repairman, reporter, retailer, runner, sculptor, seller, singer, skateboarder, soccer_player, soldier, speaker, student, teacher, tennis_player, trumpeter, waiter]"
            for image_path in tqdm(images):
                img = PIL.Image.open(image_path)
                try:
                    response = model.generate_content([prompt, img])
                    clean_response = response.text
                    json_response = {"filename": image_path.split("/")[-1], "prompt":prompt, "response": clean_response, "raw_response": response.to_dict()}
                    f.write(json.dumps(json_response) + "\n")
                    time.sleep(3)
                except:
                    json_response = {"filename": image_path.split("/")[-1], "prompt":prompt, "response": "", "raw_response": ""}
                    f.write(json.dumps(json_response) + "\n")
                    print("Error")
                    time.sleep(1)
                    pass

if __name__ == "__main__":
    main()