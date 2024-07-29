from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
import glob
import json
import clip
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer

def load_jsonl(file_path):
    data = {}
    with open(file_path, "r") as f:
        for line in f:
            line_json = json.loads(line)
            data[line_json['filename']] = line_json['response']
    return data

def facet_class_prediction(args):
    gt_classes = ['astronaut', 'backpacker', 'ballplayer', 'bartender', 'basketball_player', 'boatman', 'carpenter', 'cheerleader', 'climber', 'computer_user', 'craftsman',
                'dancer', 'disk_jockey', 'doctor', 'drummer', 'electrician', 'farmer', 'fireman', 'flutist', 'gardener', 'guard', 'guitarist', 'gymnast', 'hairdresser',
                'horseman', 'judge', 'laborer', 'lawman', 'lifeguard', 'machinist', 'motorcyclist', 'nurse', 'painter', 'patient', 'prayer', 'referee', 'repairman', 'reporter',
                'retailer', 'runner', 'sculptor', 'seller', 'singer', 'skateboarder', 'soccer_player', 'soldier', 'speaker', 'student', 'teacher', 'tennis_player', 'trumpeter',
                'waiter']
    # load data
    data = load_jsonl(args.prediction)
    gt_df = pd.read_csv('data/annotations.csv')
    sub_gt_dic = gt_df[['person_id', 'filename', 'class1']].to_dict('records')
    # Load the model
    device = "cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('sentence-transformers/sentence-t5-xxl').to(device)
    # calualate the cosine similarity
    with open(args.output, 'w') as f:
        with torch.no_grad():
            if args.prompt_id == 2:
                gt_inputs = [f"{c}" for c in gt_classes]
            else:
                gt_inputs = [f"a photo of a {c}" for c in gt_classes]
            gt_features = torch.from_numpy(model.encode(gt_inputs))
            # gt_features /= gt_features.norm(dim=-1, keepdim=True)
            for item in tqdm(sub_gt_dic):
                filename = item['filename']
                if filename in data:
                    predict_features = model.encode([data[filename].strip()])
                    predict_features = torch.from_numpy(np.repeat(predict_features, len(gt_classes), 0))
                    # predict_features /= predict_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * predict_features @ gt_features.T).softmax(dim=-1)
                    predict_class = gt_classes[similarity[0].argmax().item()]
                    item['predict_class'] = predict_class
                    f.write(json.dumps(item) + '\n')

def normalize_string(s):
    s = re.sub(r'[^\w\s]', '', s)
    s = s.lower()
    return s

def facet_class_prediction_by_re(args):
    gt_classes = ['astronaut', 'backpacker', 'ballplayer', 'bartender', 'basketball_player', 'boatman', 'carpenter', 'cheerleader', 'climber', 'computer_user', 'craftsman',
                'dancer', 'disk_jockey', 'doctor', 'drummer', 'electrician', 'farmer', 'fireman', 'flutist', 'gardener', 'guard', 'guitarist', 'gymnast', 'hairdresser',
                'horseman', 'judge', 'laborer', 'lawman', 'lifeguard', 'machinist', 'motorcyclist', 'nurse', 'painter', 'patient', 'prayer', 'referee', 'repairman', 'reporter',
                'retailer', 'runner', 'sculptor', 'seller', 'singer', 'skateboarder', 'soccer_player', 'soldier', 'speaker', 'student', 'teacher', 'tennis_player', 'trumpeter',
                'waiter']
    # load data
    data = load_jsonl(args.prediction)
    gt_df = pd.read_csv('data/annotations.csv')
    sub_gt_dic = gt_df[['person_id', 'filename', 'class1']].to_dict('records')
    # calualate the cosine similarity
    with open(args.output, 'w') as f:
        with torch.no_grad():
            if args.prompt_id == 2:
                for item in tqdm(sub_gt_dic):
                    filename = item['filename']
                    if filename in data:
                        predict_features = normalize_string(data[filename].strip())
                        item['predict_class'] = predict_features
                        for i in gt_classes:
                            normalized_class = normalize_string(i)
                            if normalized_class == predict_features:
                                item['predict_class'] = i
                                continue   
                        f.write(json.dumps(item) + '\n')

def facet_gender_age_prediciton(args):
    # Load the model
    device = "cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('sentence-transformers/sentence-t5-xxl').to(device)
    if args.prompt_id == 5:
        # load data
        gt_classes = ['male', 'female']
        data = load_jsonl(args.prediction)
        gt_df = pd.read_csv('data/annotations.csv')
        sub_gt_dic = gt_df[['person_id', 'filename', 'gender_presentation_masc', 'gender_presentation_fem']].to_dict('records')
        # calualate the cosine similarity
        with open(args.output, 'w') as f:
            with torch.no_grad():
                gt_inputs = [f"{c}" for c in gt_classes]
                gt_features = torch.from_numpy(model.encode(gt_inputs))
                for item in tqdm(sub_gt_dic):
                    filename = item['filename']
                    if filename in data:
                        predict_features = model.encode([data[filename].strip()])
                        predict_features = torch.from_numpy(np.repeat(predict_features, len(gt_classes), 0))
                        similarity = (100.0 * predict_features @ gt_features.T).softmax(dim=-1)
                        predict_class = gt_classes[similarity[0].argmax().item()]
                        item['predict_class'] = predict_class
                        f.write(json.dumps(item) + '\n')
    elif args.prompt_id == 6:
        gt_classes = ['young', 'middle', 'old']
        data = load_jsonl(args.prediction)
        gt_df = pd.read_csv('data/annotations.csv')
        sub_gt_dic = gt_df[['person_id', 'filename', 'age_presentation_young', 'age_presentation_middle', 'age_presentation_older']].to_dict('records')
        # calualate the cosine similarity
        with open(args.output, 'w') as f:
            with torch.no_grad():
                gt_inputs = [f"{c}" for c in gt_classes]
                gt_features = torch.from_numpy(model.encode(gt_inputs))
                for item in tqdm(sub_gt_dic):
                    filename = item['filename']
                    if filename in data:
                        predict_features = model.encode([data[filename].strip()])
                        predict_features = torch.from_numpy(np.repeat(predict_features, len(gt_classes), 0))
                        similarity = (100.0 * predict_features @ gt_features.T).softmax(dim=-1)
                        predict_class = gt_classes[similarity[0].argmax().item()]
                        item['predict_class'] = predict_class
                        f.write(json.dumps(item) + '\n')

def utkface_gender_age_prediction(args):
    # Load the model
    device = "cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('sentence-transformers/sentence-t5-xxl').to(device)
    if args.prompt_id == 5:
        # load data
        gt_classes = ['male', 'female']
        data = load_jsonl(args.prediction)
        sub_gt_dic = pd.read_csv('data/utkface_annotations.csv').to_dict('records')
        # calualate the cosine similarity
        with open(args.output, 'w') as f:
            with torch.no_grad():
                gt_inputs = [f"{c}" for c in gt_classes]
                gt_features = torch.from_numpy(model.encode(gt_inputs))
                for item in tqdm(sub_gt_dic):
                    filename = item['filename']
                    if filename in data:
                        predict_features = model.encode([data[filename].strip()])
                        predict_features = torch.from_numpy(np.repeat(predict_features, len(gt_classes), 0))
                        similarity = (100.0 * predict_features @ gt_features.T).softmax(dim=-1)
                        predict_class = gt_classes[similarity[0].argmax().item()]
                        item['predict_class'] = predict_class
                        f.write(json.dumps(item) + '\n')
    elif args.prompt_id == 7:
        gt_classes = ['white', 'black', 'asian', 'indian', 'others']
        data = load_jsonl(args.prediction)
        sub_gt_dic = pd.read_csv('data/utkface_annotations.csv').to_dict('records')
        # calualate the cosine similarity
        with open(args.output, 'w') as f:
            with torch.no_grad():
                gt_inputs = [f"{c}" for c in gt_classes]
                gt_features = torch.from_numpy(model.encode(gt_inputs))
                for item in tqdm(sub_gt_dic):
                    filename = item['filename']
                    if filename in data:
                        predict_features = model.encode([data[filename].strip()])
                        predict_features = torch.from_numpy(np.repeat(predict_features, len(gt_classes), 0))
                        similarity = (100.0 * predict_features @ gt_features.T).softmax(dim=-1)
                        predict_class = gt_classes[similarity[0].argmax().item()]
                        item['predict_class'] = predict_class
                        f.write(json.dumps(item) + '\n')

def main(args):
    if args.task == 'facet_class_prediction':
        facet_class_prediction(args)
    elif args.task == 'facet_gender_age_prediciton':
        facet_gender_age_prediciton(args)
    elif args.task == 'utkface_gender_age_prediction':
        utkface_gender_age_prediction(args)
    elif args.task == 'facet_re':
        facet_class_prediction_by_re(args)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default='ViT-B/32', help="model name")
    args.add_argument("--task", type=str, default='facet_gender_age_prediciton', help="task name")
    args.add_argument("--device_id", type=str, default='0', help="device id")
    args.add_argument("--prompt_id", type=int, default=1, help="prompt id")
    args.add_argument("--prediction", type=str, default='outputs/llava_7b_facet_response_1.jsonl', help="prediction jsonl file")
    args.add_argument("--output", type=str, default='outputs/llava_7b_facet_response_1_t5_encode_formatted.jsonl', help="output jsonl file")
    args = args.parse_args()
    main(args)
