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
from tqdm import tqdm

def load_jsonl(file_path):
    data = {}
    with open(file_path, "r") as f:
        for line in f:
            line_json = json.loads(line)
            data[line_json['filename']] = line_json['response']
    return data

def main(args):
    gt_classes = ['astronaut', 'backpacker', 'ballplayer', 'bartender', 'basketball_player', 'boatman', 'carpenter', 'cheerleader', 'climber', 'computer_user', 'craftsman',
                'dancer', 'disk_jockey', 'doctor', 'drummer', 'electrician', 'farmer', 'fireman', 'flutist', 'gardener', 'guard', 'guitarist', 'gymnast', 'hairdresser',
                'horseman', 'judge', 'laborer', 'lawman', 'lifeguard', 'machinist', 'motorcyclist', 'nurse', 'painter', 'patient', 'prayer', 'referee', 'repairman', 'reporter',
                'retailer', 'runner', 'sculptor', 'seller', 'singer', 'skateboarder', 'soccer_player', 'soldier', 'speaker', 'student', 'teacher', 'tennis_player', 'trumpeter',
                'waiter']
    # load data
    data = load_jsonl(args.prediction)
    gt_df = pd.read_csv('/ssd/public_datasets/FACET/annotations/annotations.csv')
    sub_gt_dic = gt_df[['person_id', 'filename', 'class1']].to_dict('records')
    # Load the model
    device = "cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.model, device)
    # calualate the cosine similarity
    with open(args.output, 'w') as f:
        with torch.no_grad():
            gt_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in gt_classes]).to(device)
            gt_features = model.encode_text(gt_inputs)
            gt_features /= gt_features.norm(dim=-1, keepdim=True)
            for item in tqdm(sub_gt_dic):
                filename = item['filename']
                if filename in data:
                    if args.prompt_id == 2:
                        cut_str = ' '.join(data[filename][0].split(' ')[:50])
                        predict_input = torch.cat([clip.tokenize("a photo of a {}".format(cut_str))]).to(device)
                    else:
                        predict_input = torch.cat([clip.tokenize(data[filename].strip(), context_length=1024)]).to(device)
                        predict_input = predict_input[:,0:77]
                        predict_input[0,76] = 49407
                    predict_features = model.encode_text(predict_input)
                    predict_features = predict_features.repeat(len(gt_classes), 1)
                    predict_features /= predict_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * predict_features @ gt_features.T).softmax(dim=-1)
                    predict_class = gt_classes[similarity[0].argmax().item()]
                    item['predict_class'] = predict_class
                    f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default='ViT-B/32', help="model name")
    args.add_argument("--device_id", type=str, default='2', help="device id")
    args.add_argument("--prompt_id", type=int, default=1, help="prompt id")
    args.add_argument("--prediction", type=str, default='outputs/llava_7b_facet_response_1.jsonl', help="prediction jsonl file")
    args.add_argument("--output", type=str, default='outputs/llava_7b_facet_response_1_formatted.jsonl', help="output jsonl file")
    args = args.parse_args()
    main(args)