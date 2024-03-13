import pandas as pd
import csv
import torch
from PIL import Image
from collections import OrderedDict
from transformers import CLIPProcessor, CLIPModel, AlignModel, AlignProcessor, BlipForImageTextRetrieval, BlipProcessor
import numpy as np
import faiss
import os
import urllib.request
from io import BytesIO
import json
from os import chdir as cd
import openai
from sklearn.metrics import precision_score, recall_score
import random
import matplotlib.pyplot as plt
import matplotlib.image as img
import time
from torch.nn.functional import normalize

api_key = "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" # SSU
openai.api_key = api_key
g_model_name = "gpt-3.5-turbo"


class Get:
    def __init__(self, model_id, dataset_name, device='cpu'):
        self.model_id = model_id
        self.dataset_name = dataset_name

        if self.model_id == "openai/clip-vit-base-patch32":
            model = CLIPModel.from_pretrained(self.model_id).to(device)
            processor = CLIPProcessor.from_pretrained(self.model_id)
            index = faiss.read_index(f"./data/clip-vit-base-patch32-{self.dataset_name}.index")
            data_index = pd.read_csv(f'./data/{self.dataset_name}_index.csv')
    
        elif self.model_id == "kakaobrain/align-base":
            model = AlignModel.from_pretrained(self.model_id).to(device)
            processor = AlignProcessor.from_pretrained(self.model_id)
            index = faiss.read_index(f"./data/align-base-{self.dataset_name}.index")
            data_index = pd.read_csv(f'./data/{self.dataset_name}_index.csv')

        elif self.model_id == "Salesforce/blip-itm-base-coco":
            model = BlipForImageTextRetrieval.from_pretrained(self.model_id).to(device)
            processor = BlipProcessor.from_pretrained(self.model_id)
            if dataset_name == 'coco':
                index = faiss.read_index(f"./data/blip-itm-base-coco-{self.dataset_name}.index")
                data_index = pd.read_csv(f'./data/{self.dataset_name}_index_blip.csv')
            
            else:
                index = faiss.read_index(f"./data/blip-itm-base-coco-{self.dataset_name}.index")
                data_index = pd.read_csv(f'./data/{self.dataset_name}_index.csv')

        gt_df = pd.read_csv(f'./data/gt/{dataset_name}.csv')

        self.model = model
        self.processor = processor
        self.index = index
        self.data_index = data_index
        self.gt_df = gt_df
    
        self.get_dataset()

    def get_dataset(self):
        if self.dataset_name == 'coco':
            coco_full = []
            for i in range(1, 12):
                with open(f'./data/coco_train_captions/coco_train_{i}.json', 'r') as f:
                    coco = json.load(f)
                    coco_full.extend(coco['annotations'])
            coco_full = {'annotations': coco_full}
            dataset = coco_full.get('annotations', [])
        
        elif self.dataset_name == 'flickr':
            dataset = pd.read_csv('./data/flickr_captions.txt', sep=',')
            
        self.dataset = dataset
        
    def get_text_embedding(self, text):
        with torch.no_grad():
            if self.model_id == "Salesforce/blip-itm-base-coco":
                text_encoder = self.model.text_encoder
                inputs = self.processor(text=text, padding=True, truncation=True, return_tensors="pt")
                embedding = text_encoder(input_ids=inputs.input_ids, return_dict=True)[0]
                text_features = normalize(self.model.text_proj(embedding[:, 0, :]), dim=-1)
                embedding_as_np = text_features.cpu().numpy()
                
            else:
                inputs = self.processor(text=text, padding=True, truncation=True, return_tensors="pt")
                embedding = self.model.get_text_features(**inputs)
                embedding_as_np = embedding.cpu().numpy()
        
        return embedding_as_np


class Exp(Get):
    def __init__(self, model_id, dataset_name, pos, k=1):
        self.model_id = model_id
        self.dataset_name = dataset_name
        super().__init__(model_id, dataset_name)
        self.pos = pos
        self.k = k

        if self.pos == 'adj':   
            input_ls = ['nervous', 'painful', 'paranoid', 'faithful', 'furious', 'refreshing', 'exhausting', 'cooperative', 'holy', 'fantastic']
        elif self.pos == 'adv':
            input_ls = ['competitively', 'technologically', 'joyously', 'fearfully', 'extravagantly', 'offensively', 'silently', 'historically', 'speedily', 'haphazardly']
        elif self.pos == 'verb':
            input_ls = ['cut', 'drink', 'stand', 'knock', 'hit', 'drive', 'paddle', 'perform', 'decorate', 'jump']
        elif self.pos == 'noun':
            input_ls = ['girl', 'giraffe', 'oven', 'bed', 'dogs', 'room', 'orange', 'pizza', 'bike', 'monitor']
        elif self.pos == 'p':
            input_ls = ['cloudy sky', 'refreshing carrot', 'snowy mountain', 'cooperative player', 'fantastic view', 'joyously walk', 'competitively game', 'speedily delivery', 'historically build', 'faithful dog' ]
        
        self.input_ls = input_ls

    def first_turn(self, input_text):
        try: 
            prompt = f'''
- Your role is to change the given word into a sentence in a specific situation consisting of specific nouns.
This sentence will be used to find the most relevant image through the Vision-Language Model(VLM).
You will change the word into a sentence through the process below.
    1. Check {input_text}.
    2. Check the dictionary definition of {input_text} in the Oxford dictionary.
    3. Remember at least three nouns that can express dictionary definitions.
    4. Select some of the nouns you remember and use them to describe a specific situation in one sentence.
    5. Modify the sentence in the form of an image description that preserves the meaning of {input_text} and is easy for the VLM to find.
    6. Output only the final sentence.
Output just final sentence.

- I'll show you this process through an example.
- input_text : competitve
    1. input text : competitve
    2. The dictionary definition of 'competitive' is "Of, pertaining to, or characterized by competition; organized on the basis of competition."
    3. Some nouns that can illustrate this definition are ['sports', 'exam', 'player', 'baseball', 'study', 'soccer', 'boxing', 'game'].
    4. Among them, you can select ['sports', 'soccer', 'game'] to create the sentence "One sportman is playing in the soccer game."
    5. Revise the sentence to "One man wearing a uniform is kicking the ball with other players." to make it easier for VLM to find the image in the text and to preserve the meaning of 'competitive'.
    6. "One man wearing a uniform is kicking the ball with other players."
- situation : "One man wearing a uniform is kicking the ball with other players."

- Let's start.
- input text : {input_text}
- situation :
'''
            
            response = openai.ChatCompletion.create(
                model=g_model_name,
                temperature = 0.5,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # extract situations
            hypo = response['choices'][0]['message']['content']
            embedded_text = self.get_text_embedding(hypo)
            
            if self.model_id == 'kakaobrain/align-base':
                faiss.normalize_L2(embedded_text.reshape((1,640)))
            elif self.model_id == 'Salesforce/blip-itm-base-coco':
                faiss.normalize_L2(embedded_text.reshape((1,256)))
            else:
                faiss.normalize_L2(embedded_text.reshape((1,512)))

            distances, indices = self.index.search(embedded_text, self.k)
            return indices[0].tolist(), hypo, distances[0, 0]
        
        except Exception as e:
            print('firstturn', e)
            return [999999999], 0, input_text
        
    def multi_turn(self, input_text, result_list):
        try:
            prompt = f'''
- Your job is to check whether the caption is a semantically similar situation that can be replaced by the input text when expressed as a single word.
The caption is the image_caption data of the (image, image_caption) pair.
The sentence data that will be used instead of looking at the image to ensure that the most relevant image for the input text is retrieved.
Answer can be either yes or no.

- Here's an example
- input text : competitive
- caption : "A pitcher, batter and catcher in a baseball game."
- answer : yes

- input text : adventurous
- caption : "A computer mouse is in front of two keyboards"
- answer : no

- Let's start
- input text : {input_text}
- caption : {result_list[0][1]}
- answer : 
'''
            
            response = openai.ChatCompletion.create(
                model=g_model_name,
                temperature = 0.5,
                messages=[{"role": "user", "content": prompt}],
                logprobs=True,
                top_logprobs=4
            )
        
            generated_text = response['choices'][0]['message']['content']
            logprob = response['choices'][0]['logprobs']['content'][0]['logprob']
            prob = np.round(np.exp(logprob)*100, 2)

            if ('yes' in generated_text or 'Yes' in generated_text) and (prob >= 30):
                return input_text, result_list[0][0]
            else: 
                return "retry"
                
        except Exception as e:
            print('multiturn', e)
            return input_text, [999999999]

    def get_captions(self, index_list):
        grouped_dict = {}
        index_list = [self.data_index.iloc[i].image_index for i in index_list]
        if self.dataset_name == 'coco':
            result = [(item['image_id'], item['caption']) for item in self.dataset if item['image_id'] in index_list]
        else:
            result = [(item['image_id'], item['caption']) for i, item in self.dataset.iterrows() if item['image_id'] in index_list]

        for key, value in result:
            if key in grouped_dict:
                grouped_dict[key].append(value)
            else:
                grouped_dict[key] = [value]
        result_list = [(key, values) for key, values in grouped_dict.items()]

        return result_list
    
    def re_first_turn(self, input_text, hypo, num):
        try: 
            num += 1
            prompt = f'''
- Your role is to change the given word into a sentence in a specific situation consisting of specific nouns.
This sentence will be used to find the most relevant image through the Vision-Language Model(VLM).
You will change the word into a sentence through the process below. Output just final sentence.
    1. Check {input_text}.
    2. Check the dictionary definition of {input_text} in the Oxford dictionary.
    3. Remember at least three nouns that can express dictionary definitions.
    4. Select some of the nouns you remember and use them to describe a specific situation in one sentence.
    5. If the context of the sentence is {hypo}, change it to another context.
    6. Modify the sentence in the form of an image description that preserves the meaning of {input_text} and is easy for the VLM to find.
    7. Output only the final sentence.

- I'll show you this process through an example.
- input_text : competitve
- exclude : ["One soccer player is kicking the ball in the field.", "A person is running while listening to music."]
    1. input text : competitve
    2. The dictionary definition of 'competitive' is "Of, pertaining to, or characterized by competition; organized on the basis of competition."
    3. Some nouns that can illustrate this definition are ['sports', 'exam', 'player', 'baseball', 'study', 'soccer', 'boxing', 'game'].
    4. Among them, you can select ['sports', 'soccer', 'game'] to create the sentence "One sportman is playing in the soccer game."
    5. Since "One sportman is playing in the soccer game." is similar to the situation with "One soccer player is kicking the ball in the field." in the exclude list, 
        Select the nouns ['exam', 'study'] and create the sentence "Many students are taking exams."
    6. Revise the sentence to "Many students are writing the answer on the paper in the classroom." to make it easier for VLM to find the image in the text and to preserve the meaning of 'competitive'.
    7. "Many students are writing the answer on the paper in the classroom."
- situation : "Many students are writing the answer on the paper in the classroom."

- Let's start.
- input text : {input_text}
- exclude : [{hypo}]
- situation :
'''

            response = openai.ChatCompletion.create(
                model=g_model_name,
                temperature = 0.5,
                messages=[{"role": "user", "content": prompt}]
            )
        
            re_hypo = response['choices'][0]['message']['content']
            embedded_text = self.get_text_embedding(re_hypo)

            if self.model_id == 'kakaobrain/align-base':
                faiss.normalize_L2(embedded_text.reshape((1,640)))
            elif self.model_id == 'Salesforce/blip-itm-base-coco':
                faiss.normalize_L2(embedded_text.reshape((1,256)))
            else:
                faiss.normalize_L2(embedded_text.reshape((1,512)))

            distances, indices = self.index.search(embedded_text, self.k)
            
            return indices[0].tolist(), re_hypo, distances[0, 0], num 
        
        except Exception as e:
            print(e)
            return [999999999], input_text, num