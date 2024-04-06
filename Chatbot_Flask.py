#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from threading import Thread, get_ident
import joblib
import torch
import os
import io
import csv
import random
import string
import time


# In[ ]:


from recommend import collaborative_main, content_main, popular_main
from response import intent_detection, response_text


# In[ ]:


import pandas as pd


# In[ ]:


def generate_profile_name(existing_names):
    # Generate a random length for the profile name (up to 24 characters)
    name_length = random.randint(5, min(8, len(string.ascii_letters)))
    # Generate a random profile name
    profile_name = ''.join(random.choices(string.ascii_letters, k=name_length))
    # Ensure the generated name is unique
    while profile_name in existing_names:
        profile_name = ''.join(random.choices(string.ascii_letters, k=name_length))
    return profile_name

def add_update_record(csv_file, anime_uid, score = '', profile = '', scores = '', link = ''):
    fieldnames = ['uid', 'profile', 'anime_uid', 'score', 'scores', 'link']
    # Create a temporary list to hold all records
    records = []
    max_uid = 0
    existing_profile_name = []
    is_new_record = True
    # Read existing records, update if necessary
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['profile'] == profile and row['anime_uid'] == anime_uid:
                if score != '': 
                    row['score'] = score
                if scores != '': 
                    row['scores'] = scores
                if link != '':
                    row['link'] = link
                is_new_record = False
            existing_profile_name.append(row['profile'])
            if int(row['uid']) > max_uid:
                max_uid = int(row['uid'])
            records.append(row)
    # If it's a new record, add it with a new uid
    if is_new_record:
        new_uid = max_uid + 1
        if profile == '':
            profile = generate_profile_name(existing_profile_name)
        new_record = {'uid': new_uid, 'profile': profile, 'anime_uid': anime_uid, 'score': score, 'scores': scores, 'link': link}
        records.append(new_record)
    # Write all records back to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


# In[ ]:


def Google_drive_data_load():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = 'Google_Drive_Credentials.json'
    global_data['PARENT_FOLDER_ID'] = '1BHLej78d1OZOMgeZ-XDHiU0rbVX9sLet'
    global_data['creds'] = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE,scopes=SCOPES)

def load_ChatModel(): # Load DialogPT model
    global_data['tokenizer'] = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side='left')
    global_data['model'] = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    print("Model Load Complete")

def load_model_from_memory(file_id, service, name):
    unique_name = f"{name}_{get_ident()}"
    #Get file from Google Drive
    request = service.files().get_media(fileId=file_id)
    file_stream = io.FileIO(unique_name, 'w+')
    with file_stream:
        #Download file to stream
        downloader = MediaIoBaseDownload(file_stream, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        try:
            global_data[f'{name}'] = joblib.load(file_stream)
            print(f'{name} loaded')
        except Exception as e:
            print(e, 'a')

def list_and_load_models():
    init_service = build('drive', 'v3', credentials=global_data.get('creds', None))

    # Update the query to search for files in the specified folder
    query = f"'{global_data.get('PARENT_FOLDER_ID', None)}' in parents and trashed=false and mimeType='application/octet-stream'"
    results = init_service.files().list(q=query).execute()

    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        for item in items:
            service = build('drive', 'v3', credentials=global_data.get('creds', None))
            thread = Thread(target=load_model_from_memory, args=(item['id'], service, item['name']))
            thread.start()


# In[ ]:


app = Flask(__name__)
@app.route("/")
def index():
    threadstoExecute = [Thread(target=load_ChatModel), Thread(target=list_and_load_models, daemon=True)]
    for thread in threadstoExecute: thread.start()
    return render_template('chat.html')


# In[ ]:


def initial_chatting(step, **kwargs):
    if step==0:
        chat_output = 'Hello, My name is Animebot Nice to Meet you. Provide me your user_id?'
        return chat_output, 1
    if step==1:
        user_id=kwargs.get('message', None)
        while 'user_data.joblib' not in global_data:
            time.sleep(1)
        if user_id in global_data['user_data.joblib']['profile'].values:
            chat_output = f"Hi {user_id}! What kind of anime are you interested in, or is there a specific type of recommendation you're looking for today?"
            return True, chat_output, 2, user_id
        else:
            user_id=generate_profile_name(global_data['user_data.joblib']['profile'].values)
            chat_output = f"Hi! I have created a new User Id for you, Your User Id is: '{user_id}'. What kind of anime are you interested in, or is there a specific type of recommendation you're looking for today?"
            return False, chat_output, 2, user_id


# In[ ]:


def memoizing_Chat():
    step=0
    is_user=False
    user_id = ''
    def memoized_Chat():
        msg = request.form["msg"]
        user_input = msg


        nonlocal step, is_user, user_id
        

        # Testing Code Begin
        is_recommend = False
        is_content = False
        
        if step==0:
            chat_output, step=initial_chatting(step, message=msg)
            return chat_output
        if step==1:
            is_user, chat_output, step, user_id = initial_chatting(step, message=msg)
            return chat_output
        
        else:

            for word in word_list:
                if word in user_input.lower():
                    is_recommend = True
            
            if is_recommend:  # Checking if the input is asking for a recommendation

                anime_features = intent_detection(user_input)

                if anime_features: is_content=True
                
                if is_content: # Checking if the input has a feature list
                    
                    # Waiting for the relevant files to be loaded
                    while 'knn_model_Content.joblib' not in global_data:
                        time.sleep(1)
                    while 'Combined_Embedding.joblib' not in global_data:
                        time.sleep(1)
                    while 'anime_data.joblib' not in global_data:
                        time.sleep(1)
                    
                    lom, found = content_main(anime_features, 
                                  global_data['knn_model_Content.joblib'],
                                  global_data['Combined_Embedding.joblib'],
                                  global_data['anime_data.joblib'])
                
                elif is_user: # Checking if it is an exising user
                    
                    # Waiting for the relevant files to be loaded
                    while 'knn_model_Collaborative.joblib' not in global_data:
                        time.sleep(1)
                    while 'profile_to_index.joblib' not in global_data:
                        time.sleep(1)
                    while 'index_to_profile.joblib' not in global_data:
                        time.sleep(1)
                    while 'user_item_matrix.joblib' not in global_data:
                        time.sleep(1)
                    while 'user_data.joblib' not in global_data:
                        time.sleep(1)
                    while 'rating_data.joblib' not in global_data:
                        time.sleep(1)
                    while 'anime_data.joblib' not in global_data:
                        time.sleep(1)
                        
                    lom, found = collaborative_main(user_id, global_data['knn_model_Collaborative.joblib'], 
                                global_data['profile_to_index.joblib'], 
                                global_data['index_to_profile.joblib'], 
                                global_data['user_item_matrix.joblib'], 
                                global_data['user_data.joblib'], 
                                global_data['rating_data.joblib'], 
                                global_data['anime_data.joblib'], 5)
                
                else: # Recommending the most popular movies if it is a new user without any specific request
                    
                    while 'anime_data.joblib' not in global_data:
                        time.sleep(1)
                    
                    lom, found = popular_main(global_data['anime_data.joblib'])
                    
                chat_output = response_text(user_input, lom, found)
                
                return chat_output
            
            else:
                
                return get_Chat_response(user_input, global_data.get('model', None), global_data.get('tokenizer', None))
            
    return memoized_Chat


# In[ ]:


@app.route("/get", methods=["GET", "POST"])
def call_Chat():
    return Chat()


# In[ ]:


def get_Chat_response(user_input, model, tokenizer):
    text = user_input
    encoded_user_input = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
    encoded_bot_output = model.generate(encoded_user_input, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(encoded_bot_output[:, encoded_user_input.shape[-1]:][0], skip_special_tokens=True)


# In[ ]:


anime_data = joblib.load('../Joblib/anime_data.joblib')
anime_data


# In[ ]:


# if __name__ == '__main__':
global_data = {}
Chat = memoizing_Chat()
Google_drive_data_load()
app.run()

