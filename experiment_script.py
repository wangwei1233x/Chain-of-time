import cv2
import numpy as np
from pathlib import Path
from natsort import natsorted
import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import pandas as pd
import json
import cv2
import os
import copy
import itertools
from google.genai.types import GenerateContentConfig, HttpOptions
import xml.etree.ElementTree as ET
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging
import base64
import argparse
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
import io

os.environ['GEMINI_API_KEY'] = 'xxx'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',default="gemini-3-pro-preview")
parser.add_argument('-e', '--event',default="complex_system")
args = vars(parser.parse_args())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scene_content1 = "an object exploding after being hit by a rubber bullet"
scene_content2 = "a glass mug being filled with water, at a constant rate"
scene_content3 = "a bouncy ball falling towards the ground"
scene_content4 = "a bouncy ball bouncing upward after hitting the ground"
scene_content5 = "a ball rolling inside a track"
scene_content6 = "a bob swinging back-and-forth in simple pendulum motion"
scene_content7 = "a running complex pulley system"
scene_content8 = "a object rolling down a inclined ramp"
scene_content9 = "a object moving towards and then colliding with another object that is lighter (in absolute mass)."


first_prompt = """\
Consider the following sequence of 5 images, which show {scene_content}. Each image frame video is precisely 0.5 seconds after the last frame.

Please generate an image that continues this sequence, simulating what the scene will look like {number of seconds forward} seconds further into the future after the last frame. """


second_prompt = """\
Continue this simulation {number of seconds forward} seconds into the future after the last frame that you generated. """

if args["event"] == "complex_system":

    simulation_param = "xxxexperiment_gemini/complex_system_simulation_parameter.json"
    scene = scene_content5
    aspect_ratio = {"0" : "9:16", "1" : "9:16", "2": "9:16", "3": "9:16", "4": "9:16", "5": "16:9","6": "16:9"}
    resolution = "1K"

if args["event"] == "rope_pulley":

    simulation_param = "xxxexperiment_gemini/rope_pulley_simulation_parameter.json"
    scene = scene_content7
    aspect_ratio = {"0":"16:9","1":"16:9","2":"16:9","3":"16:9","4":"16:9","5":"16:9",
                    "6":"16:9","7":"16:9","8":"16:9","9":"16:9","10":"16:9","11":"16:9","12":"16:9"}
    resolution = "1K"

if args["event"] == "inclined":

    simulation_param = "xxxexperiment_gemini/incline_simulation_parameter.json"
    scene = scene_content8
    aspect_ratio = {"0":"1:1","1":"1:1","2":"1:1","3":"1:1","4":"1:1","5":"1:1","6":"1:1","7":"1:1","8":"1:1"}
    resolution = "1K"

if args["event"] == "pendulum":

    simulation_param = "xxxexperiment_gemini/pendulum_simulation_parameter.json"
    scene = scene_content6
    aspect_ratio = {"0": "16:9", "1": "16:9", "2": "4:5", "3": "16:9", "4": "2:3", "5": "2:3", "6": "16:9", "7": "2:3", "8": "4:5"}
    resolution = "1K"

if args["event"] == "collision":

    simulation_param = "xxxexperiment_gemini/collision_simulation_parameter.json"
    scene = scene_content9
    aspect_ratio = {"0":"1:1","1":"1:1","2":"1:1","3":"1:1","4":"1:1","5":"1:1","6":"1:1","7":"1:1","8":"1:1","9":"1:1"}
    resolution = "1K"

with open(simulation_param, 'r') as f:
    parameter = json.load(f)

def load_media(image):
    media = []
    if not len(image) == 0:
        for i in image:
            image_bytes = None
            with open(i, 'rb') as f:
                image_bytes = f.read()
            media.append(types.Part.from_bytes(
                    data=copy.deepcopy(image_bytes),
                    mime_type='image/png'
                    ))
    return media

@retry(
    # Retry ONLY if we get a 429 (Too Many Requests) or 5xx (Server Error)
    # The specific exception class depends on the library version, but usually
    # API errors are wrapped in genai.errors or google.api_core.exceptions
    retry=retry_if_exception_type(Exception), 
    
    # Wait 2^x * 1 seconds between each retry starting with 4s, then 8s, up to 60s
    # This is "Exponential Backoff"
    wait=wait_exponential(multiplier=1, min=4, max=600),
    
    # Stop after 10 attempts (to prevent infinite loops)
    stop=stop_after_attempt(10),
    
    # Log a message before waiting so you know it's retrying
    before_sleep=before_sleep_log(logger, logging.INFO)
)
def simulation_cot02(sample_num):
    current_sample = "sample_"+str(sample_num)
    new_param = []

    for i in parameter:
        if i["cot_type"] == "0.2":
            new_param.append(i)

    for i in new_param:

        client = genai.Client()

        initial_prompt = first_prompt.format(scene_content = scene, interval = "next")

        followup_prompt = second_prompt.format(interval = "next")

        file_name = str(i["speed"])+ "_" + str(i["video_num"])

        original_frames = i["original_frames"]

        openai_frame = []

        path_name = "xxxgenerated_ar/"+args["event"]+"/CoT_0.2s/"+current_sample+"/"+file_name

        print("Simulating: "+str(path_name))

        if os.path.exists(path_name):
            continue

        if not os.path.exists(path_name):
            os.makedirs(path_name)

        openai_frame = load_media(original_frames)

        chat = client.chats.create(
        model="gemini-3-pro-image-preview",
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"], # Explicitly ask for text and image back
        )
    )

        # Combine text prompt and image parts into one content payload
        initial_message_contents =  openai_frame + [initial_prompt]

        response = chat.send_message(initial_message_contents,
                                     config=types.GenerateContentConfig(
                                    image_config=types.ImageConfig(
                                        aspect_ratio=aspect_ratio[str(i["video_num"])],
                                        image_size=resolution
                                    ),
                                ))

        # 6. Handle the Response (Save Image)
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                # Decode and save the image
                img_data = part.inline_data.data
                img = Image.open(io.BytesIO(img_data))
                img.save(os.path.join(path_name,"generated_frame_1.png"))

        # 7. Multi-Turn: Continue the conversation
        # You can now ask for changes without re-uploading the images.

        response_2 = chat.send_message(followup_prompt,config=types.GenerateContentConfig(
                                    image_config=types.ImageConfig(
                                        aspect_ratio=aspect_ratio[str(i["video_num"])],
                                        image_size=resolution
                                    ),
                                ))

        for part in response_2.candidates[0].content.parts:
            if part.inline_data:
                img_data = part.inline_data.data
                img = Image.open(io.BytesIO(img_data))
                img.save(os.path.join(path_name,"generated_frame_2.png"))
        
        response_3 = chat.send_message(followup_prompt,config=types.GenerateContentConfig(
                                    image_config=types.ImageConfig(
                                        aspect_ratio=aspect_ratio[str(i["video_num"])],
                                        image_size=resolution
                                    ),
                                ))

        for part in response_3.candidates[0].content.parts:
            if part.inline_data:
                img_data = part.inline_data.data
                img = Image.open(io.BytesIO(img_data))
                img.save(os.path.join(path_name,"generated_frame_3.png"))

        response_4 = chat.send_message(followup_prompt,config=types.GenerateContentConfig(
                                    image_config=types.ImageConfig(
                                        aspect_ratio=aspect_ratio[str(i["video_num"])],
                                        image_size=resolution
                                    ),
                                ))

        for part in response_4.candidates[0].content.parts:
            if part.inline_data:
                img_data = part.inline_data.data
                img = Image.open(io.BytesIO(img_data))
                img.save(os.path.join(path_name,"generated_frame_4.png"))
            
    return current_sample

@retry(
    # Retry ONLY if we get a 429 (Too Many Requests) or 5xx (Server Error)
    # The specific exception class depends on the library version, but usually
    # API errors are wrapped in genai.errors or google.api_core.exceptions
    retry=retry_if_exception_type(Exception), 
    
    # Wait 2^x * 1 seconds between each retry starting with 4s, then 8s, up to 60s
    # This is "Exponential Backoff"
    wait=wait_exponential(multiplier=1, min=4, max=600),
    
    # Stop after 10 attempts (to prevent infinite loops)
    stop=stop_after_attempt(10),
    
    # Log a message before waiting so you know it's retrying
    before_sleep=before_sleep_log(logger, logging.INFO)
)
def simulation_cot04(sample_num):
    current_sample = "sample_"+str(sample_num)
    new_param = []

    for i in parameter:
        if i["cot_type"] == "0.4":
            new_param.append(i)

    for i in new_param:

        client = genai.Client()

        initial_prompt = first_prompt.format(scene_content = scene, interval = "7th")

        followup_prompt = second_prompt.format(interval = "9th")

        file_name = str(i["speed"])+ "_" + str(i["video_num"])

        original_frames = i["original_frames"]

        openai_frame = []

        path_name = "xxxgenerated_ar/"+args["event"]+"/CoT_0.4s/"+current_sample+"/"+file_name

        print("Simulating: "+str(path_name))

        if os.path.exists(path_name):
            continue

        if not os.path.exists(path_name):
            os.makedirs(path_name)

        openai_frame = load_media(original_frames)

        chat = client.chats.create(
        model="gemini-3-pro-image-preview",
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"], # Explicitly ask for text and image back
        )
    )

        # Combine text prompt and image parts into one content payload
        initial_message_contents =  openai_frame + [initial_prompt]

        response = chat.send_message(initial_message_contents,config=types.GenerateContentConfig(
                                    image_config=types.ImageConfig(
                                        aspect_ratio=aspect_ratio[str(i["video_num"])],
                                        image_size=resolution
                                    ),
                                ))

        # 6. Handle the Response (Save Image)
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                # Decode and save the image
                img_data = part.inline_data.data
                img = Image.open(io.BytesIO(img_data))
                img.save(os.path.join(path_name,"generated_frame_1.png"))

        # 7. Multi-Turn: Continue the conversation
        # You can now ask for changes without re-uploading the images.

        response_2 = chat.send_message(followup_prompt,config=types.GenerateContentConfig(
                                    image_config=types.ImageConfig(
                                        aspect_ratio=aspect_ratio[str(i["video_num"])],
                                        image_size=resolution
                                    ),
                                ))

        for part in response_2.candidates[0].content.parts:
            if part.inline_data:
                img_data = part.inline_data.data
                img = Image.open(io.BytesIO(img_data))
                img.save(os.path.join(path_name,"generated_frame_2.png"))
        
    return current_sample


def simulation_cot08(sample_num):

    current_sample = "sample_"+str(sample_num)
    new_param = []

    for i in parameter:
        if i["cot_type"] == "0.8":
            new_param.append(i)

    for i in new_param:

        client = genai.Client()

        initial_prompt = first_prompt.format(scene_content = scene, interval = "9th")

        file_name = str(i["speed"])+ "_" + str(i["video_num"])

        original_frames = i["original_frames"]

        openai_frame = []

        path_name = "xxxgenerated_ar/"+args["event"]+"/CoT_0.8s/"+current_sample+"/"+file_name

        print("Simulating: "+str(path_name))

        if os.path.exists(path_name):
            continue

        if not os.path.exists(path_name):
            os.makedirs(path_name)

        openai_frame = load_media(original_frames)

        chat = client.chats.create(
        model="gemini-3-pro-image-preview",
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"], # Explicitly ask for text and image back
        )
    )

        # Combine text prompt and image parts into one content payload
        initial_message_contents =  openai_frame + [initial_prompt]

        response = chat.send_message(initial_message_contents,config=types.GenerateContentConfig(
                                    image_config=types.ImageConfig(
                                        aspect_ratio=aspect_ratio[str(i["video_num"])],
                                        image_size=resolution
                                    ),
                                ))

        # 6. Handle the Response (Save Image)
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                # Decode and save the image
                img_data = part.inline_data.data
                img = Image.open(io.BytesIO(img_data))
                img.save(os.path.join(path_name,"generated_frame_1.png"))
        
    return current_sample

if __name__ == "__main__":

    sample_num = [i for i in range(0,10,1)]

    with ThreadPoolExecutor(max_workers=cpu_count()) as ex:
        result = ex.map(simulation_cot02, sample_num)
    with ThreadPoolExecutor(max_workers=cpu_count()) as ex:
        result = ex.map(simulation_cot04, sample_num)
    with ThreadPoolExecutor(max_workers=cpu_count()) as ex:
        result = ex.map(simulation_cot08, sample_num)  # preserves order

    
    print("completed")
