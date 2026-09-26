# Switch to SAM3 and environment for object recognition
# This is an example of SAM3 usage
import os
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import argparse
from natsort import natsorted
import json
import pandas as pd

os.environ["HF_HOME"]="xxx"
os.environ["HF_TOKEN"] = "xxx"

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',default="gemini-3-pro-preview")
parser.add_argument('-e', '--event',default="complex_system")
args = vars(parser.parse_args())
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)

import os
import copy

def get_centroid(bbox):
    cx = (int(bbox[0]) + int(bbox[2])) / 2
    cy = (int(bbox[1]) + int(bbox[3])) / 2
    return (cx, cy)

if __name__ == "__main__":

    if args["event"] == "complex_system":

        simulation_param = "XXXcomplex_system_simulation_parameter.json"
        save_file = "XXX/experiment_data_gemini/complex_system_data.csv"
        base_path = "XXX/generated_ar/complex_downsized"
        ball_type = {"0": "A Orange Ball", "1": "A Silver Ball", "2": "A Silver Ball", "3": "A Red Ball", "4": "A Silver Ball", 
             "5": "A Silver Ball", "6": "A Silver Ball"}

    if args["event"] == "rope_pulley":

        simulation_param = "XXXrope_pulley_simulation_parameter.json"
        save_file = "XXX/experiment_data_gemini/rope_pulley_data.csv"
        base_path = "XXX/generated_ar/rope_downsized"
        ball_type = {"0": "A White Cube", "1": "A Yellow Cube", "2": "A Yellow Ball", "3": "A Orange Cube", "4": "A Purple Ball", 
             "5": "A Blue Ball", "6": "A Purple Ball", "7": "A Cyan Ball", "8": "A White Cube", "9": "A White Ball", "10": "A Grey Ball",
             "11": "A Flesh-Color Ball", "12": "A Red Ball"}

    if args["event"] == "inclined":

        simulation_param = "XXXincline_simulation_parameter.json"
        save_file = "XXX/experiment_data_gemini/inclined_data.csv"
        base_path = "XXX/generated_ar/inclined_downsized"
        ball_type = {"0": "A Brown Ball", "1": "A Brown Object", "2": "A Brown Ball", "3": "A Dark Purple Ball", 
                "4": "A Orange Object", "5": "A Dark Purple Ball", "6": "A Green Cube", "7": "A Dark Green Cube", 
                "8": "A Dark Green Object"}

    if args["event"] == "pendulum":

        simulation_param = "XXXpendulum_simulation_parameter.json"
        save_file = "XXX/experiment_data_gemini/pendulum_data.csv"
        base_path = "XXX/generated_ar/pendulum_downsized"
        ball_type = {"0": "A Sliver Ball", "1": "A Silver Ball", "2": "A Silver Object", "3": "A Silver Ball", "4": "A Blue Ball", 
             "5": "A Brass Ball", "6": "A Silver Ball", "7": "A Brass Ball", "8": "A Silver Object"}

    if args["event"] == "collision":

        simulation_param = "XXXcollision_simulation_parameter.json"
        save_file = "XXX/experiment_data_gemini/collision_data.csv"
        base_path = "XXX/generated_ar/collision_downsized"
        ball_type = {"0": "A Black Object", "1": "A Green Object", "2": "A Green Ball", "3": "A Green Ball", 
             "4": "A Green Ball", "5": "A Green Cube", "6": "A Green Ball", "7": "A Green Cylinder", 
             "8": "A Green Object", "9": "A Green Ball"}
    
    with open(simulation_param, 'r') as f:
        parameter = json.load(f)

    
    cot_type = {"0.2": "CoT_0.2s", "0.4": "CoT_0.4s", "0.8": "CoT_0.8s"}
    sample_size = 10
    sample_index = [i for i in range(sample_size)]
    frame_order = {"0.2":[6,7,8,9], "0.4": [7,9], "0.8" : [9]}
    frame_index_dict = []

    for i in parameter:
        frame_path = os.path.join(base_path,cot_type[i["cot_type"]])
        all_frames = []
        gt_frames_location = []
        for index in sample_index:
            sample_path = os.path.join(frame_path,"sample_"+str(index))
            file_path = os.path.join(sample_path,str(i["speed"])+"_"+str(i["video_num"]))
            for file_num in range(len(natsorted(os.listdir(file_path)))):
                # Load an image
                info_holder = copy.deepcopy(i)
                file = natsorted(os.listdir(file_path))[file_num]
                image = Image.open(os.path.join(file_path,file))
                inference_state = processor.set_image(image)
                prompt = ball_type[str(i["video_num"])]
                # Prompt the model with text
                output = processor.set_text_prompt(state=inference_state, prompt=prompt)
                # Get the masks, bounding boxes, and scores
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                if boxes.tolist() == []:
                    info_holder["predicted"] = None
                    info_holder["predicted_x"] = None
                    info_holder["predicted_y"] = None
                else:
                    info_holder["predicted"] = (boxes.tolist()[0])
                    info_holder["predicted_x"] = get_centroid(boxes.tolist()[0])[0]
                    info_holder["predicted_y"] = get_centroid(boxes.tolist()[0])[1]
                #parse gt frames
                frames = i["gt_frames"][file_num]
                image = Image.open(frames)
                inference_state = processor.set_image(image)
                prompt = ball_type[str(i["video_num"])]
                # Prompt the model with text
                output = processor.set_text_prompt(state=inference_state, prompt=prompt)
                # Get the masks, bounding boxes, and scores
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                if boxes.tolist() == []:
                    info_holder["gt"] = None
                    info_holder["gt_x"] = None
                    info_holder["gt_y"] = None
                else:
                    info_holder["gt"] = boxes.tolist()[0]
                    info_holder["gt_x"] = get_centroid(boxes.tolist()[0])[0]
                    info_holder["gt_y"] = get_centroid(boxes.tolist()[0])[1]
                info_holder["frame_num"] = frame_order[i['cot_type']][file_num]
                info_holder["sample_num"] = index
                frame_index_dict.append(info_holder)

        
    aggregate_data = pd.DataFrame(frame_index_dict)
    aggregate_data.to_csv(save_file)