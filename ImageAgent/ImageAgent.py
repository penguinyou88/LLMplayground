# import packages
from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv('huggingface_token')

# Build Custom Tools Function


class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a simple caption describing the image."

    def _run(self, img_path):

        # # using huggingface model directly: download model to loca -- > 1.88 G --> won't work for deployment
        # image = Image.open(img_path).convert('RGB')

        # model_name = "Salesforce/blip-image-captioning-large"
        # device = "cpu"  # cuda

        # processor = BlipProcessor.from_pretrained(model_name)
        # model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        # inputs = processor(image, return_tensors='pt').to(device)
        # output = model.generate(**inputs, max_new_tokens=20)

        # caption = processor.decode(output[0], skip_special_tokens=True)

        API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        caption = query(img_path, API_URL, headers)

        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class ObjectDetectionTool(BaseTool):
    name = "Object detector"
    description = "Use this tool when given the path to an image that you would like to detect objects. " \
                  "It will return a list of all detected objects. Each element in the list in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score."

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))

        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

################## helper function #############################
# def get_image_caption(image_path):
#     """
#     Generates a short caption for the provided image.

#     Args:
#         image_path (str): The path to the image file.

#     Returns:
#         str: A string representing the caption for the image.
#     """
#     image = Image.open(image_path).convert('RGB')

#     model_name = "Salesforce/blip-image-captioning-large"
#     device = "cpu"  # cuda

#     processor = BlipProcessor.from_pretrained(model_name)
#     model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

#     inputs = processor(image, return_tensors='pt').to(device)
#     output = model.generate(**inputs, max_new_tokens=20)

#     caption = processor.decode(output[0], skip_special_tokens=True)

#     return caption


# def detect_objects(image_path):
#     """
#     Detects objects in the provided image.

#     Args:
#         image_path (str): The path to the image file.

#     Returns:
#         str: A string with all the detected objects. Each object as '[x1, x2, y1, y2, class_name, confindence_score]'.
#     """
#     image = Image.open(image_path).convert('RGB')

#     processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
#     model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

#     inputs = processor(images=image, return_tensors="pt")
#     outputs = model(**inputs)

#     # convert outputs (bounding boxes and class logits) to COCO API
#     # let's only keep detections with score > 0.9
#     target_sizes = torch.tensor([image.size[::-1]])
#     results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

#     detections = ""
#     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#         detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
#         detections += ' {}'.format(model.config.id2label[int(label)])
#         detections += ' {}\n'.format(float(score))

#     return detections


def query(filename, API_URL, headers):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

