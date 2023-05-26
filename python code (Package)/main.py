from skimage import io
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from inference import classification_inference, detection_inference, color_scheme, commodity_config
from log_settings import logger

try:
    input_image = input("Enter a valid image path/ URL of an image : ")
    logger.info("Input Data Received Initiating Inference")
    input_image = io.imread(input_image)
    classifier_model_obj = classification_inference(input_img=input_image)
    classifier_model_results = classifier_model_obj.run_classifier_inference()
    detection_model_obj = detection_inference(input_image=input_image, commodity_configuration=commodity_config[classifier_model_results.lower()], defined_color_scheme=color_scheme)
    annotated_img_path, grain_counts = detection_model_obj.run_inference()
    print("\n")
    logger.info(f"annotated image saved in {annotated_img_path}")
    print(f"object counts :{grain_counts}")
except Exception as ex:
    logger.error(str(ex))
    logger.error("Error in performing inference check the input data !")
    