import cv2
import os
import torch
import json
import random,string
import tensorflow as tf
import numpy as np
from log_settings import logger


class classification_inference():
  """
  Module to perform classification model inference
  """
  def __init__(self, input_img):
    self.input_img = input_img
    self.label_list = []
    self.input_details = None
    self.output_details = None
    
  def load_classification_model(self,classifier_model_path):
    """
    This function will loads the classification model from the saved path
    """
    try:
        interpreter = tf.lite.Interpreter(model_path=classifier_model_path)
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        return interpreter
    except Exception as ex:
      logger.error(str(ex))
      logger.error("Error in loading the classification model !")
  def load_classifier_labels(self,classifier_label_path):
    """
    This function will load the labels from the saved path
    """
    try:
        with open(classifier_label_path, encoding="UTF-8") as labels_file:
            self.label_list = [line.split()[1].strip() for line in labels_file.readlines()]
        return self.label_list
    except Exception as ex:
       logger.error(str(ex))
       logger.error("Error in loading the classification model labels !")

  def run_classifier_inference(self):
    """
    Main function to perform classification model inference
    """
    try:
        logger.info("Initiating classification model inference")
        interpreter = self.load_classification_model(os.path.join(os.getcwd(),"models","classifier.tflite"))
        labels_list = self.load_classifier_labels(os.path.join(os.getcwd(),"models","labels.txt"))
        new_img = cv2.resize(self.input_img, (224, 224))
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        processed_img = np.expand_dims(new_img, axis=0)
        logger.info("Data processed for classification model")
        interpreter.set_tensor(self.input_details[0]['index'], processed_img)
        interpreter.invoke()
        try:
            output_data = interpreter.get_tensor(self.output_details[0]['index'])
            logger.info("Results predicted for classification model returning results")
            output_list = output_data[0].tolist()
            predicted_result = str(labels_list[output_list.index(max(output_list))])
            logger.info(f"The classification model has predicted the input image as {predicted_result}")
            return predicted_result
        except:
           logger.error("Error in performing classification model prediction")
    except Exception as ex:
       logger.error(str(ex))
       logger.error("Error in performing infernce of the classification model")

commodity_config = {
    "chana":
        {
            "prob_threshold":0.5,
            "iou_threshold":0.5,
            "model_path":os.path.join(os.getcwd(),"models","desi.pt")
        },
    "maize":
        {
            "prob_threshold":0.1,
            "iou_threshold":0.1,
            "model_path":os.path.join(os.getcwd(),"models","yellow maize.pt")
        },
    "rice":
        {
            "prob_threshold":0.5,
            "iou_threshold":0.7,
            "model_path":os.path.join(os.getcwd(),"models","1121.pt")
        }
}
color_scheme = {"foreign material":(245, 33, 252),"organic foreign material":(245, 33, 252),"inorganic foreign material":(48, 76, 201),"dehusked":(245, 218, 64),"red grain":(237, 14, 25),"healthy":(50, 168, 82),"fungus":(74, 84, 81),"undetected":(20, 20, 20),"discolour":(230, 119, 23),"weeveled":(21, 206, 209),"other edible seeds":(143, 35, 219),"shriveled":(198, 204, 18),"healthy-1121":(50, 168, 82),"broken": (204, 18, 102),"blacktip":(117, 91, 25),"damage":(214, 68, 45),"damaged":(214, 68, 45),"damaged-discolor":(214, 68, 45),"damaged-discolour":(214, 68, 45),"damage-discolor":(214, 68, 45),"chalky":(58, 35, 74),"immature":(50, 102, 100),"fm":(74, 35, 40),"green":(31, 144, 196),"label":(255,255,255)}

try:
    detection_results_path = os.path.join(os.getcwd(),"Detections")
    if not os.path.exists(detection_results_path):
        os.mkdir(detection_results_path)
except:
   logger.info("Failed to create detections directory")

class detection_inference():
  def __init__(self, input_image, commodity_configuration, defined_color_scheme):
    logger.info(f"Initiating Detection model inference")
    self.input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=commodity_configuration["model_path"],
        )
    self.model.conf = float(commodity_configuration["prob_threshold"])
    self.model.iou = float(commodity_configuration["iou_threshold"])
    self.color_scheme = defined_color_scheme
    self.grain_counts = {}
  def perform_detection(self, input_image):
    prediction_results = self.model(input_image[..., ::-1], size=1280)
    json_string = prediction_results.pandas().xyxy[0].to_json(orient="records")
    logger.info("Inference completed for detection model sending results to draw annotations")
    results = json.loads(json_string)
    return results

  def get_grain_counts_per_category(self, grain_type):
    if grain_type in self.grain_counts.keys():
      self.grain_counts[grain_type]+=1
    else:
      self.grain_counts[grain_type]=1

  def draw_annotations_on_image(self, annot_data):
    logger.info("Writing annotations !....")
    for co_ords in annot_data:
      
      #For drawing bounding box
      cv2.rectangle(
          img=self.input_image, 
          pt1=(round(co_ords.get("xmin")),round(co_ords.get("ymin"))),
          pt2=(round(co_ords.get("xmax")),round(co_ords.get("ymax"))), 
          color=self.color_scheme[co_ords.get("name").lower()], 
          thickness=5)
      
      start_point = (round(co_ords.get("xmin")), round(co_ords.get("ymin"))-25)
      end_point = (round(co_ords.get("xmax")), round(co_ords.get("ymin")))
      cv2.rectangle(
          img=self.input_image,
          pt1=start_point,
          pt2=end_point,
          color=self.color_scheme[co_ords.get("name").lower()],
          thickness=-1
      )

      self.get_grain_counts_per_category(f"{co_ords.get('name')}")

      # Calculate text size and font scale
      (text_width, text_height), _ = cv2.getTextSize(f"{co_ords.get('name')}  {round(co_ords.get('confidence')*100)}%", cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)
      font_scale = min((end_point[0] - start_point[0]) / text_width, (end_point[1] - start_point[1]) / text_height)

      # Calculate text location
      text_x = int(start_point[0] + (end_point[0] - start_point[0] - text_width * font_scale) / 2)
      text_y = int(start_point[1] + (end_point[1] - start_point[1] + text_height * font_scale) / 2)

      #To write class name on the top of the annotation
      cv2.putText(
          img=self.input_image,
          text=f"{co_ords.get('name')}  {round(co_ords.get('confidence')*100)}%",
          org=(text_x, text_y), 
          fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
          fontScale=font_scale, 
          color=self.color_scheme["label"], 
          thickness=2
      )
    logger.info("Annotation drawn on the input image saving the annotated image !")
    file_path = f"{detection_results_path}{os.sep}{''.join(random.choices(string.ascii_letters, k=5))}_annotated.jpg"
    cv2.imwrite(file_path,self.input_image)
    return file_path
  def run_inference(self):
    infernece_data = self.perform_detection(self.input_image)
    annotated_image_path = self.draw_annotations_on_image(infernece_data)
    self.grain_counts["Total"] = sum(self.grain_counts.values())
    return annotated_image_path, self.grain_counts  