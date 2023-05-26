from skimage import io
from PyQt5.QtWidgets import QApplication, QMessageBox
from inference import classification_inference, detection_inference, color_scheme, commodity_config
from log_settings import logger

try:
    input_image = input("Enter a valid image path/ URL of an image : ")
    logger.info("Input Data Received Initiating Inference")
    input_image = io.imread(input_image)
    classifier_model_obj = classification_inference(input_img=input_image)
    classifier_model_results = classifier_model_obj.run_classifier_inference()
    try:
        classifier_pop_up_app = QApplication([])
        # Create a message box dialog
        classifier_pop_up = QMessageBox()
        classifier_pop_up.setIcon(QMessageBox.Information)
        classifier_pop_up.setText(f"The given input image is classified as : {classifier_model_results}")
        classifier_pop_up.setWindowTitle("Classification model Prediction")
        classifier_pop_up.setStandardButtons(QMessageBox.Ok)
        ok_button = classifier_pop_up.button(QMessageBox.Ok)
        ok_button.clicked.connect(classifier_pop_up.close)
        # Show the message box
        classifier_pop_up.show()
        # Run the application event loop
        classifier_pop_up_app.exec_()
    except:
        logger.error("Error in showing the classifier model results")
    detection_model_obj = detection_inference(input_image=input_image, commodity_configuration=commodity_config[classifier_model_results.lower()], defined_color_scheme=color_scheme)
    annotated_img_path, grain_counts = detection_model_obj.run_inference()
    try:
        detection_popup_app = QApplication([])
        detection_popup = QMessageBox()
        detection_popup.setIcon(QMessageBox.Information)
        detection_popup.setText(f"Detection results are saved in : {annotated_img_path} \n \n Grain counts : {grain_counts}")
        detection_popup.setWindowTitle("Detection model Prediction")
        detection_popup.setStandardButtons(QMessageBox.Ok)
        ok_button = detection_popup.button(QMessageBox.Ok)
        ok_button.clicked.connect(detection_popup.close)
        # Show the message box
        detection_popup.show()
        detection_popup_app.exec_()
    except:
        logger.info("Error in showing detection results")
except Exception as ex:
    logger.error(str(ex))
    logger.error("Error in performing inference check the input data !")