import flask
from flask import render_template
import os
import shutil

import src.utilities.pickling as pickling
from src.cropping.crop_mammogram import crop_mammogram
import src.utilities.data_handling as data_handling
from src.optimal_centers.get_optimal_centers import get_optimal_centers
from src.heatmaps.run_producer import produce_heatmaps, load_model
from src.modeling.run_model import load_run_save

from src.constants import VIEWS, VIEWANGLES, LABELS, MODELMODES

import random
import pandas as pd


DATA_FOLDER='uploads'
CROPPED_IMAGE_PATH='rest_api_output/cropped_images'
INITIAL_EXAM_LIST_PATH='uploads/exam_list_before_cropping.pkl'
CROPPED_EXAM_LIST_PATH='rest_api_output/cropped_images/cropped_exam_list.pkl'
EXAM_LIST_PATH='rest_api_output/data.pkl'
NUM_PROCESSES=10

HEATMAP_BATCH_SIZE=100
PATCH_MODEL_PATH='models/sample_patch_model.p'
HEATMAPS_PATH='rest_api_output/heatmaps'
IMAGE_PREDICTIONS_PATH = 'rest_api_output/image_predictions.csv'
IMAGE_MODEL_PATH='models/sample_image_model.p'
IMAGEHEATMAPS_MODEL_PATH='models/sample_imageheatmaps_model.p'
IMAGEHEATMAPS_PREDICTIONS_PATH='rest_api_output/imageheatmaps_predictions.csv'

DEVICE_TYPE='cpu'
GPU_NUMBER=0
NUM_EPOCHS=10



app = flask.Flask(__name__)

def clear_uploads(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)

def get_predictions():
    
    img_views = ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']
    clear_uploads(DATA_FOLDER)
    os.mkdir(DATA_FOLDER)
    
    exam_list = []
    exam_item = {}
    exam_item['horizontal_flip'] = 'NO'

    for view in img_views:

        if view in flask.request.files and flask.request.files[view].filename !='':
            view_file = flask.request.files[view]
            filename = view_file.filename
            view_file.save(os.path.join(DATA_FOLDER, view_file.filename))
            exam_item[view] = [os.path.splitext(filename)[0]]
        else:
            # delet the uploaded files
            clear_uploads(DATA_FOLDER)
            return " Error: Missing " + view + ". Four standard views are required"
    exam_list.append(exam_item)    
    print("Upload successfull")
    print(exam_list)
    pickling.pickle_to_file('uploads/exam_list_before_cropping.pkl', exam_list)
    
    clear_uploads(CROPPED_IMAGE_PATH)
    crop_mammogram(
        input_data_folder=DATA_FOLDER, 
        exam_list_path=INITIAL_EXAM_LIST_PATH, 
        cropped_exam_list_path=CROPPED_EXAM_LIST_PATH, 
        output_data_folder=CROPPED_IMAGE_PATH, 
        num_processes=NUM_PROCESSES,
        num_iterations=100,
        buffer_size=50,
    )
    print("Stage 2: Extract Centers")
    exam_list = pickling.unpickle_from_file(CROPPED_EXAM_LIST_PATH)
    data_list = data_handling.unpack_exam_into_images(exam_list, cropped=True)
    optimal_centers = get_optimal_centers(
        data_list=data_list,
        data_prefix=CROPPED_IMAGE_PATH,
        num_processes=NUM_PROCESSES
    )
    data_handling.add_metadata(exam_list, "best_center", optimal_centers)
    os.makedirs(os.path.dirname(EXAM_LIST_PATH), exist_ok=True)
    pickling.pickle_to_file(EXAM_LIST_PATH, exam_list)
    
    print("Stage 3: Generate Heatmaps")
    
    parameters = dict(
        device_type=DEVICE_TYPE,
        gpu_number=GPU_NUMBER,

        patch_size=256,

        stride_fixed=70,
        more_patches=5,
        minibatch_size=HEATMAP_BATCH_SIZE,
        seed=0,

        initial_parameters=PATCH_MODEL_PATH,
        input_channels=3,
        number_of_classes=4,

        data_file=EXAM_LIST_PATH,
        original_image_path=CROPPED_IMAGE_PATH,
        save_heatmap_path=[os.path.join(HEATMAPS_PATH, 'heatmap_malignant'),
                           os.path.join(HEATMAPS_PATH, 'heatmap_benign')],

        heatmap_type=[0, 1],  # 0: malignant 1: benign 0: nothing

        use_hdf5=False
    )
    random.seed(parameters['seed'])
    model, device = load_model(parameters)
    produce_heatmaps(model, device, parameters)

    print("Stage 4a: Run Classifier (Image)")
    
    parameters = {
            "device_type": DEVICE_TYPE,
            "gpu_number": GPU_NUMBER,
            "max_crop_noise": (100, 100),
            "max_crop_size_noise": 100,
            "image_path": CROPPED_IMAGE_PATH,
            "batch_size": 1,
            "seed": 0,
            "augmentation": True,
            "num_epochs": NUM_EPOCHS,
            "use_heatmaps": False,
            "heatmaps_path": None,
            "use_hdf5": False,
            "model_mode": MODELMODES.VIEW_SPLIT,
            "model_path": IMAGE_MODEL_PATH,
        }
    load_run_save(
            data_path=EXAM_LIST_PATH,
            output_path=IMAGE_PREDICTIONS_PATH,
            parameters=parameters,
        )
    
    
    print("Stage 4b: Run Classifier (Image + Heatmaps)")
    
    parameters = {
        "device_type": DEVICE_TYPE,
        "gpu_number": GPU_NUMBER,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": CROPPED_IMAGE_PATH,
        "batch_size": 1,
        "seed": 0,
        "augmentation": True,
        "num_epochs": NUM_EPOCHS,
        "use_heatmaps": True,
        "heatmaps_path": HEATMAPS_PATH,
        "use_hdf5": False,
        "model_mode": MODELMODES.VIEW_SPLIT,
        "model_path": IMAGEHEATMAPS_MODEL_PATH,
    }
    load_run_save(
        data_path=EXAM_LIST_PATH,
        output_path=IMAGEHEATMAPS_PREDICTIONS_PATH,
        parameters=parameters,
    ) 
    
    image_df = pd.read_csv(IMAGE_PREDICTIONS_PATH)
    image_and_heatmap_df = pd.read_csv(IMAGEHEATMAPS_PREDICTIONS_PATH)

    image_df['type'] = 'image only'
    image_and_heatmap_df['type'] = 'image+heatmaps'
    final_df = pd.concat([image_df, image_and_heatmap_df], ignore_index=True)
    final_df.set_index('type')
    results = final_df.to_json(orient='records')

    
    return results
        
@app.route("/")
def home():
    return render_template('upload.html')

@app.route("/api/v0", methods=["POST"])
def process_api():
    return get_predictions()

@app.route("/upload", methods=["POST"])
def upload():
    return get_predictions()
    


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5002)