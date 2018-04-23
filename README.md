# Grocery Identification In Smart Refrigerators 

## Installation Instruction

## Tensorflow object detection API depends upon the following library:

-  Protobuf 3.4.0
-  Python
-  Pillow 1.0
-  lxml
-  tf Slim (which is included in the "tensorflow/models/research/")
-  Jupyter notebook
-  Matplotlib
-  Tensorflow
-  Cython

First clone or download the github of tensorflow,

*git clone https://github.com/tensorflow/models.git*

Next, run the command

```
#### For CPU
pip install tensorflow
#### For GPU
pip install tensorflow-gpu

```
in your terminal window. If you have tensorflow already installed, upgrade it to the latest version. For upgarding tensorflow version, run the command

```
pip install tensorflow --upgrade

```

in your terminal window. Now install all the dependent libraries by running

```
pip install Cython

pip install jupyter

pip install matplotlib

```
## For Protobuf Compilation 
Download the protobuf 4.0 executable from [here](https://github.com/google/protobuf/releases)
Unzip the file and copy protoc.exe from /protoc/bin folder and copy it to models/research folder
Navigate to /models/research folder on the terminal widow and execute
```
protoc object_detection/protos/*.proto --python_out=.

```
Set the environment variable **PYTHONPATH** and set the value as the path to the research folder and also the path till slim folder (which is inside models/research/slim folder inside tensorflow github)

## Testing the installation 
we can test that we have correctly installed the Tensorflow Object Detection API by running the following command:
```
python object_detection/builders/model_builder_test.py

```

## Dataset
Dataset used is [Fruit-360] (https://www.kaggle.com/moltean/fruits) for image classifications and scrapped images from internet for object detection

## Setting up Object Detection API

### Picking Model Parameters
There are a large number of model parameters to configure. The best settings will depend on your given application. Faster R-CNN models are better suited to cases where high accuracy is desired and latency is of lower priority. Conversely, if processing time is the most important factor, SSD models are recommended. In our case we have used RFCN model.

### Defining Inputs
The Tensorflow Object Detection API accepts inputs in the TFRecord file format. Users must specify the locations of both the training and evaluation files. Additionally, users should also specify a label map, which define the mapping between a class id and class name. The label map should be identical between training and evaluation datasets.

An example input configuration looks as follows:

```
tf_record_input_reader {
  input_path: "/usr/home/username/data/train.record"
}
label_map_path: "/usr/home/username/data/label_map.pbtxt"

```
Users should substitute the input_path and label_map_path arguments and insert the input configuration into the train_input_reader and eval_input_reader fields in the skeleton configuration.

### Preparing Inputs TF Records
Tensorflow Object Detection API reads data using the TFRecord file format.

Images are labelled using lblImg [https://github.com/tzutalin/labelImg] which is in xml format. 
XML files are then converted to csv format using xml_to_csv.py. This python file is present in the repository.

To convert these into TFRecords, run the following commands:

```
python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record

python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record

```
## Tensorflow model

Model used for object detection is coco trained rfcn_resnet101_coco which can be downloaded from [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md] with the configuration file rfcn_resnet101_coco.config from [https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs]

Run the below command to start training your model
```
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/rfcn_resnet101_coco.config

```
Navigate to models/object_detection, via terminal, and run the command:

```
tensorboard --logdir='training'

```
to start the TensorBoard. On Tensorboard you can monitor the graph for different loss functions and understand how model is trainig on your dataset.

To see how the model is performing on your dataset, navigate to your TensorFlow object detection folder and copy the export_inference_graph.py file into the research folder and run the below command.

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix training/model.ckpt-160 --output_directory groceryIdentfication

```
In the above command, output_directory is the name of the output folder where your frozen file would be created.

Run **Object_detection.ipynb** jupyter notebook to test your model on your dataset by passing it test images

# Contributors
-  Apoorva Lakhmani [https://github.com/lakhmania/]
-  Neha Lalwani [https://github.com/LalwaniN/]
-  Nirali Merchant [https://github.com/nirali-merchant/]
