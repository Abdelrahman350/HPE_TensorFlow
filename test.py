import tensorflow as tf
import os
from utils.data_utils.data_preparing import load_json
from data_generator.data_generator import DataGenerator
from models.HOPENet import HOPENet
from utils.data_utils.plotting_data import plot_gt_predictions
tf.compat.v1.enable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#from tensorflow import ConfigProto
#from tensorflow import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


json_file_path = "../../Datasets/labels.json"
labels = load_json(json_file_path)
        
json_file_path = "../../Datasets/partition.json"
partition = load_json(json_file_path)
        
json_file_path = "../../Datasets/pt2d.json"
pt2d = load_json(json_file_path)

training_data_generator = DataGenerator(partition['train'], labels, pt2d,
 batch_size=32, input_shape=(224, 224, 3), shuffle=True)

validation_data_generator = DataGenerator(partition['valid'], labels, pt2d,
 batch_size=32, input_shape=(224, 224, 3), shuffle=True)

net = HOPENet(train_dataset=training_data_generator,
 valid_dataset=validation_data_generator, class_num=66, input_size=224, loss='huber')
net.model.load_weights("checkpoints/HOPNet_LP.h5")
net.model.summary()

net.test(validation_data_generator)
#net.investigate_failed_images(validation_data_generator)
