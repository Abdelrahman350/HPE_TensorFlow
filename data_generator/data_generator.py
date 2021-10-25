import tensorflow as tf
import cv2
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, labels, pt2d, batch_size=32, input_shape=(128,128,3),
                 shuffle=True, dataset_path='../../Datasets/'):
        self.list_IDs = list_IDs
        self.labels = labels
        self.pt2d = pt2d
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.dataset_path = dataset_path
        self.indices = np.arange(len(self.list_IDs))
        
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def on_epoch_end(self):
        """Shuffle after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def __getitem__(self, index):
        """Get a batch of data"""
        start_index = index * self.batch_size
        end_index = (index+1) * self.batch_size
        batch_IDs = self.indices[start_index : end_index]
        batch = [self.list_IDs[k] for k in batch_IDs]
        X, y = self.__data_generation(batch)
        return X, y
    
    def __data_generation(self, batch):
        # Initializing input data
        X = []
        
        batch_yaw = []
        batch_pitch = []
        batch_roll = []
        
        for index, image_id in enumerate(batch):
            image_path = self.dataset_path + image_id + '.jpg'
            image = self.parse_image(image_path)
            image = self.crop_image(image_id, image)
            if image.shape[0]==0 or image.shape[1]==0:
                continue
            image, binned_pose, cont_labels = self.flip(image_id, image)
            X.append(self.normalization(image))
            batch_yaw.append([binned_pose[0], cont_labels[0]])
            batch_pitch.append([binned_pose[1], cont_labels[1]])
            batch_roll.append([binned_pose[2], cont_labels[2]])

        batch_yaw = np.array(batch_yaw)
        batch_pitch = np.array(batch_pitch)
        batch_roll = np.array(batch_roll)
        X = np.array(X)
        return X, [batch_yaw, batch_pitch, batch_roll]
    
    def data_generation(self, batch):
        # Initializing input data
        X = []
        
        batch_yaw = []
        batch_pitch = []
        batch_roll = []
        
        for index, image_id in enumerate(batch):
            image_path = self.dataset_path + image_id + '.jpg'
            image = self.parse_image(image_path)
            image = self.crop_image(image_id, image)
            if image.shape[0]==0 or image.shape[1]==0:
                continue
            image, binned_pose, cont_labels = self.flip(image_id, image)
            X.append(self.normalization(image))
            batch_yaw.append([binned_pose[0], cont_labels[0]])
            batch_pitch.append([binned_pose[1], cont_labels[1]])
            batch_roll.append([binned_pose[2], cont_labels[2]])

        batch_yaw = np.array(batch_yaw)
        batch_pitch = np.array(batch_pitch)
        batch_roll = np.array(batch_roll)
        X = np.array(X)
        return X, [batch_yaw, batch_pitch, batch_roll]
    
    def parse_image(self, image_path):
        image_ = cv2.imread(image_path)
        image = image_.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def crop_image(self, image_id, image):
        # Crop the face loosely
        pt2d = self.pt2d[image_id]
        x_min = min(pt2d[0])
        y_min = min(pt2d[1])
        x_max = max(pt2d[0])
        y_max = max(pt2d[1])
        
        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        image = image[int(x_min):int(x_min+x_max), int(y_min):int(y_min+y_max)]
        return image
    
    def flip(self, image_id, image):
        # We get the pose in radians
        pose = self.labels[image_id][:3]
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            image = cv2.flip(image, 1)
        
        cont_labels = [yaw, pitch, roll]
        bins = np.array(range(-99, 99, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1
        return image, binned_pose, cont_labels
        
    def normalization(self, image):
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]),
                           interpolation = cv2.INTER_CUBIC)
        image = image.astype(np.float32)
        image /= 255.0
        return image
