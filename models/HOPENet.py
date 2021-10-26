import tensorflow as tf
from tensorflow.losses import softmax_cross_entropy
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
from tensorflow.compat.v1.losses import huber_loss, mean_squared_error
from tensorflow.keras.losses import MAE
from tensorflow.keras.callbacks import ModelCheckpoint
from .backbones.resnet import ResNet10, ResNet18
from utils.data_utils.plotting_data import plot_gt_predictions

class HOPENet:
    def __init__(self, train_dataset, valid_dataset, class_num, input_size, backbone='ResNet18', loss='wrapped'):
        self.class_num = class_num
        self.input_size = input_size
        self.idx_tensor = [idx for idx in range(self.class_num)]
        self.idx_tensor = tf.Variable(np.array(self.idx_tensor, dtype=np.float32))
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.epochs = 3
        self.backbone = backbone
        self.backbone = self.__backbone_handler()
        self.loss = loss
        self.__loss_angle = self.__loss_handler()
        self.model = self.__create_model()
        self.model_fit = None

    def __create_model(self):
        inputs = tf.keras.layers.Input(shape=(self.input_size, self.input_size, 3))
        
        feature = self.backbone(weights='imagenet', include_top=False,
         input_shape=(self.input_size, self.input_size, 3))(inputs)
        
        feature = GlobalAveragePooling2D()(feature)
        fc_yaw = tf.keras.layers.Dense(name='yaw', units=self.class_num)(feature)
        fc_pitch = tf.keras.layers.Dense(name='pitch', units=self.class_num)(feature)
        fc_roll = tf.keras.layers.Dense(name='roll', units=self.class_num)(feature)
    
        model = tf.keras.Model(inputs=inputs, outputs=[fc_yaw, fc_pitch, fc_roll], name='HOPENet')

        losses = {
            'yaw':self.__loss_angle,
            'pitch':self.__loss_angle,
            'roll':self.__loss_angle,
        }
        
        model.compile(optimizer= tf.compat.v1.train.AdamOptimizer(learning_rate=0.00001), loss=losses)
        return model
    
    def __backbone_handler(self):
        if self.backbone == 'ResNet18':
            return ResNet18
        elif self.backbone == 'ResNet10':
            return ResNet10
        else:
            print("This backbone is not implemented")
            return

    def __loss_handler(self):
        if self.loss == 'wrapped':
            return self.__wrapped_loss
        elif self.loss == 'huber':
            return self.__Huber_loss
        elif self.loss == 'mse':
            return self.__MSE_loss
        elif self.loss == 'mae':
            return self.__MAE_loss
        else:
            print("This loss is not implemented")
            return

    def __wrapped_loss(self, y_true, y_pred, alpha=0.5, beta=2):
        # cross entropy loss
        bin_true = y_true[:,0]
        cont_true = y_true[:,1]
        oh_labels = tf.cast(tf.one_hot(tf.cast(bin_true, tf.int32), 66), dtype=bin_true.dtype)

        cls_loss = softmax_cross_entropy(onehot_labels=oh_labels, logits=y_pred)
        # Wrapped loss
        pred_cont = tf.reduce_sum(tf.nn.softmax(y_pred) * self.idx_tensor, 1) * 3 - 99
        MAWE = MAE(y_true=cont_true, y_pred=pred_cont)
        reg_loss = tf.math.minimum(MAWE**2, (360-MAWE)**2)
        # Total loss
        total_loss = alpha * reg_loss + beta * cls_loss
        return total_loss
        
    def __Huber_loss(self, y_true, y_pred, alpha=0.5, beta=2):
        # cross entropy loss
        bin_true = y_true[:,0]
        cont_true = y_true[:,1]
        oh_labels = tf.cast(tf.one_hot(tf.cast(bin_true, tf.int32), 66), dtype=bin_true.dtype)

        cls_loss = softmax_cross_entropy(onehot_labels=oh_labels, logits=y_pred)
        # Huber loss
        pred_cont = tf.reduce_sum(tf.nn.softmax(y_pred) * self.idx_tensor, 1) * 3 - 99
        reg_loss = huber_loss(y_true=cont_true, y_pred=pred_cont)
        # Total loss
        total_loss = alpha * reg_loss + beta * cls_loss
        return total_loss
    
    def __MSE_loss(self, y_true, y_pred, alpha=0.5, beta=2):
        # cross entropy loss
        bin_true = y_true[:,0]
        cont_true = y_true[:,1]
        oh_labels = tf.cast(tf.one_hot(tf.cast(bin_true, tf.int32), 66), dtype=bin_true.dtype)

        cls_loss = softmax_cross_entropy(onehot_labels=oh_labels, logits=y_pred)
        # MSE loss
        pred_cont = tf.reduce_sum(tf.nn.softmax(y_pred) * self.idx_tensor, 1) * 3 - 99
        reg_loss = mean_squared_error(y_true=cont_true, y_pred=pred_cont)
        # Total loss
        total_loss = alpha * reg_loss + beta * cls_loss
        return total_loss
    
    def __MAE_loss(self, y_true, y_pred, alpha=0.5, beta=2):
        # cross entropy loss
        bin_true = y_true[:,0]
        cont_true = y_true[:,1]
        oh_labels = tf.cast(tf.one_hot(tf.cast(bin_true, tf.int32), 66), dtype=bin_true.dtype)

        cls_loss = softmax_cross_entropy(onehot_labels=oh_labels, logits=y_pred)
        # MAE loss
        pred_cont = tf.reduce_sum(tf.nn.softmax(y_pred) * self.idx_tensor, 1) * 3 - 99
        reg_loss = MAE(y_true=cont_true, y_pred=pred_cont)
        # Total loss
        total_loss = alpha * reg_loss + beta * cls_loss
        return total_loss

    def train(self, model_name, load_weight=True, epochs=3):
        self.epochs = epochs
        relative_path = "checkpoints/"
        model_path = relative_path + model_name
        self.model.summary()
        if load_weight:
            self.model.load_weights(model_path)

        model_checkpoint_callback = ModelCheckpoint(
            filepath=model_path,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1)

        self.model_fit = self.model.fit(
            x=self.train_dataset,
            validation_data=self.valid_dataset,
            epochs=self.epochs, 
            verbose=1,
            callbacks=[model_checkpoint_callback])            
    
    def predict(self, batch_images):
        predictions = self.model.predict(batch_images, verbose=1)
        predictions = np.asarray(predictions)
        pred_cont_yaw = tf.reduce_sum(tf.nn.softmax(predictions[0,:,:]) * self.idx_tensor, 1) * 3 - 99
        pred_cont_pitch = tf.reduce_sum(tf.nn.softmax(predictions[1,:,:]) * self.idx_tensor, 1) * 3 - 99
        pred_cont_roll = tf.reduce_sum(tf.nn.softmax(predictions[2,:,:]) * self.idx_tensor, 1) * 3 - 99
        predictions = pred_cont_yaw, pred_cont_pitch, pred_cont_roll
        return predictions

    def test(self, dataset_test):
        test_accuracy_yaw = 0
        test_accuracy_pitch = 0
        test_accuracy_roll = 0

        for i, (images, [batch_yaw, batch_pitch, batch_roll]) in enumerate(dataset_test):
            predictions = self.model.predict(images, verbose=1)
            test_accuracy_yaw += self.__loss_angle(batch_yaw, predictions[0])
            test_accuracy_pitch += self.__loss_angle(batch_pitch, predictions[1])
            test_accuracy_roll += self.__loss_angle(batch_roll, predictions[2])
            
        print("Test MSE for Yaw: {:.3%}".format(test_accuracy_yaw/(i+1)))
        print("Test MSE for pitch: {:.3%}".format(test_accuracy_pitch/(i+1)))
        print("Test MSE for roll: {:.3%}".format(test_accuracy_roll/(i+1)))
    
    def test_history(self, dataset_test, model_name):
        relative_path = "checkpoints/"
        model_path = relative_path + model_name
        self.model.load_weights(model_path)
        f = open(relative_path + 'csvfile_valid.csv','w')
        f.write('yaw_true, yaw_predicted, pitch_true, pitch_predicted, roll_true, roll_predicted\n')

        count = 0
        for i, (images, [batch_yaw, batch_pitch,
                         batch_roll]) in enumerate(dataset_test):
            predictions = self.model.predict(images, verbose=1)
            predictions = np.asarray(predictions)
            pred_cont_yaw = tf.reduce_sum(tf.nn.softmax(predictions[0,:,:]) * self.idx_tensor, 1) * 3 - 99
            pred_cont_pitch = tf.reduce_sum(tf.nn.softmax(predictions[1,:,:]) * self.idx_tensor, 1) * 3 - 99
            pred_cont_roll = tf.reduce_sum(tf.nn.softmax(predictions[2,:,:]) * self.idx_tensor, 1) * 3 - 99
            for i in range(len(batch_yaw)):
                f.write(str(batch_yaw[i][1])+\
                    ', '+str(pred_cont_yaw[i].numpy())+\
                        ', '+str(batch_pitch[i][1])+', '+\
                            str(pred_cont_pitch[i].numpy())+\
                                ', '+str(batch_roll[i][1])+', '+str(pred_cont_roll[i].numpy())+'\n')
        f.close()
    
    def investigate_failed_images(self, dataset_test):
        for i, (images, [batch_yaw, batch_pitch, batch_roll]) in enumerate(dataset_test):
            predictions = self.predict(images)
            gt_poses = batch_yaw, batch_pitch, batch_roll
            plot_gt_predictions(images, gt_poses, predictions, i, backbone=self.backbone)
            