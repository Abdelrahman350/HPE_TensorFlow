import cv2
from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)), (0,0,1), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)), (0,1,0), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)), (1,0,0), 2)
    return img
    
def show(image):
    plt.figure()
    plt.imshow(image)
    plt.axis('off')

def plot_gt_predictions(images, gt_poses, predicted_poses, batch_number, backbone, show=False, threshold=10):
    yaw, pitch, roll = gt_poses
    yaw_p, pitch_p, roll_p = predicted_poses
    i = 0
    for (image, yaw_i, pitch_i, roll_i, yaw_p_i, pitch_p_i, roll_p_i) in zip(images,
     yaw, pitch, roll, yaw_p, pitch_p, roll_p):
        image_predicted = image.copy()
        image = draw_axis(image, yaw_i[1], pitch_i[1], roll_i[1], tdx=None, tdy=None, size = 100)
        image_predicted = draw_axis(image_predicted, yaw_p_i, pitch_p_i, roll_p_i, tdx=None, tdy=None, size = 100)
        
        if show:
            f, axarr = plt.subplots(1,2, figsize=(10, 20))
            axarr[0].set_title(f"yaw={yaw_i[1]:.2f}, pitch={pitch_i[1]:.2f}, roll={roll_i[1]:.2f}")
            axarr[1].set_title(f"yaw={yaw_p_i[0]:.2f}, pitch={pitch_p_i[0]:.2f}, roll={roll_p_i[0]:.2f}")
            axarr[0].imshow(image)
            axarr[1].imshow(image_predicted)
            plt.show()
        
        error = abs(yaw_i[1]-yaw_p_i) + abs(pitch_i[1]-pitch_p_i) + abs(roll_i[1]-roll_p_i)
        yaw_error = abs(yaw_i[1]-yaw_p_i)
        pitch_error = abs(pitch_i[1]-pitch_p_i)
        roll_error = abs(roll_i[1]-roll_p_i)
        
        if yaw_error > threshold:
            final_frame = cv2.hconcat((image, image_predicted))
            final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"../Outputs/{backbone}/image_{batch_number}_{i}_yaw_error_{yaw_error}.png", final_frame*255)
        if pitch_error > threshold:
            final_frame = cv2.hconcat((image, image_predicted))
            final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"../Outputs/{backbone}/image_{batch_number}_{i}_pitch_error_{pitch_error}.png", final_frame*255)
        if roll_error > threshold:
            final_frame = cv2.hconcat((image, image_predicted))
            final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"../Outputs/{backbone}/image_{batch_number}_{i}_roll_error_{roll_error}.png", final_frame*255)
        if error > threshold*4:
            final_frame = cv2.hconcat((image, image_predicted))
            final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"../Outputs/{backbone}/image_{batch_number}_{i}_all_error_{error}.png", final_frame*255)
        i = i+1
