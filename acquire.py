import sys, pdb, time
import numpy as np
import pybullet_data
import pybullet as pb
from pybullet_utils import bullet_client
import os
from PIL import Image
import cv2

def getcamera(p, width, height, view_matrix, projection_matrix,step):
    _, _, rgb_pixels, depth_pixels, segmentation_mask = \
            p.getCameraImage(width=width,
                                     height=height,
                                     viewMatrix=view_matrix,
                                     projectionMatrix=projection_matrix
                                     )
    
        # 从摄像头获取图像
    img_arr = rgb_pixels
    #变成640-480-3
    rgb_img = np.transpose(img_arr, (1, 0, 2))
        # 提取RGB图像数据，img_arr是颜色部分okokok维度是640*480*3
    #rgb_img = np.reshape(img_arr, (width, height, 4))

        # 如果你不需要 alpha 通道（透明度），可以去掉
    rgb_img = rgb_img[:, :, :3]

    rgb_img33 = np.transpose(rgb_img, (1, 0, 2))
    cv2.imwrite('C:/Users/ASUS/Desktop/kasut/final3/hy-tmp/3DGNN_pytorch-master/rgb_result/'+str(step)+'.png', rgb_img33)

        # 转换为 float32 类型，类似于之前的 `.astype(np.float32)`
    rgb_img = rgb_img.astype(np.float32)

    #height*width
    depth_image = np.array(depth_pixels)
    depth_image = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)) * 255  # 归一化到 0-255
    depth_image = depth_image.astype(np.uint8)

    hha = save_HHA('C:/Users/ASUS/Desktop/kasut/hhatest/', depth_image, step)
    hha2 = np.transpose(hha, [1, 0, 2])
    hha3 = hha2.astype(np.float32)

    #hha3=np.zeros((640, 480, 3), dtype=np.uint8)

    rgb_hha = np.concatenate([rgb_img, hha3], axis=2).astype(np.float32)

    '''构造和rgb维度相同的零矩阵，因为有0：2，所以维度为 (640, 480, 2)'''
    xy = np.zeros_like(rgb_img)[:,:,0:2].astype(np.float32)

    return rgb_hha, xy

def save_HHA(outDir, depth_image,step):
    """
    生成 HHA 数据并保存为图像文件。
    
    imName: 保存图像的名称
    outDir: 保存路径
    depth_image: 归一化到 0-255 的深度图像
    """
    height, width = depth_image.shape
    
    # 反向深度（视差）
    disparity = 31000.0 / depth_image  # 反向深度（毫米）
    disparity = np.clip(disparity, 0, 255)  # 限制在 0 到 255 之间

    #计算地面以上高度。
    
   # :param depth_image: 深度图像，单位为米
   # :return: 高度图像
    height = depth_image  # 假设高度等于深度
    height = np.clip(height, 0, 255)  # 限制在 0 到 255 之间

# 计算法线
    gradient_x, gradient_y = np.gradient(depth_image)
    normal_x = -gradient_x
    normal_y = -gradient_y
    normal_z = np.ones_like(depth_image)

    # 法线归一化
    normals = np.dstack((normal_x, normal_y, normal_z))
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)  # 归一化法线

    # 计算与重力方向 (0, 0, 1) 的夹角
    gravity_direction = np.array([0, 0, 1])
    angles = np.arccos(np.clip(np.dot(normals, gravity_direction) / (np.linalg.norm(normals, axis=2) * np.linalg.norm(gravity_direction)), -1, 1))
    angles = np.degrees(angles)  # 转换为度
    angles = np.clip(angles, 0, 255)  # 限制在 0 到 255 之间




    hha_image = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
    hha_image[..., 0] = disparity  # 通道 1：水平视差r
    hha_image[..., 1] = height  # 通道 2：高度g
    hha_image[..., 2] = angles  # 通道 3：角度b


    # 保存 HHA 数据
    if outDir:
        os.makedirs(outDir, exist_ok=True)  # 确保输出目录存在
        cv2.imwrite(outDir+str(step)+".png", hha_image)

    return hha_image

