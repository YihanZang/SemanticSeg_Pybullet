import os
import numpy as np
import datetime
import logging
import torch
import torch.optim
import torch.nn as nn
from models import Model
import config
import argparse
from PIL import Image
import acquire
import numpy as np
import pybullet_data
import pybullet as pb
from pybullet_utils import bullet_client
import os
from PIL import Image
import acquire
import cv2

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True

'''参数初始化'''
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('3dgnn')
    
    '''指定循环次数'''
    parser.add_argument('--num_epochs', default=100,type=int,
                        help='Number of epoch')
    
    '''指定批量大小'''
    parser.add_argument('--batchsize', type=int, default=1,
                        help='batch size in training')
                        
    '''修改default，可以使用训练好的参数'''                    
    parser.add_argument('--pretrain', type=str, default='./models/checkpoint_15.pth',
                        help='Direction for pretrained weight')
    
    '''指定GPU，一般从0开始编号'''
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    return parser.parse_args()

def main(args):

    pb_client = bullet_client.BulletClient(connection_mode=pb.GUI)
    pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb_client.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=0, cameraPitch=-45,
        cameraTargetPosition=(0, 0, .5))

    pb_client.setGravity(0, 0, -30)

    for i in range(4):
        x, y, z = np.random.random(3)
        pb_client.loadURDF('./cube_dataset/cube{}.urdf'.format(i),
                           basePosition=[1.5*(x-.5), 1.5*(y-.5), z+1],
                           baseOrientation=pb.getQuaternionFromEuler(2*np.pi*np.random.random(3)),
                           globalScaling=1)
        

    '''指定GPU'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    '''指定日志文件的存放位置'''
    logger = logging.getLogger('3dgnn')
    log_path = './eval/'+ str(datetime.datetime.now().strftime('%Y-%m-%d-%H')).replace(' ', '/') + '/'
    print('log path is:',log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    hdlr = logging.FileHandler(log_path + 'log.txt')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    logger.info("Loading data...")
    print("Loading data...")
    
    '''创建文件夹，保存前向传播后的预测结果'''
    eval_path = './results/'
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    
    '''日志文件需要添加的信息'''
    logger.info("Preparing model...")
    print("Preparing model...")
    
    '''模型初始化'''
    model = Model(config.nclasses, config.mlp_num_layers,config.use_gpu)
    loss = nn.NLLLoss(reduce=not config.use_bootstrap_loss, weight=torch.FloatTensor(config.class_weights))
    
    '''dim表示维度，dim=0,表示行，dim=1，表示列'''
    softmax = nn.Softmax(dim=1)
    log_softmax = nn.LogSoftmax(dim=1)

    '''使用cuda加速'''
    if config.use_gpu:
        model = model.cuda()
        loss = loss.cuda()
        softmax = softmax.cuda()
        log_softmax = log_softmax.cuda()
    

    
    '''评估/预测,对输入数据进行评估/预测'''

    def eval_set():
        model.eval()
        
        '''torch.no_grad()函数使得程序不计算梯度，只进行前向传播，用在预测中正合适'''
        with torch.no_grad():
            loss_sum = 0.0
            

            for step in range(100000):
                pb_client.stepSimulation()
                width, height, view_matrix, projection_matrix, _, _, _, _, _, _, _, _ = pb_client.getDebugVisualizerCamera()

                width=640
                height=480
                
                x0,xy0 =acquire.getcamera(pb_client, width, height, view_matrix, projection_matrix, step)

                x1 = np.expand_dims(x0, axis=0)
                xy1 = np.expand_dims(xy0, axis=0)

                x = torch.from_numpy(x1)
                xy = torch.from_numpy(xy1)
                x = x.float()
                xy = xy.float()
                print("1111111111111111111111111111111111111111111")
                print(x.shape)
                
                '''permute函数用于转换Tensor的维度，contiguous()使得内存是连续的'''
                input = x.permute(0, 3, 1, 2).contiguous()
                xy = xy.permute(0, 3, 1, 2).contiguous()
                if config.use_gpu:
                    input = input.cuda()
                    xy = xy.cuda()
                    # target = target.cuda()
                
                '''经过网络，计算输出, 维度为 ([6, 14, 640, 480])'''
                output = model(input, gnn_iterations=config.gnn_iterations, k=config.gnn_k, xy=xy, use_gnn=config.use_gnn)

                
                '''pred维度为 ([6, 640, 480, 14]), 连续内存'''
                pred = output.permute(0, 2, 3, 1).contiguous()
                '''源程序没有，专门存放这个batch的大小，后面循环用到'''
                name_for_txt=len(pred)
                '''此时pred维度为 ([1843200, 14]), 其中1843200=6*640*480  config.nclasses=14'''
                pred = pred.view(-1, config.nclasses)
                '''每一行进行softmax运算，相当于对每一个像素的分类进行softmax运算'''
                pred = softmax(pred)
                '''pred_max_val, pred_arg_max都是1843200维，分别存储每个像素最大的分类概率及分类'''
                pred_max_val, pred_arg_max = pred.max(1)
                
                '''将分类数据保存为数组，并且改变形状，使得每行代表一张图片'''
                result = pred_arg_max.cpu().numpy().reshape(name_for_txt,307200)
                Width = 640
                Height = 480
                '''创建空矩阵，用于存放一张图片每个像素的分类数据'''
                Empty_array = np.zeros((Width,Height,3), dtype = np.uint8)
                
                for ii in range(name_for_txt):
                    row_Frame = result[ii] #将一张图片的数据单独保存
                    for w in range(Width):
                        for h in range(Height):
                            '''判断属于哪一类，不一样的类赋予不同的颜色'''
                            if row_Frame[w*Height+h] == 0:
                                #未知类，RGB为0，黑色
                                Empty_array[w,h,0] = 0
                                Empty_array[w,h,1] = 0
                                Empty_array[w,h,2] = 0
                            elif row_Frame[w*Height+h] == 1:
                                #block1 red
                                Empty_array[w,h,0] = 255
                                Empty_array[w,h,1] = 0
                                Empty_array[w,h,2] = 0
                            elif row_Frame[w*Height+h] == 2:
                                #block2
                                Empty_array[w,h,0] = 255
                                Empty_array[w,h,1] = 255
                                Empty_array[w,h,2] = 0
                            elif row_Frame[w*Height+h] == 3:
                                #block3BLUE
                                Empty_array[w,h,0] = 0
                                Empty_array[w,h,1] = 0
                                Empty_array[w,h,2] = 255
                            elif row_Frame[w*Height+h] == 4:
                                #block4GREEN
                                Empty_array[w,h,0] = 0
                                Empty_array[w,h,1] = 255
                                Empty_array[w,h,2] = 0
                            elif row_Frame[w*Height+h] == 5:
                                #background
                                Empty_array[w,h,0] = 0
                                Empty_array[w,h,1] = 0
                                Empty_array[w,h,2] = 0

                    # 转置后的维度是H*W*3，即rgb默认的顺序
                    useimg = np.transpose(Empty_array, [1, 0, 2])
                    
                    # 实时展示分割图像
                    cv2.imshow('Segmentation', useimg)

                    # 按键响应退出窗口
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    img = Image.fromarray(useimg).convert('RGB').rotate(0)  
                    # 将数组保存为图片
                    print("222222222222222222222222222222222222222222")
                    img.save(eval_path+str(step)+'.png')  
                


    '''Training parameter 训练参数'''
    model_to_load = args.pretrain
    logger.info("num_epochs: %d", args.num_epochs)
    print("Number of epochs: %d"%args.num_epochs)


    '''判断使用原来训练过的模型参数，还是从零开始训练'''
    if model_to_load:
        logger.info("Loading old model...")
        print("Loading old model...")
        model.load_state_dict(torch.load(model_to_load))
    else:
        logger.info("Starting training from scratch...")
        print("Starting training from scratch...")
         

    eval_set()
    pb_client.disconnect()

if __name__ == '__main__':
    args = parse_args()
    main(args)
