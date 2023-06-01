#-*- coding: UTF-8 -*-
import cv2
import time
import socket
from .utils import *


import rclpy
import cv2
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from rclpy.qos import qos_profile_sensor_data

class DaBaiSubscriber(Node):

        def __init__(self):
                super().__init__('DaBai_Subscriber')
                self.color_subscription = self.create_subscription(Image,'/camera/color/image_raw',
                self.color_subscription_callback,qos_profile_sensor_data)
                self.color_subscription  # prevent unused variable warning

                self.depth_subscription = self.create_subscription(Image,'/camera/depth/image_raw',
                self.depth_subscription_callback,
                qos_profile_sensor_data)
                self.depth_subscription  # prevent unused variable warning

                # 服务端ip地址
                HOST = '192.168.180.8'
                # 服务端端口号
                PORT = 8080
                ADDRESS = (HOST, PORT)

                # 创建一个套接字
                self.tcpClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 连接远程ip
                print('------正在连接计算棒TCP服务------')
                self.tcpClient.connect(ADDRESS)
                print('------已连接计算棒TCP服务------')


        def color_subscription_callback(self, msg):
                self.color_width = msg.width
                self.color_height = msg.height
                self.color_data = np.array(msg.data).reshape((self.color_height,self.color_width,3))
                self.srcimg = cv2.cvtColor(self.color_data,cv2.COLOR_RGB2BGR)

                 # 计时
                start = time.perf_counter()
                # 读取图像
                # srcimg = cv2.imread(r'/workspaces/rknn-toolkit/nanodet-client/img3.jpg')
                # 预处理图片，得到图片预处理后的数据，只要数据，发送到服务端的还是原图
                cv_image, newh, neww, top, left = pre_process(self.srcimg)
                # 压缩图像
                img_encode = cv2.imencode('.jpg', self.srcimg, [cv2.IMWRITE_JPEG_QUALITY, 99])[1]
                # 转换为字节流
                bytedata = img_encode.tostring()
                # 标志数据，包括待发送的字节流长度等数据，用‘,’隔开
                flag_data = (str(len(bytedata))).encode() + ",".encode() + " ".encode()
                self.tcpClient.send(flag_data)
                # 接收服务端的应答
                data = self.tcpClient.recv(1024)
                if ("ok" == data.decode()):
                    # 服务端已经收到标志数据，开始发送图像字节流数据
                    self.tcpClient.send(bytedata)
                # 接收服务端的应答
                data = self.tcpClient.recv(1024)

                result_ = np.fromstring(data).reshape((-1,6))
                print(result_)
                det_bboxes, det_conf, det_classid = result_[:,:4],result_[:,4],result_[:,5].astype(int)
                # print(det_bboxes)
                # print(det_conf)
                # print(det_classid)
                result_img = img_draw(self.srcimg,det_bboxes, det_conf, det_classid,newh, neww, top, left)
                # cv2.imshow('result',result_img)
                # cv2.waitKey(0)
                # cv2.destroyAllwindows()
                # cv2.imwrite(r'/workspaces/rknn-toolkit/nanodet-client/img3_result.jpg',result_img)
                #     # 计算发送完成的延时
                print("延时：" + str(int((time.perf_counter() - start) * 1000)) + "ms")


                # self.get_logger().info('color_height: "%s"' % self.color_height)
                # self.get_logger().info('color_width: "%s"' % self.color_width)
                # self.get_logger().info('data: "%s"' % self.data)
                cv2.imshow('img',result_img)
                cv2.waitKey(1)

        def depth_subscription_callback(self, msg):
                self.depth_width = msg.width
                self.depth_height = msg.height
                self.depth_data = msg.data
                self.color_data = np.array(msg.data).reshape((self.depth_height,self.depth_width,2))[:,:,0]

                # self.get_logger().info('depth height: "%s"' % self.depth_height)
                # self.get_logger().info('depth width: "%s"' % self.depth_width)
                # self.get_logger().info('depth data: "%s"' % self.color_data)
                # cv2.imshow('depth0',self.color_data)
                # cv2.waitKey(1)




def main(args=None):
        rclpy.init(args=args)

        minimal_subscriber = DaBaiSubscriber()

        rclpy.spin(minimal_subscriber)

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        minimal_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
        main()









   