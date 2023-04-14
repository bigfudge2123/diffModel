import json
from math import cos, sin, pi
import math
 
# 将欧拉角转换为旋转矩阵
def xyz2r(x,y,z):
    x, y, z  = x*pi/180 ,y*pi/180 ,z*pi/180
    r=[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0]]
    r[0][0],r[0][1],r[0][2]=cos(z)*cos(y), cos(z)*sin(y)*sin(x) - sin(z)*cos(x), cos(z)*sin(y)*cos(x) + sin(z)*sin(x)
    r[1][0],r[1][1],r[1][2]=sin(z)*cos(y), sin(z)*sin(y)*sin(x) + cos(z)*cos(x), sin(z)*sin(y)*cos(x) - cos(z)*sin(x)
    r[2][0],r[2][1],r[2][2]=-sin(y), cos(y)*sin(x), cos(y)*cos(x)
    return r
 
# 输入分别为相机到物体的距离，相机的方位角以及俯仰角；输出为相机平移参数
def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)
 
if __name__ == "__main__":
    # 定义的欧拉角：分别为方位角、俯仰角和翻滚角
    view_params=[[]for i in range(109)]
    for i in range(72):
        view_params[i]=[5*i,30,0]
    for i in range(36):
        view_params[i+72]=[10*i,60,0]
    view_params[108]=[0,90,0]
    
    dist=3    # 相机到物体的距离
    fov=0.6911111611634243    # 手动计算的相机水平视场角，计算公式为2*math.atan(36/(2*50))，其中36为传感器尺寸，50为焦距
    jsontext={'camera_angle_x':fov,'frames':[]}    # 定义一个字典
    
    # 获得每张图像的外参矩阵
    i=0
    for param in view_params:
        azimuth_deg = param[0]
        elevation_deg = param[1]
        theta_deg = param[2]
        r=xyz2r(azimuth_deg,elevation_deg,theta_deg)
        r[0][3],r[1][3],r[2][3]=obj_centened_camera_pos(dist,azimuth_deg,elevation_deg)
        jsontext['frames'].append({'file_path':'./train/%03d'%(round(i)),'transform_matrix':r})
        i+=1
    
    jsondata = json.dumps(jsontext,indent=4,separators=(',', ': '))# 生成json文件，后面的参数是格式控制
    f = open('transform_train.json', 'w')
    f.write(jsondata)
    f.close()