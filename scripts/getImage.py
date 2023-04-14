import argparse
import json
import torch

import nvdiffrast.torch as dr

# Import topology / geometry trainers
from geometry.dmtet import DMTetGeometry
from geometry.dlmesh import DLMesh

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render

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

def getTransformation(NumofImages, filaPath):
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
    return view_params.__len__
    
@torch.no_grad()
def prepare_batch(target, bg_type='black'):
    assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type

    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()
    target['img'] = target['img'].cuda()
    target['background'] = background

    target['img'] = torch.cat((torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]), target['img'][..., 3:4]), dim=-1)

    return target

if __name__ == "__main__":
    Parser = argparse.ArgumentParser(description='getImage')
    Parser.add_argument('--config', type=str, default=None, help='Config file')
    Parser.add_argument('-o', '--out_dir', type=str, default=None)
    Parser.add_argument('-rm', '--ref_mesh', type=str)
    Parser.add_argument('-bm', '--base_mesh', type=str, default=None)
    Parser.add_argument('-di', '--display-interval', type=int, default=1)
    Parser.add_argument('-si', '--save-interval', type=int, default=1)

    Parser.add_argument('-iter', '--iter', type=int, default=100)
    Parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    Parser.add_argument('-s', '--spp', type=int, default=1)

    FLAGS = Parser.parse_args()

    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]

    BaseMesh = mesh.load_mesh(FLAGS.base_mesh,FLAGS.base_mtl)
    Geometry = DLMesh(BaseMesh, FLAGS)
    TestNum = getTransformation(FLAGS.iter, FLAGS.out_dir)

    glctx = dr.RasterizeGLContext()
    for i in range(TestNum):
        TrainImage = Geometry.render(glctx,)
        buffers = geometry.render(glctx, target, lgt, opt_material)
