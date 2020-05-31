import numpy as np
import os
import struct
import open3d as o3d
import vtk
import math
from pyquaternion import Quaternion


# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def bbox2vtk(data, attributes, vtk_file):
    points_data = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    num_cells = data.shape[0]
    for i in range(num_cells):
        x = 0.5 * data[i, 3]  # len, x
        y = 0.5 * data[i, 4]  # width, y
        h = 0.5 * data[i, 5]  # height
        yaw = data[i, 6]
        q = Quaternion(axis=[0,0,1], angle=yaw)
        v00 = q.rotate([-x, -y, -h])
        v01 = q.rotate([-x, y, -h])
        v02 = q.rotate([x, y, -h])
        v03 = q.rotate([x, -y, -h])
        points_data.InsertNextPoint(v00[0], v00[1], -h)
        points_data.InsertNextPoint(v01[0], v01[1], -h)
        points_data.InsertNextPoint(v02[0], v02[1], -h)
        points_data.InsertNextPoint(v03[0], v03[1], -h)

        points_data.InsertNextPoint(v00[0], v00[1], h)
        points_data.InsertNextPoint(v01[0], v01[1], h)
        points_data.InsertNextPoint(v02[0], v02[1], h)
        points_data.InsertNextPoint(v03[0], v03[1], h)
        cells.InsertNextCell(5, [0, 1, 2, 3, 0])
        cells.InsertNextCell(5, [4, 5, 6, 7, 4])
        cells.InsertNextCell(2, [0, 4])
        cells.InsertNextCell(2, [1, 5])
        cells.InsertNextCell(2, [2, 6])
        cells.InsertNextCell(2, [3, 7])
    bx_attrs = []
    for attr in attributes:
        if attr[1] == 'int1':
            tmp_attr = vtk.vtkIntArray()
            tmp_attr.SetName(attr[0])
            tmp_attr.SetNumberOfComponents(1)
            tmp_attr.SetNumberOfTuples(num_cells*6)
            for i in range(0, num_cells):
                tmp_attr.SetTuple1(i*6, attr[2][i])
                tmp_attr.SetTuple1(i * 6+1, attr[2][i])
                tmp_attr.SetTuple1(i * 6+2, attr[2][i])
                tmp_attr.SetTuple1(i * 6+3, attr[2][i])
                tmp_attr.SetTuple1(i * 6+4, attr[2][i])
                tmp_attr.SetTuple1(i * 6+5, attr[2][i])
            bx_attrs.append(tmp_attr)
        elif attr[1] == 'float1':
            tmp_attr = vtk.vtkFloatArray()
            tmp_attr.SetName(attr[0])
            tmp_attr.SetNumberOfComponents(1)
            tmp_attr.SetNumberOfTuples(num_cells*6)
            for i in range(0, num_cells):
                tmp_attr.SetTuple1(i*6, attr[2][i])
                tmp_attr.SetTuple1(i * 6+1, attr[2][i])
                tmp_attr.SetTuple1(i * 6+2, attr[2][i])
                tmp_attr.SetTuple1(i * 6+3, attr[2][i])
                tmp_attr.SetTuple1(i * 6+4, attr[2][i])
                tmp_attr.SetTuple1(i * 6+5, attr[2][i])
            bx_attrs.append(tmp_attr)

    bboxes = vtk.vtkPolyData()
    bboxes.SetPoints(points_data)
    bboxes.SetLines(cells)
    for bx_at in bx_attrs:
        bboxes.GetCellData().AddArray(bx_at)
    poly_data_writer = vtk.vtkPolyDataWriter()
    poly_data_writer.SetInputData(bboxes)
    poly_data_writer.SetFileName(vtk_file)
    poly_data_writer.SetFileTypeToBinary()
    poly_data_writer.Write()


def points2vtk(pt_data_raw, attributes, vtk_file):
    points_data = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    points_cloud = vtk.vtkPolyData()
    num_pts = pt_data_raw.shape[0]
    for i in range(num_pts):
        id = points_data.InsertNextPoint(pt_data_raw[i, 0], pt_data_raw[i, 1], pt_data_raw[i, 2])
        cells.InsertNextCell(1)
        cells.InsertCellPoint(id)

    pc_attrs = []
    for attr in attributes:
        if attr[1] == 'uchar3':
            tmp_attr = vtk.vtkUnsignedCharArray()
            tmp_attr.SetName(attr[0])
            tmp_attr.SetNumberOfComponents(3)
            tmp_attr.SetNumberOfTuples(num_pts)
            for i in range(0, num_pts):
                tmp_attr.SetTuple3(i, attr[2][i, 0], attr[2][i, 1], attr[2][i, 2])
            pc_attrs.append(tmp_attr)
        elif attr[1] == 'int1':
            tmp_attr = vtk.vtkIntArray()
            tmp_attr.SetName(attr[0])
            tmp_attr.SetNumberOfComponents(1)
            tmp_attr.SetNumberOfTuples(num_pts)
            for i in range(0, num_pts):
                tmp_attr.SetTuple1(i, attr[2][i])
            pc_attrs.append(tmp_attr)
        elif attr[1] == 'float1':
            tmp_attr = vtk.vtkFloatArray()
            tmp_attr.SetName(attr[0])
            tmp_attr.SetNumberOfComponents(1)
            tmp_attr.SetNumberOfTuples(num_pts)
            for i in range(0, num_pts):
                tmp_attr.SetTuple1(i, attr[2][i])
            pc_attrs.append(tmp_attr)
        elif attr[1] == 'float3':
            tmp_attr = vtk.vtkFloatArray()
            tmp_attr.SetName(attr[0])
            tmp_attr.SetNumberOfComponents(3)
            tmp_attr.SetNumberOfTuples(num_pts)
            for i in range(0, num_pts):
                tmp_attr.SetTuple3(i, attr[2][i,0], attr[2][i,1], attr[2][i,2])
            pc_attrs.append(tmp_attr)

    points_cloud.SetPoints(points_data)
    points_cloud.SetVerts(cells)
    for pc_at in pc_attrs:
        points_cloud.GetPointData().AddArray(pc_at)

    poly_data_writer = vtk.vtkPolyDataWriter()
    poly_data_writer.SetInputData(points_cloud)
    poly_data_writer.SetFileName(vtk_file)
    poly_data_writer.SetFileTypeToBinary()
    poly_data_writer.Write()


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_calib(calib_path):
    """
    读取标定信息, 从second上摘抄
    """
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P0 = np.array(
        [float(info) for info in lines[0].split(' ')[1:13]]).reshape(
        [3, 4])
    P1 = np.array(
        [float(info) for info in lines[1].split(' ')[1:13]]).reshape(
        [3, 4])
    P2 = np.array(
        [float(info) for info in lines[2].split(' ')[1:13]]).reshape(
        [3, 4])
    P3 = np.array(
        [float(info) for info in lines[3].split(' ')[1:13]]).reshape(
        [3, 4])

    P0 = _extend_matrix(P0)
    P1 = _extend_matrix(P1)
    P2 = _extend_matrix(P2)
    P3 = _extend_matrix(P3)
    image_info = {}
    image_info['calib/P0'] = P0
    image_info['calib/P1'] = P1
    image_info['calib/P2'] = P2
    image_info['calib/P3'] = P3
    R0_rect = np.array([
        float(info) for info in lines[4].split(' ')[1:10]
    ]).reshape([3, 3])

    rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
    rect_4x4[3, 3] = 1.
    rect_4x4[:3, :3] = R0_rect

    image_info['calib/R0_rect'] = rect_4x4
    Tr_velo_to_cam = np.array([
        float(info) for info in lines[5].split(' ')[1:13]
    ]).reshape([3, 4])
    Tr_imu_to_velo = np.array([
        float(info) for info in lines[6].split(' ')[1:13]
    ]).reshape([3, 4])
    Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
    Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
    image_info['calib/Tr_velo_to_cam'] = Tr_velo_to_cam
    image_info['calib/Tr_imu_to_velo'] = Tr_imu_to_velo
    return image_info


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
        -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0],))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


if __name__ == '__main__':
    pt_data = read_velodyne_bin('/home/wegatron/data_set_kitti/kitti_object/train/velodyne/000000.bin')
    # # pc = o3d.geometry.PointCloud()
    # # pc.points = o3d.utility.Vector3dVector(pt_data)
    # # open3d.geometry.OrientedBoundingBox()
    # # o3d.visualization.draw_geometries_with_editing([pc])
    # attribuits = []
    # num_pts = pt_data.shape[0]
    # vcolor = np.empty([num_pts, 3], dtype=np.uint8) # rgb
    # vvals = np.empty([num_pts], dtype=float)
    # for i in range(num_pts):
    #     vcolor[i] = [(i%255)/255.0, (255-i%255)/255.0, 0]
    #     vvals[i] = i
    # attribuits.append(('colorRGB', 'uchar3', vcolor))
    # attribuits.append(('vals', 'float1', vvals))
    points2vtk(pt_data, [], '/home/wegatron/tmp/test.vtk')

    # load gth labels
    calib_info = get_calib('/home/wegatron/data_set_kitti/kitti_object/training/calib/000000.txt')


    # oritented_bbox_data = np.empty([1, 7]) # center x,y,z, h, w, length, yaw(-pi, pi)
    # oritented_bbox_data[0, 0] = 0
    # oritented_bbox_data[0, 1] = 0
    # oritented_bbox_data[0, 2] = 0
    # oritented_bbox_data[0, 3] = 1
    # oritented_bbox_data[0, 4] = 2
    # oritented_bbox_data[0, 5] = 3
    # oritented_bbox_data[0, 6] = math.pi/2
    #
    # scores = np.empty([1])
    # scores[0] = 8.0
    # attribuits = []
    # attribuits.append(('score', 'float1', scores))
    # bbox2vtk(oritented_bbox_data, attribuits, '/home/wegatron/tmp/bbox.vtk')
