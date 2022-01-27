import os
import os.path as osp
import json
from scipy.spatial.transform import Rotation
import numpy as np

# 
def load_lidar_result(json_path, id_max=None):
    id_list = list()
    id_list = [47, 32, 86, 56]
    # id_list = [47]
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    ts = sorted(json_data.keys())

    gt_list = list()
    gt_list.append(dict())
    for t in ts:
        frame_dict = dict()
        quat_l = [json_data[t]['lidar2world']['qx'], json_data[t]['lidar2world']['qy'],
                  json_data[t]['lidar2world']['qz'], json_data[t]['lidar2world']['qw']]
        r = Rotation.from_quat(np.asarray(quat_l))

        lidar2world = np.zeros((4, 4), float)
        lidar2world[0:3, 0:3] = r.as_matrix()
        lidar2world[0:4, 3] = np.asarray([json_data[t]['lidar2world']['x'], json_data[t]['lidar2world']['y'],
                                          json_data[t]['lidar2world']['z'], 1])

        for obj in json_data[t]['tracked_objects']:
            if not id_max is None:
                if obj['id'] not in id_list:
                    if len(id_list) < id_max:
                        id_list.append(obj['id'])
                    else:
                        continue
            center_lidar = np.asarray([obj['center'][0], obj['center'][1], obj['center'][2], 1])
            center_global = np.dot(lidar2world, center_lidar)
            frame_dict[obj['id']] = np.asarray(center_global[0:2].tolist() + obj['vel'][0:2])
        gt_list.append(frame_dict)
    return gt_list


def convert2measure(gt_list):
    measure_list = list()
    for gt_frame in gt_list:
        measure_frame = list()
        for id in gt_frame.keys():
            measure_frame.append(gt_frame[id][0:2])
        measure_list.append(measure_frame)
    return measure_list

if __name__ == '__main__':
    gt_list = load_lidar_result('/media/other/bug_record/velocity/3072526/offline.json')
    convert2measure(gt_list)