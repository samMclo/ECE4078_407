import numpy as np
import json
import os
import ast
import cv2
from YOLO.detector import Detector
from copy import deepcopy

def parse_user_map(fname : str) -> dict:
    with open(fname, 'r') as f:
        usr_dict = ast.literal_eval(f.read())
        aruco_dict = {}
        for (i, tag) in enumerate(usr_dict["taglist"]):
            aruco_dict[tag] = np.reshape([usr_dict["map"][0][i],usr_dict["map"][1][i]], (2,1))
    return aruco_dict

def create_aruco_map(est):
    aruco_dict = {}
    for i in range(1, 11):
        dict_number = {f'aruco{i}_0': {'x': float(np.squeeze(est[i][0])), 'y': float(np.squeeze(est[i][1]))}}
        aruco_dict = {**aruco_dict, **dict_number}
    return aruco_dict

def parse_object_map(fname):
    with open(fname, 'r') as fd:
        usr_dict = json.load(fd)
        return usr_dict
    
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import argparse
    
    parser = argparse.ArgumentParser('Matching the estimated map and the true map')
    parser.add_argument('--true-map', type=str, default='M4_prac_map_full.txt')
    parser.add_argument('--slam-est', type=str, default='lab_output/slam.txt')
    parser.add_argument('--target-est', type=str, default='lab_output/targets.txt')
    parser.add_argument('--slam-only', action='store_true')
    parser.add_argument('--target-only', action='store_true')
    args, _ = parser.parse_known_args()

    aruco_est = parse_user_map(args.slam_est)
    objects_est = parse_object_map(args.target_est)
    aruco_map = create_aruco_map(aruco_est)
    conc = {**aruco_map, **objects_est}

    with open(f'{script_dir}/Map.txt', 'w') as fo:
        json.dump(conc, fo, indent=4)

    print('Map saved!')