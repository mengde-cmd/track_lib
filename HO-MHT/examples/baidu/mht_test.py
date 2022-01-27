#coding utf-8
import os
import sys

import matplotlib

"""Add this file as the main func."""
example_path = os.path.dirname(os.path.dirname(__file__))
# print(example_path)
# print(os.path.pardir)
# print(os.path.join(example_path, os.path.pardir))
sys.path.insert(0, os.path.join(example_path, os.path.pardir))
""" Start Main """
""" Import Modules """
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from mht.tracker import (Tracker)
from mht.tracker.tracker import *
# Application implementation
from examples.linear_model.cv_target import (TargetPosition_CV2D)

from mht.scan_volume import (CartesianVolume)

from mht.utils import generation
from mht.utils import gaussian
from mht.utils import plot
from mht.utils import OSPA
import time
from tqdm import tqdm

# add by mengde
from load_lidar_result import load_lidar_result
from load_lidar_result import convert2measure

json_file_path = '/media/other/bug_record/velocity/3072526/offline.json'
gt_list = load_lidar_result(json_file_path, 4)
measure_list = convert2measure(gt_list)

""" Init """
targetmodel = TargetPosition_CV2D
measmodel = targetmodel.measure()#Here we use the classMethod to get the measmodel

#init_lambda is used to get the object birth s.t. Possion(init_lambda) in <generation.random_ground_truth>
init_lambda = 0.5
#量测边界为ranges  检测概率P_D=0.9  生成杂波的泊松分布的系数为clutter_lambda=1.0
volume = CartesianVolume(
    ranges = np.array([[-1200.0, 1200.0], [-1200.0, 1200.0]]),
    P_D=0.95,
    clutter_lambda = 10,
    init_lambda = init_lambda
)
calc_time = list()#Record the total tracking time

show_plots = True

dt = 0.1#sampling time
nof_rounds = 1#Monte Carlo

""" M.C. """
for i in range(nof_rounds):
    """Get tracker,ground_truth and measurements"""
    tracker = Tracker(
        max_nof_hyps = 30,
        hyp_weight_threshold = np.log(0.05),
    )#跟踪器:_M=10,即m-best为10; 假设的权重阈值为ln(0.05)


    ground_truth = gt_list
    measurements = measure_list

    gt_list = gt_list[0:20]
    measurements = measurements[0:20]

    end_time_eachMC = len(measure_list)

    #record the history of Estim.=>list({trid: Density})
    estimations = list()
    """Start Tracking"""
    tic = time.time()
    for t, detections in enumerate(tqdm(measurements)):
        t_now = dt * t#get the cur. time
        # for key in tracker.tracks.keys():
            # if tracker.tracks[key].lhyps_size() > 1:
                # print(str(t) + " " + str(key) + " " + str(tracker.tracks[key].lhyps_size()))
        estimations.append(tracker.process(detections, volume, targetmodel, t_now))#Track and record

    aa = tracker.estimate_all_history()

    """End Tracking"""
    # print(estimations)
    calc_time.append(time.time()-tic)#记录跟踪的时间
    # get the track state=>{trid:xk}
    track_states = [
        {trid: density.x for trid, density in est.items()}
        for est in estimations
    ]


    # """OPSA metric"""
    # assert(len(track_states)==len(ground_truth))
    # runtimes = len(track_states)
    # ## OSPA_metric = OSPA.OSPA()
    # rec_OPSA = list()
    # for i in range(runtimes):
    #     score = OSPA.OSPA(ground_truth[i],track_states[i],1,2).solver()
    #     rec_OPSA.append(score)

    if show_plots:

        gt = list()
        mt = list()
        for frame in track_states:
            if len(frame) <= 0:
                mt.append(0)
            else:
                keys = sorted(frame.keys())
                mt.append(np.sqrt(frame[keys[0]][2]**2 + frame[keys[0]][3]**2))

        i = 0
        for frame in ground_truth:
            if len(frame) <= 0:
                gt.append(0)
            else:
                if i > 0 and 86 in ground_truth[i-1].keys() and False:
                    keys = sorted(frame.keys())
                    gt.append(np.sqrt((frame[keys[0]][0] - ground_truth[i-1][keys[0]][0]) ** 2 + (frame[keys[0]][1] - ground_truth[i-1][keys[0]][1]) ** 2) / 0.1)
                else:
                    keys = sorted(frame.keys())
                    gt.append(np.sqrt(frame[keys[0]][2] ** 2 + frame[keys[0]][3] ** 2))
            i = i + 1

        # print(record)
        # plt.hist(record)
        # plt.plot(mt)
        # plt.plot(gt)
        # plt.show()

        p = plot.Plotter(to_plot_coordinates=measmodel.h, num=1)
        # p.trajectory_2d(ground_truth, linestyle='-')
        # p.measurements_2d(measurements, marker='.', color='k')
        p.trajectory_2d(estimations, linestyle='-')
        # p.covariances_2d(estimations, measmodel, edgecolor='k', linewidth=1)

        p2 = plot.Plotter(to_plot_coordinates=measmodel.h, num=4)
        p2.trajectory_2d(ground_truth, linestyle='-')
        p2.show()
        p.show()


    # if show_plots:
    #     #show_plot = False
    #     q = plot.Plotter(to_plot_coordinates=measmodel.h, num=2)
    #     #q.trajectory_2d(ground_truth, linestyle='-')
    #     q.measurements_2d(measurements, marker='.', color='k')
    #     #q.trajectory_2d(estimations, linestyle='--')
    #     #q.covariances_2d(estimations, measmodel, edgecolor='k', linewidth=1)
    #
    #     p = plot.Plotter(to_plot_coordinates=measmodel.h, num=1)
    #     p.trajectory_2d(ground_truth, linestyle='-')
    #     p.measurements_2d(measurements, marker='.', color='k')
    #     p.trajectory_2d(estimations, linestyle='dotted')
    #     p.covariances_2d(estimations, measmodel, edgecolor='k', linewidth=1)
    #     #p.legend()
    #     # paint OSPA
    #     p_o = plt.figure(num=3)
    #     p_o.gca().plot(np.arange(0,end_time_eachMC,dt),rec_OPSA)
    #     print(rec_OPSA)
    #
    #     p2 = plot.Plotter(to_plot_coordinates=measmodel.h, num=4)
    #     p2.trajectory_2d(ground_truth, linestyle='-')
    #     p2.show()
    #
    #     q.show()
    #     p.show()
    #     p_o.show()



# print(time.time()-tic)
