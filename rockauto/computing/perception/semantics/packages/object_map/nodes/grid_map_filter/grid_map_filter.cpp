/*
 *  Copyright (c) 2018, Nagoya University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither the name of Autoware nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************/

#include "grid_map_filter.h"

namespace object_map
{

// Constructor
	GridMapFilter::GridMapFilter() :
			private_node_handle_("~")
	{
		InitializeRosIo();
		LoadRoadAreasFromVectorMap(private_node_handle_, area_points_);
	}

	void GridMapFilter::InitializeRosIo()
	{
		private_node_handle_.param<std::string>("map_frame", map_frame_, "map");
		private_node_handle_.param<std::string>("map_topic", map_topic_, "/realtime_cost_map");
		private_node_handle_.param<double>("dist_transform_distance", dist_transform_distance_, 3.0);
		private_node_handle_.param<bool>("use_dist_transform", use_dist_transform_, false);
		private_node_handle_.param<bool>("use_wayarea", use_wayarea_, false);
		private_node_handle_.param<bool>("use_fill_circle", use_fill_circle_, false);
		private_node_handle_.param<int>("fill_circle_cost_threshold", fill_circle_cost_thresh_, 20);
		private_node_handle_.param<double>("circle_radius", circle_radius_, 1.7);

		occupancy_grid_sub_ = nh_.subscribe<nav_msgs::OccupancyGrid>(map_topic_, 10,
		                                                             &GridMapFilter::OccupancyGridCallback, this);

		grid_map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("filtered_grid_map", 1, true);

	}


	void GridMapFilter::Run()
	{
		ros::spin();
	}

	void GridMapFilter::OccupancyGridCallback(const nav_msgs::OccupancyGridConstPtr &in_message)
	{
		// timer start
		//auto start = std::chrono::system_clock::now();

		std::string original_layer = "original";

		grid_map::GridMap map({original_layer, "distance_transform", "wayarea", "dist_wayarea", "circle"});

		//store costmap map_topic_ into the original layer
		nav_msgs::OccupancyGrid base_link_message = *in_message;  
    base_link_message.header.frame_id = "base_link";  
    base_link_message.info.origin.position.x += 1.2;//1.2???base_link???velodyne?????????x??????????????????
    base_link_message.info.origin.position.z = 0;
    grid_map::GridMapRosConverter::fromOccupancyGrid(base_link_message, "original", map);

		// apply distance transform to OccupancyGrid
		if (use_dist_transform_)
		{
			CreateDistanceTransformLayer(map, original_layer);
		}

		// fill polygon
		if (!area_points_.empty() && use_wayarea_)
		{
			FillPolygonAreas(map, area_points_, grid_road_layer_, OCCUPANCY_NO_ROAD, OCCUPANCY_ROAD, grid_min_value_,
			                 grid_max_value_, map.getFrameId(), map_frame_, tf_listener_);

			map["dist_wayarea"] = map["distance_transform"] + map["wayarea"];
		}

		// fill circle
		if (use_fill_circle_)
		{
			int cost_threshold = fill_circle_cost_thresh_;
			// convert to cv image size
			int radius = circle_radius_ / map.getResolution();
			DrawCirclesInLayer(map, original_layer, cost_threshold, radius);
		}

		// publish grid map as ROS message
		PublishGridMap(map, grid_map_pub_);

		// timer end
		//auto end = std::chrono::system_clock::now();
		//auto usec = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		//std::cout << "time: " << usec / 1000.0 << " [msec]" << std::endl;
	}

	void GridMapFilter::CreateDistanceTransformLayer(grid_map::GridMap &out_grid_map, const std::string &in_layer)
	{
		cv::Mat original_image;
		if (!out_grid_map.exists(in_layer))
		{
			ROS_INFO("%s layer not yet available", in_layer.c_str());
			return;
		}
		grid_map::GridMapCvConverter::toImage<unsigned char, 1>(out_grid_map,
		                                                        in_layer,
		                                                        CV_8UC1,
		                                                        original_image);

		cv::Mat binary_image;
		cv::threshold(original_image,
		              binary_image,
		              fill_circle_cost_thresh_,
		              grid_max_value_,
		              cv::THRESH_BINARY_INV);

		// distance transform method
		// 3: fast
		// 5: slow but accurate
		cv::Mat dt_image;
		cv::distanceTransform(binary_image, dt_image, CV_DIST_L2, 5);

		// Convert to int...
		cv::Mat dt_int_image(dt_image.size(), CV_8UC1);
		cv::Mat dt_int_inv_image(dt_image.size(), CV_8UC1);

		// max distance for cost propagation
		double max_dist = dist_transform_distance_; // meter
		double resolution = out_grid_map.getResolution();

		for (int y = 0; y < dt_image.rows; y++)
		{
			for (int x = 0; x < dt_image.cols; x++)
			{
				// actual distance [meter]
				double dist = dt_image.at<float>(y, x) * resolution;
				if (dist > max_dist)
					dist = max_dist;

				// Make value range 0 ~ 255
				int round_dist = dist / max_dist * grid_max_value_;
				int inv_round_dist = grid_max_value_ - round_dist;

				dt_int_image.at<unsigned char>(y, x) = round_dist;
				dt_int_inv_image.at<unsigned char>(y, x) = inv_round_dist;
			}
		}

		// convert to ROS msg
		grid_map::GridMapCvConverter::addLayerFromImage<unsigned char, 1>(dt_int_inv_image,
		                                                                  "distance_transform",
		                                                                  out_grid_map,
		                                                                  grid_min_value_,
		                                                                  grid_max_value_);
	}

	void GridMapFilter::DrawCirclesInLayer(grid_map::GridMap &out_gridmap,
	                                       const std::string &in_layer_name,
	                                       double in_draw_threshold,
	                                       double in_radius)
	{
		cv::Mat original_image;

		grid_map::GridMapCvConverter::toImage<unsigned char, 1>(out_gridmap,
		                                                        in_layer_name,
		                                                        CV_8UC1,
		                                                        costmap_min_,
		                                                        costmap_max_,
		                                                        original_image);

		cv::Mat filled_image = original_image.clone();

		for (int y = 0; y < original_image.rows; y++)
		{
			for (int x = 0; x < original_image.cols; x++)
			{
				// uchar -> int
				int data = original_image.at<unsigned char>(y, x);

				if (data > fill_circle_cost_thresh_)
				{
					cv::circle(filled_image, cv::Point(x, y), in_radius, cv::Scalar(OCCUPANCY_CIRCLE), -1, CV_AA);
				}
			}
		}
		// convert to ROS msg
		grid_map::GridMapCvConverter::addLayerFromImage<unsigned char, 1>(filled_image,
		                                                                  "circle",
		                                                                  out_gridmap,
		                                                                  grid_min_value_,
		                                                                  grid_max_value_);
	}


}  // namespace object_map
