from collections import defaultdict
import gzip
import os
import habitat
from habitat.utils.visualizations import maps
from habitat_sim import Simulator as Sim
import json
import habitat_sim
import numpy as np
from habitat.tasks.nav.nav import TopDownMap
from omegaconf import OmegaConf
import quaternion
from common.embodied_utils.simulator import HabitatSimulator
from frontier_utils import convert_meters_to_pixel, detect_frontier_waypoints, get_closest_waypoint, get_polar_angle, map_coors_to_pixel, pixel_to_map_coors, reveal_fog_of_war
from sim_utils import get_simulator
import cv2
from data_utils import PQ3DModel
import random
import sys
from habitat_sim.utils.common import quat_from_coeffs

# hyperparameter
eqa_data_path = '/mnt/fillipo/ML/xilin/a-eqa-184.json'
hm3d_data_base_path = "/mnt/fillipo/ML/zhuofan/data/scene_datasets/hm3d/val"
pq3d_stage1_path = "/home/zhuziyu/work/saved_models/embodied-pq3d-final/stage1-pretrain-all/"
pq3d_stage2_path = "/home/zhuziyu/work/saved_models/embodied-pq3d-final-stage2/stage2-pretrain-all/"
output_dir = "/mnt/fillipo/zhuziyu/embodied_saved_data/a-eqa-184"
output_json_path = os.path.join(output_dir, "eqa_traj.json")
enable_visualization = False

# load eqa data
with open(eqa_data_path, 'r') as f:
    eqa_data = json.load(f)

# record result
if os.path.exists(output_json_path):
    with open(output_json_path, 'r') as f:
        result = json.load(f)   
else:
    result = []
# filter out data that has been processed
eqa_data = [data for data in eqa_data if data['question_id'] not in [r['question_id'] for r in result]]

# load pq3d model
pq3d_model = PQ3DModel(pq3d_stage1_path, pq3d_stage2_path)

for data in eqa_data:
    scene_id = data['scene_id']
    question_id = data['question_id']
    start_position = data['start_position']
    start_rotation = data['start_rotation']
    question = data['question']
    
    # reset pq3d
    pq3d_model.reset()
    
    # get simulator
    sim_settings = OmegaConf.load('configs/habitat/eqa_sim_config.yaml')
    goat_agent_setting = OmegaConf.load('configs/habitat/eqa_agent_config.yaml')
    clean_scene_id = scene_id.split("-")[-1]
    scene_path = os.path.join(hm3d_data_base_path, scene_id, f"{clean_scene_id}.basis.glb")
    sim_settings['scene'] = scene_path
    abstract_sim = HabitatSimulator(sim_settings, goat_agent_setting)
    sim = abstract_sim.simulator
    agent = abstract_sim.agent
    agent_state = habitat_sim.AgentState()
    agent_state.position = start_position
    agent_state.rotation = quaternion.from_float_array(start_rotation)
    agent.set_state(agent_state)
    path_finder = sim.pathfinder
    
    # get frontier param
    map_resolution = 512
    top_down_map = maps.get_topdown_map_from_sim(sim, map_resolution=map_resolution, draw_border=False)
    fog_of_war_mask = np.zeros_like(top_down_map)
    area_thres_in_pixels = convert_meters_to_pixel(9, map_resolution, sim)
    visibility_dist_in_pixels = convert_meters_to_pixel(2, map_resolution, sim)
    
    # start decision
    total_steps = 0
    global_color_list = []
    goto_color_list = []
    goto_depth_list = []
    goto_agent_state_list = []
    visited_frontier_set = set()
    
    while total_steps < 500:
        color_list = []
        depth_list = []
        agent_state_list = []
        
        if len(goto_color_list) > 6:
            goto_color_list = [goto_color_list[i] for i in range(0, len(goto_color_list), len(goto_color_list) // 6)][:6]
            goto_depth_list = [goto_depth_list[i] for i in range(0, len(goto_depth_list), len(goto_depth_list) // 6)][:6]
            goto_agent_state_list = [goto_agent_state_list[i] for i in range(0, len(goto_agent_state_list), len(goto_agent_state_list) // 6)][:6]
        
        color_list.extend(goto_color_list)
        depth_list.extend(goto_depth_list)
        agent_state_list.extend(goto_agent_state_list)
        
        action_list = ['turn_left'] * 12
        for action in action_list:
            observations = sim.step(action=action)
            color = observations['color_sensor'][:, :, :3]
            cv2.imwrite('color.png', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            color_list.append(color)
            global_color_list.append(color)
            depth = observations['depth_sensor'][:, :]
            depth_list.append(depth)
            agent_state = agent.get_state()
            agent_state_list.append(agent_state)
            fog_of_war_mask = reveal_fog_of_war(top_down_map=top_down_map, current_fog_of_war_mask=fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=42, max_line_len=visibility_dist_in_pixels, enable_debug_visualization=enable_visualization)
            total_steps += 1
        
        agent_state = agent.get_state()
        frontier_waypoints = detect_frontier_waypoints(top_down_map, fog_of_war_mask, area_thres_in_pixels, xy=map_coors_to_pixel(agent_state.position, top_down_map, sim)[::-1], enable_visualization=enable_visualization)
        if len(frontier_waypoints) == 0:
            frontier_waypoints = []
        else:
            frontier_waypoints = frontier_waypoints[:, ::-1]
            frontier_waypoints = pixel_to_map_coors(frontier_waypoints, agent_state.position, top_down_map, sim)
        
        frontier_waypoints = [waypoint for waypoint in frontier_waypoints if tuple(np.round(waypoint, 1)) not in visited_frontier_set]
        
        try:
            target_position, is_final_decision = pq3d_model.decision(color_list, depth_list, agent_state_list, frontier_waypoints, question, total_steps)
        except Exception as e:
            print(f"Error in decision making, question_id: {question_id}, scene_id: {scene_id}, {e}")
            sys.exit(1)
        
        if not is_final_decision:
            visited_frontier_set.add(tuple(np.round(target_position, 1)))
        
        agent_island = path_finder.get_island(agent_state.position)
        target_on_navmesh = path_finder.snap_point(point=target_position, island_index=agent_island)
        follower = habitat_sim.GreedyGeodesicFollower(path_finder, agent, forward_key="move_forward", left_key="turn_left", right_key="turn_right")
        
        try:
            action_list = follower.find_path(target_on_navmesh)
        except:
            print("Error in finding path")
            break
        
        goto_color_list = []
        goto_depth_list = []
        goto_agent_state_list = []
        
        for action in action_list:
            if action:
                observations = sim.step(action=action)
                global_color_list.append(observations['color_sensor'][:, :, :3])
                agent_state = agent.get_state()
                color = observations['color_sensor'][:, :, :3]
                goto_color_list.append(color)
                depth = observations['depth_sensor'][:, :]
                goto_depth_list.append(depth)
                goto_agent_state_list.append(agent_state)
                fog_of_war_mask = reveal_fog_of_war(top_down_map=top_down_map, current_fog_of_war_mask=fog_of_war_mask, current_point=map_coors_to_pixel(agent_state.position, top_down_map, sim), current_angle=get_polar_angle(agent_state), fov=42, max_line_len=visibility_dist_in_pixels, enable_debug_visualization=enable_visualization)
                total_steps += 1
        
        if is_final_decision:
            break
    
    height, width, layers = global_color_list[0].shape
    video_path = os.path.join(output_dir, f'{question_id}.avi')
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 2, (width, height))
    for color_frame in global_color_list:
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
        video.write(color_frame)
    video.release()
    
    result.append({'scene_id': scene_id, 'question_id': question_id, 'video_length': len(global_color_list)})
    with open(output_json_path, 'w') as f:
        json.dump(result, f)


