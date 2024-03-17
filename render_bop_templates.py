import blenderproc as bproc
import argparse
import os
import numpy as np
import bpy
from PIL import Image
import sys
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('config_path', default="bop_templates_cfg.yaml", help="Path to config file")
args = parser.parse_args()

dirname = os.path.dirname(__file__) 

#read config
with open(os.path.join(dirname, args.config_path), "r") as stream:
    config = yaml.safe_load(stream)


print(config['bop_parent_path'])
bproc.init()

poses = np.load(config['obj_pose'])

if config['poses'] == 'upper':
    cam_poses = np.load(config['cam_pose'])
    poses = poses[cam_poses[:, 2, 3] >= 0]
poses[:, :3, 3] *= config["camera_distance_scaling"]

# Lighting Factor
factor = 1000.0
poses[:, :3, :3] = poses[:, :3, :3] / factor
poses[:, :3, 3] = poses[:, :3, 3] / factor
# load specified bop objects into the scene
print(poses.shape)

if config['bop_dataset_name'] in ['lm', 'ycbv', 'hb', 'tyol', 'itodd', 'tudl', 'icbin']:
    bop_objs = bproc.loader.load_bop_objs(
        bop_dataset_path=os.path.join(config['bop_parent_path'], 
        config['bop_dataset_name']),
        object_model_unit='mm')
elif config['bop_dataset_name'] == 'tless':
    bop_objs = bproc.loader.load_bop_objs(
        bop_dataset_path=os.path.join(config['bop_parent_path'], 
        config['bop_dataset_name']),
        model_type = 'cad',
        object_model_unit='mm')
else:
    raise ValueError('Unknown dataset name')

np.random.seed(0)
for j, obj in enumerate(bop_objs):
    obj.set_shading_mode('auto')
    mat = obj.get_materials()[0]
    if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:

        grey_col = np.random.uniform(0.5, 0.7)   
        mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
            

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path=os.path.join(config['bop_parent_path'], config['bop_dataset_name']))

# get image witdh and height from bproc

img_size = (480, 640)


light = bproc.types.Light()
light.set_type("POINT")
light.set_location([1, -1, 1])
light.set_energy(200)
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([-1, -1, -1])
light.set_energy(200)
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([-1, 0, -1])
light.set_energy(20)
light.set_type("POINT")
light.set_location([1, 0, 1])
light.set_energy(20)



# bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

# set shading
for j, obj in enumerate(bop_objs):
    obj.set_shading_mode('auto')
    obj.hide(True)


cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
    np.eye(4), ["X", "-Y", "-Z"]
)
bproc.camera.add_camera_pose(cam2world)
bproc.renderer.set_max_amount_of_samples(100)

# activate depth rendering
bproc.renderer.enable_distance_output(True)

#get image shape
data = bproc.renderer.render()
depth = bproc.postprocessing.dist2depth(data["distance"])[0]
mask = np.uint8(depth)

black_img = Image.new('RGB', (mask.shape[1], mask.shape[0]))

for obj in bop_objs:
    obj.hide(False)
    name = config['bop_dataset_name']
    obj_data_dir = os.path.join(config['output_dir'], name, f'obj_{obj.get_cp("category_id")}')
    if not os.path.exists(obj_data_dir):
        os.makedirs(obj_data_dir)
        print(f"Directory '{obj_data_dir}' created.")
    else:
        print(f"Directory '{obj_data_dir}' already exists.")
    for idx_frame, obj_pose in enumerate(poses):
        obj.set_local2world_mat(obj_pose)
        data = bproc.renderer.render()
        data.update(bproc.renderer.render_segmap(map_by="class", use_alpha_channel=True))
        # Map distance to depth
        depth = bproc.postprocessing.dist2depth(data["distance"])[0]
        mask = np.uint8((depth < 1000) * 255)
        print(mask.shape)
        mask = Image.fromarray(mask)
        mask.save(os.path.join(obj_data_dir, "mask_{:06d}.png".format(idx_frame)))

        rgb = Image.fromarray(np.uint8(data["colors"][0]))

        img = Image.composite(rgb, black_img, mask)
        img.save(os.path.join(obj_data_dir, "{:06d}.png".format(idx_frame)))
    obj.hide(True)

poses[:, :3, :3] = poses[:, :3, :3] * factor
poses[:, :3, 3] = poses[:, :3, 3] * factor
np.save(os.path.join(config['output_dir'], name, "obj_poses.npy"), poses)
