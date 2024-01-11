import blenderproc as bproc
import argparse
import os
import numpy as np
from PIL import Image
import yaml
import sys


def render(config):

    bproc.init()

    poses = np.load(config['obj_pose'])

    if config['poses'] == 'upper':
        cam_poses = np.load(config['cam_pose'])
        poses = poses[cam_poses[:, 2, 3] >= 0]
    poses[:, :3, 3] *= config["camer_distance_scaling"]
    poses[:, :3, :3] = poses[:, :3, :3] / 1000.0
    poses[:, :3, 3] = poses[:, :3, 3] / 1000.0
    # load specified bop objects into the scene

    print(poses.shape)
    id_mapping = {}
    category_ids = []
    ply_objs = []
    mesh_dir = config['model_dir']

    files = sorted(os.listdir(mesh_dir))
    if len(files) == 0:
        print(f"Warning: no objects loaded from dir: {mesh_dir}")
    else:
        for filename in sorted(os.listdir(mesh_dir)):
            if filename[-3:] != "ply":
                continue
            obj = bproc.loader.load_obj(os.path.join(mesh_dir, filename))[0]
            id = int(filename[4:-4])
            if id >= 255:
                raise Exception("filename needs to be in format obj_xxxxxx.ply, where xxxxxx is 0 padded number smaller than 255")
            if id not in category_ids:
                print(f"Loading object {filename} with id {id}")
                obj.hide(True) 
                category_ids.append(id)
                obj.set_cp("category_id", id)
                id_mapping[filename] = id
                obj.set_shading_mode('auto')
                ply_objs.append(obj)
            else:
                raise Exception("filename needs to be in format obj_xxxxxx.ply, where xxxxxx is 0 padded number smaller than 255")
    if len(ply_objs) == 0:
        raise Exception("No objects loaded from dir: {mesh_dir}")

    if config['set_color'] == "Grey":
        print("Setting color to grey")
        for j, obj in enumerate(ply_objs):
            obj.set_shading_mode('auto')
            mat = obj.get_materials()[0]
            grey_col = np.random.uniform(0.5, 0.7)   
            mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
    elif config['set_color'] == "RandomTexture":
        print("Loading cc textures")
        cc_textures = bproc.loader.load_ccmaterials(config["texture_dir"])
        for j, obj in enumerate(ply_objs):
            random_cc_texture = np.random.choice(cc_textures)
            obj.replace_materials(random_cc_texture)
            mat = obj.get_materials()[0]      
            mat.set_principled_shader_value("Alpha", 1.0)
    elif config['set_color'] == "False":
        pass
    else:
        raise ValueError("Unknown color setting in config set_color")


    bproc.camera.set_intrinsics_from_K_matrix(np.reshape(config["cam"]["K"], (3, 3)), 
                                                config["cam"]["width"], 
                                                config["cam"]["height"])

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
    for j, obj in enumerate(ply_objs):
        obj.set_shading_mode('auto')
        obj.hide(True)


    cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
        np.eye(4), ["X", "-Y", "-Z"]
    )
    bproc.camera.add_camera_pose(cam2world)
    bproc.renderer.set_max_amount_of_samples(100)

    # activate depth rendering
    bproc.renderer.enable_distance_output(True)

    black_img = Image.new('RGB', (config["cam"]["width"], config["cam"]["height"]))
    name = config['dataset_name']
    for obj in ply_objs:
        obj.hide(False)
        
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
            # # Map distance to depth
            depth = bproc.postprocessing.dist2depth(data["distance"])[0]
            mask = np.uint8((depth < 1000) * 255)
            mask = Image.fromarray(mask)
            mask.save(os.path.join(obj_data_dir, "mask_{:06d}.png".format(idx_frame)))

            rgb = Image.fromarray(np.uint8(data["colors"][0]))
            img = Image.composite(rgb, black_img, mask)
            img.save(os.path.join(obj_data_dir, "{:06d}.png".format(idx_frame)))
        obj.hide(True)
    np.save(os.path.join(config['output_dir'], name, "obj_poses.npy"), poses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', default="ply_template_cfg.yml", help="Path to config file")
    args = parser.parse_args()

    dirname = os.path.dirname(__file__)

    #read config
    with open(os.path.join(dirname, args.config_path), "r") as stream:
        config = yaml.safe_load(stream)
    render(config)