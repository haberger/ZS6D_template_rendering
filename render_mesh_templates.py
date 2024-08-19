import blenderproc as bproc
import argparse
import os
import numpy as np
from PIL import Image
import yaml
import json
import bpy

def render(config):

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
    id_mapping = {}
    category_ids = []
    mesh_objs = []
    mesh_dir = config['model_dir']

    files = sorted(os.listdir(mesh_dir))
    object_mapping = {}
    if len(files) == 0:
        print(f"Warning: no objects loaded from dir: {mesh_dir}")
    else:
        idx = 1
        for filename in sorted(os.listdir(mesh_dir)):
            if config['mesh_type'] == "ply":
                if filename[-3:] != "ply":
                    continue
                if filename[:-4] in category_ids:
                    raise Exception("duplicate object names")
                obj = bproc.loader.load_obj(os.path.join(mesh_dir, filename))[0]
                print(f"Loading object {filename}")
                obj.hide(True) 
                category_ids.append(filename[:-4])
                obj.set_cp("category_id", filename[:-4])
                obj.set_shading_mode('auto')
                mesh_objs.append([obj])
                object_mapping[filename[:-4]] = str(idx)
                idx += 1
            if config['mesh_type'] == "obj":
                if filename[-3:] != "obj":
                    continue
                if filename[:-4] in category_ids:
                    raise Exception("duplicate object names")
                category_ids.append(filename[:-4])
                obj = bproc.loader.load_obj(os.path.join(mesh_dir, filename))[0]
                print(f"Loading object {filename}")
                obj.set_cp("category_id", filename[:-4])
                obj.hide(True) 
                obj.set_shading_mode('auto')
                mesh_objs.append([obj])
                object_mapping[filename[:-4]] = str(idx)
                idx += 1
            if config['mesh_type'] == "blend":
                obj_parts = []
                if filename[-5:] != "blend":
                    continue
                objs = bproc.loader.load_blend(os.path.join(mesh_dir, filename))
                for obj in objs:
                    if obj.get_name() == filename[:-6]:
                        continue
                    # category_ids.append(obj.get_cp("category_id"))
                    obj.set_cp("category_id", filename[:-6])
                    obj.hide(True) 
                    obj.set_shading_mode('auto')
                    obj_parts.append(obj)
                    object_mapping[filename[:-6]] = str(idx)
                    idx += 1
                mesh_objs.append(obj_parts)

    if len(mesh_objs) == 0:
        raise Exception("No objects loaded from dir: {mesh_dir}")
    
    name = config['dataset_name']
    models_dir = os.path.join(config['output_dir'], name, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Directory '{models_dir}' created.")
    else:
        print(f"Directory '{models_dir}' already exists.")

    for obj_parts in mesh_objs:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in obj_parts:
            obj.blender_obj.select_set(True)
        bpy.ops.export_mesh.ply(filepath=os.path.join(models_dir, f"obj_{int(object_mapping[obj_parts[0].get_cp('category_id')]):06d}.ply"), use_selection=True, use_colors=True)
    
    for obj_parts in mesh_objs:
        for obj in obj_parts:
            obj.blender_obj.scale *= config["mesh_scale"]
            obj.persist_transformation_into_mesh()

    if config['set_color'] == "Grey":
        print("Setting color to grey")
        for obj_parts in mesh_objs:
            for j, obj in enumerate(obj_parts):
                obj.set_shading_mode('auto')
                mat = obj.get_materials()[0]
                grey_col = np.random.uniform(0.5, 0.7)   
                mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
    elif config['set_color'] == "RandomTexture":
        print("Loading cc textures")
        cc_textures = bproc.loader.load_ccmaterials(config["texture_dir"])
        for obj_parts in mesh_objs:
            for j, obj in enumerate(obj_parts):
                random_cc_texture = np.random.choice(cc_textures)
                obj.replace_materials(random_cc_texture)
                mat = obj.get_materials()[0]      
                mat.set_principled_shader_value("Alpha", 1.0)
    elif config['set_color'] == "False":
        pass
        # mat = obj.get_materials()[0]
        # mat.set_principled_shader_value("Roughness", 0.5)
        # mat.set_principled_shader_value("Specular", 0.0)
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
    for obj_parts in mesh_objs:
        for j, obj in enumerate(obj_parts):
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
    
    for obj_parts in mesh_objs:
        obj_data_dir = os.path.join(config['output_dir'], name, f'obj_{object_mapping[obj_parts[0].get_cp("category_id")]}')

        if not os.path.exists(obj_data_dir):
            os.makedirs(obj_data_dir)
            print(f"Directory '{obj_data_dir}' created.")
        else:
            print(f"Directory '{obj_data_dir}' already exists.")

        for idx_frame, obj_pose in enumerate(poses):
            for obj in obj_parts:
                obj.hide(False)
                obj.set_local2world_mat(obj_pose)
            # break
            data = bproc.renderer.render()
            #data.update(bproc.renderer.render_segmap(map_by="class", use_alpha_channel=True))
            # # Map distance to depth
            depth = bproc.postprocessing.dist2depth(data["distance"])[0]
            mask = np.uint8((depth < 1000) * 255)
            mask = Image.fromarray(mask)
            mask.save(os.path.join(obj_data_dir, "mask_{:06d}.png".format(idx_frame)))

            rgb = Image.fromarray(np.uint8(data["colors"][0]))
            img = Image.composite(rgb, black_img, mask)
            img.save(os.path.join(obj_data_dir, "{:06d}.png".format(idx_frame)))
        for obj in obj_parts:
            obj.hide(True)
    poses[:, :3, :3] = poses[:, :3, :3] * factor
    poses[:, :3, 3] = poses[:, :3, 3] * factor
    np.save(os.path.join(config['output_dir'], name, "obj_poses.npy"), poses)


    template_gt_config = {}
    template_gt_config["path_templates_folder"] = os.path.join("./templates/", config['dataset_name'])
    template_gt_config["path_template_poses"] = os.path.join("./templates/", config['dataset_name'], "obj_poses.npy")
    template_gt_config["path_object_models_folder"] = os.path.join("./templates/", config['dataset_name'], "models")
    template_gt_config["obj_models_scale"] = 0.001
    template_gt_config["path_output_templates_and_descs_folder"] = os.path.join("./templates/", config['dataset_name'] + "_desc")
    template_gt_config["path_output_models_xyz"] = os.path.join("./templates/", config['dataset_name'] + "_desc", "models_xyz")
    template_gt_config["cam_K"] = config["cam"]["K"]
    template_gt_config["template_resolution"] = [config["cam"]["width"], config["cam"]["height"]]
    template_gt_config["output_template_gt_file"] = os.path.join("./gts/template_gts/", config['dataset_name'] + "_template_gt.json")
    # write config file as json
    with open(os.path.join(config['output_dir_zs6d_configs'], "template_gt_preparation_configs", f"cfg_template_gt_generation_{config['dataset_name']}.json"), "w") as f:
        json.dump(template_gt_config, f, indent=2)

    zs6d_config = {}
    zs6d_config["templates_gt_path"] = template_gt_config["output_template_gt_file"]
    zs6d_config["template_subset"] = 1
    zs6d_config["norm_factor_path"] = os.path.join("./templates/", config['dataset_name'] + "_desc", "models_xyz", "norm_factor.json")
    zs6d_config["image_resolution"] = [config["cam"]["height"], config["cam"]["width"]]
    zs6d_config["cam_K"] = [538.391033533567, 0.0, 315.3074696331638, 0.0, 538.085452058436, 233.0483557773859, 0.0, 0.0, 1.0]
    zs6d_config["object_mapping"] = object_mapping

    # write config file as json
    with open(os.path.join(config['output_dir_zs6d_configs'], "bop_eval_configs", f"cfg_{config['dataset_name']}_inference_bop.json"), "w") as f:
        json.dump(zs6d_config, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="mesh_templates_cfg.yaml", help="Path to config file")
    args = parser.parse_args()

    dirname = os.path.dirname(__file__)

    #read config
    with open(os.path.join(dirname, args.config_path), "r") as stream:
        config = yaml.safe_load(stream)
    render(config)
