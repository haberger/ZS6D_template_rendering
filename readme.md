## Template rendering for ZS6D: Zero-shot 6D Object Pose Estimation using Vision Transformers

1. Set up [conda](https://docs.anaconda.com/miniconda/) environment:

```conda env create -f env.yml```

```conda activate render_temp```

### Rendering with bop datasets

2. set up the config file

```bop_templates_cfg.yaml```

3. run script to render templates

```blenderproc run render_bop_templates.py bop_templates_cfg.yaml```

### Rendering with custom mesh files (ply/obj)

2. set up the config file (mesh files are expected to be in mm)

```mesh_templates_cfg.yaml```

3. run script to render templates

```blenderproc run render_mesh_templates.py mesh_templates_cfg.yaml```

### Pose generation

If more poses than 2562 are wanted, further poses can be calculated with the script below
In line 133 for the script increase the levels, higher means more poses

```blenderproc run generate_poses```


