# Point·E
This is an experimental fork of point E for creating and repairing meshes with AI as well as traditional algorithms in meshlab and blender.

Scripts extend point e to generate a mesh with textures from a point cloud suitable to be dragg/dropped into unity!

relies on blender and meshlab for mesh repair, which need to be installed

For AI including text to speech, speech to text, and AI text generation checkout https://text-generator.io

For a data and dashbording AI assistant checkout https://askFelix.ai


There are also some experiments with mesh alignment with iterative closest point for merging meshes for better quality / resolution.
####


```shell
pip install -r requirements.txt
PYTHONPATH=$PYTHONPATH:$(pwd) python generate_obj.py --image example_data/toytrain.png --save_file_name results/toytrain13.ply 
```

