import os
import bpy

# put the location to the folder where the plys are located here in this fashion
path_to_obj_dir = '/mnt/fast/code/point-e/point_e/examples/'

# get list of all files in directory
file_list = sorted(os.listdir(path_to_obj_dir))

# get a list of files ending in 'ply'
ply_list = [item for item in file_list if item[-3:] == 'ply']

# loop through the strings in ply_list and add the files to the scene
for item in ply_list:
    # full path and file name
    path_to_file = os.path.join(path_to_obj_dir, item)
    # import ply into the scene
    bpy.ops.import_mesh.ply(filepath = path_to_file)
    # change suffix to ".obj"
    portion = os.path.splitext(item)
    newname = portion[0]+".obj"
    # new full path and file name
    path_to_file2 = os.path.join(path_to_obj_dir, newname)
    # export current scene to an obj file
    bpy.ops.export_scene.obj(filepath = path_to_file2)
    # delete the object from the scene
    bpy.ops.object.delete()
