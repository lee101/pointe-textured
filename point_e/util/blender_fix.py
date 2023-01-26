
import bpy
# delete starting scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

bpy.ops.wm.obj_import(filepath="/mnt/fast/code/point-e/point_e/examples/corgi_mesh_3.obj", directory="/mnt/fast/code/point-e/point_e/examples/", files=[{"name":"corgi_mesh_2.obj", "name":"corgi_mesh_2.obj"}])
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=True)
# save normals
bpy.ops.object.editmode_toggle()
# save normals

# bpy.context.space_data.params.directory = "/home/lee/test/"
bpy.ops.export_scene.obj(filepath="/home/lee/test/corgi_mesh_3.obj", check_existing=False, axis_forward='Z', axis_up='Y', filter_glob="*.obj;*.mtl", use_selection=False, use_animation=False, use_mesh_modifiers=True, use_edges=True, use_smooth_groups=False, use_smooth_groups_bitflags=False, use_normals=True, use_uvs=True, use_materials=False, use_triangles=True, use_nurbs=False, use_vertex_groups=False, use_blen_objects=True, group_by_object=False, group_by_material=False, keep_vertex_order=False, global_scale=1, path_mode='AUTO')

# embed_textures=False batch_mode='OFF' use_batch_own_dir=True,
# use_metadata=True
