import pymeshlab
import pyvista as pv
import numpy as np
from loguru import logger

def pyvista_mesh_to_meshlab_mesh(mesh):
    # get points
    points = v.points
    # get faces
    faces = v.faces
    # get normals
    normals = v.point_arrays['Normals']
    # get colors
    colors = v.point_arrays['Colors']
    # save mesh
    return colors, normals, points, faces

def load_mesh_into_meshlab(file_path):
    # load mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_path)
    return ms


def mesh_repair(ms):
    # close holes
    ms.repair_non_manifold_edges()

    # ms.close_holes()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    # ms.remove_isolated_pieces()
    #ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.Percentage(0.1))
    max_size_to_be_closed=600
#    ms.meshing_close_holes(maxholesize=max_size_to_be_closed)
    logger.info(f"Closing holes with max size {max_size_to_be_closed}")
    ms.meshing_close_holes(maxholesize=max_size_to_be_closed)
    logger.info(f"repair non manifoldednes")

    ms.repair_non_manifold_edges()
    logger.info(f"Closing holes with max size {max_size_to_be_closed}")

    ms.meshing_close_holes(maxholesize=max_size_to_be_closed)
    logger.info(f"repair - delete self intersects")
    # ms = remove_self_intersections(ms)
    ms.repair_non_manifold_edges()
    logger.info(f"Closing holes with max size 900")

    ms.meshing_close_holes(maxholesize=900)

    #ms.repair_non_manifold_edges() # todo tihs can create holes?
    # logger.info(f"subdiv")

    # subdivide catmul clark TOOD this crashes meshlab
    # ms.meshing_surface_subdivision_catmull_clark()
    # ms.simplification_edge_collapse_decimation(targetfacenum=100000)
    # smooth
    # ms.filter_laplacian_smooth() #todo

    # ms.remove_degenerated_faces()
    # ms.remove_degenerated_triangles()
    # ms.remove_non_manifold_vertices()
    # ms.remove_non_manifold_faces()
    return ms

def mesh_simplify_for_marching_cube_meshes(ms):
    ms.simplification_edge_collapse_for_marching_cube_meshes()
    return ms

def remove_self_intersections(ms):
    # remove self intersections
    # ms.filter_remove_self_intersections()
    ms.select_self_intersecting_faces()
    ms.delete_selected_faces()
    ms.select_problematic_faces()
    ms.delete_selected_faces()
    return ms

def meshlab_process(colors, normals, points, faces, save_path):
    ms = load_mesh_into_meshlab(save_path)

    #ms.generate_surface_reconstruction_screened_poisson(depth=5, scale=1.1)
    #ms.generate_surface_reconstruction_ball_pivoting()
    ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
    # close holes in mesh
    ms = mesh_repair(ms)


    # generate normals
    # create texture using UV map and vertex colors
    name = save_path.split("/")[-1].split('.')[0]
    ms.compute_texmap_from_color(
        textname=name)  # textname will be filename of a png, should not be a full path
    # texture file won't be saved until you save the mesh
    # ms.save_current_mesh(save_path)
    # save as obj
    ms.save_current_mesh(save_path.replace(".ply", ".obj"))

def add_color_save_meshlab(colors, normals, points, faces, save_path):
    # open obj file in meshlab
    # convert to numpy
    colors = np.array(colors)
    # add alpha channel
    colors = np.concatenate((colors, np.ones((colors.shape[0], 1))), axis=1).astype(np.float64)
    normals = np.array(normals).astype(np.float64)
    points = np.array(points).astype(np.float64)
    faces = np.array(faces).astype(np.int32)
    m = pymeshlab.Mesh(points, v_normals_matrix=normals,
                       # face_list_of_indicies=faces,
                       v_color_matrix=colors / 255.0)  # color is N x 4 with alpha info
    # # generate normals
    # # m.compute_vertex_normals()
    # # m.compute_per_vertex_normals()
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "pc_scan")
    ms.generate_surface_reconstruction_screened_poisson(depth=5, scale=1.1)
    # not familiar with the crop API, but I'm sure it's doable
    # now we generate UV map; there are a couple options here but this is a naive way
    ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
    # generate normals
    # create texture using UV map and vertex colors
    ms.compute_texmap_from_color(
        textname=f"my_texture_name")  # textname will be filename of a png, should not be a full path
    # texture file won't be saved until you save the mesh
    # ms.save_current_mesh(save_path)
    # save as obj
    ms.save_current_mesh(save_path.replace(".ply", ".obj"))


if __name__ == "__main__":
    import numpy as np

    # colors = np.random.randint(0, 255, (1000, 4))
    # points = np.random.rand(1000, 3)
    # add_color_save_meshlab(colors, points, "my_mesh.obj")
    #
    # output_mesh = pymeshlab.Mesh("my_mesh.obj")
    # print(output_mesh.vertex_color_matrix)

    # load obj
    pyvista_mesh = pv.read("corgi_mesh.ply255.obj")

    # convert to numpy
    points = pyvista_mesh.points
    normals = pyvista_mesh.point_arrays["Normals"]
    colors = pyvista_mesh.point_arrays["Colors"]  # doesnt work
    add_color_save_meshlab(colors, points, "my_mesh.obj")
