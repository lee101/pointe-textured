import pymeshlab
import pyvista as pv
import numpy as np


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


def meshlab_process(colors, normals, points, faces, save_path):
    ms = load_mesh_into_meshlab(save_path)
    #ms.generate_surface_reconstruction_screened_poisson(depth=5, scale=1.1)
    # ms.generate_surface_reconstruction_ball_pivoting()
    ms.compute_texcoord_parametrization_triangle_trivial_per_wedge()
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
