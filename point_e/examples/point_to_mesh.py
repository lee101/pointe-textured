import numpy as np
import open3d
import open3d as o3d
import torch
from loguru import logger
from tqdm import tqdm

from point_e.examples.draw_triangles import draw_triangles
from point_e.examples.meshlab_add_color import meshlab_process, add_color_save_meshlab, \
    meshlab_process_simplify_marching_cube_meshes
from point_e.examples.open3d_mesh_process import open3d_process
from point_e.examples.utils import convert_float_to_bfloat16
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.mesh import TriMesh
from point_e.util.pc_to_mesh import marching_cubes_mesh, marching_cubes_mesh_no_model
from point_e.util.point_cloud import PointCloud

import pyvista as pv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating SDF model...')
name = 'sdf'
model = model_from_config(MODEL_CONFIGS[name], device)
# model = convert_float_to_bfloat16(model)
model.eval()

print('loading SDF model...')
model.load_state_dict(load_checkpoint(name, device))
# model = convert_float_to_bfloat16(model)

model.eval()

# Load a point cloud we want to convert into a mesh.
def convert_point_cloud_to_mesh(filename_or_pointcloud='example_data/pc_corgi.npz', grid_size=32, save_file_name
='corgi_mesh.ply'):
    if isinstance(filename_or_pointcloud, str):
        pc = PointCloud.load(filename_or_pointcloud)
    else:
        pc = filename_or_pointcloud

    # Plot the point cloud as a sanity check.
    # fig = plot_point_cloud(pc, grid_size=2)
    # Produce a mesh (with vertex colors)
    # for grid_size in range(8,128,2):
    #     mesh = marching_cubes_mesh(
    #         pc=pc,
    #         model=model,
    #         batch_size=4096,
    #         grid_size=grid_size,  # increase to 128 for resolution used in evals
    #         progress=True,
    #     )
    #     # Write the mesh to a PLY file to import into some other program.
    #     with open(save_file_name + str(grid_size) +".ply", 'wb') as f:
    #         mesh.write_ply(f)
    #     mesh_vista = pv.read(save_file_name + str(grid_size) +".ply")
    #     mesh_vista.plot(off_screen=True, screenshot=save_file_name + str(grid_size) +".png")
    # log pc dtype
    logger.info(f'pc dtype: {pc.coords.dtype}')
    # convert pc to float32
    pc.coords = pc.coords.astype(np.float32)
    # lgo model dtype
    # logger.info(f'model dtype: {model.dtype}')
    mesh = marching_cubes_mesh(
        pc=pc,
        model=model,
        batch_size=4096,
        grid_size=grid_size,  # increase to 128 for resolution used in evals
        progress=True,
    )
    # Write the mesh to a PLY file to import into some other program.
    with open(save_file_name+ "2.ply", 'wb') as f:
        mesh.write_ply(f)
    logger.info("stuff")
    mesh_vista = pv.read(save_file_name + "2.ply")

    # mesh_vista = pv.PolyData(mesh.verts)
    # # set faces
    # mesh_vista.faces = mesh.faces
    # set normals
    # mesh_vista.point_arrays['normals'] = mesh.normals
    # mesh_vista = mesh_vista.compute_normals(auto_orient_normals=True) # recompute
    surf = mesh_vista
    logger.info("texture surf")
    # textured_surf = surf
    textured_surf = texture_surf_using_colored_pcd(surf, pc)
    # surf.point_data['colors'] = colors.transpose()
    # textured_surf.plot()
    textured_surf = textured_surf.rotate_x(270)
    textured_surf = ground(textured_surf)
    logger.info("save")

    # textured_surf.save(save_file_name, texture='colors')
    # faces = []  # .reshape(-1, 4)[:, 1:]
    # for i in range(1, len(faces), 4):
    #     faces.extend(faces[i:i + 3])
    # faces = np.array(faces).reshape(-1, 3)
    # delete every third face
    # calculate normals
    textured_surf = textured_surf.compute_normals(auto_orient_normals=True)
    textured_surf.save(save_file_name, texture='colors')

    colors = textured_surf.point_data['colors']
    normals = textured_surf.point_data['Normals']
    faces = []  # .reshape(-1, 4)[:, 1:]
    for i in range(1, len(faces), 4):
        faces.extend(faces[i:i + 3])
    # faces = np.array(faces).reshape(-1, 3)
    # delete every third face

    # add_color_save_meshlab(colors, normals, textured_surf.points, faces, save_file_name)
    # logger.info("save obj")
    # save_file_name = save_file_name.split('.')[0] + ".obj"
    # save_an_obj(textured_surf, save_file_name)

    meshlab_process_simplify_marching_cube_meshes(save_file_name)
    meshlab_process(save_file_name) # gets rotated again here :(



def pc_to_pointcloud(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.coords)
    channels = np.array([pc.channels['R'], pc.channels['G'], pc.channels['B']])
    channels = channels.transpose()
    pcd.colors = o3d.utility.Vector3dVector(channels)
    return pcd


def texture_surf_using_colored_pcd(surf, pc):
    """texture a surface flat per vertex using a colored point cloud
    1 build a octree
    2 for each vertex find the closest four points in the octree
    3 lerp between the four colors for each point
    """
    # build octree
    # octree = o3d.geometry.Octree(max_depth=8)
    # octree = o3d.geometry.Octree(max_depth=8)
    pointcloud = pc_to_pointcloud(pc)
    octree = o3d.geometry.KDTreeFlann(pointcloud)

    # octree.convert_from_point_cloud(pointcloud)
    # create KDTreeFlann
    # kdtree = o3d.geometry.KDTreeFlann(pointcloud)
    # # init kd tree
    # kdtree.init_index()

    texture = np.zeros((surf.n_points, 3), np.uint8)
    for i in range(surf.n_points):
        # find closest four points
        current_point = surf.points[i]
        closest_points = octree.search_knn_vector_3d(current_point, 4)
        # lerp between the four colors based on distance to points
        closest_points_indexes = closest_points[1]
        closest_points_values = []
        closest_points_color_values = []
        for index in closest_points_indexes:
            closest_points_values.append(pointcloud.points[index])
            closest_points_color_values.append(pointcloud.colors[index])
        distances = []
        for point in closest_points_values:
            distances.append(np.linalg.norm(point - current_point))
        # distances = np.linalg.norm(closest_points[1] - current_point, axis=1)
        # for j in range(3):
        # texture[i, j] = np.average(closest_points_color_values[j], weights=1/np.array(distances))
        color = np.average(closest_points_color_values, weights=1 / np.array(distances), axis=0)
        texture[i] = color * 255
    surf.point_data['colors'] = texture
    return surf


def fix_normals(save_file_name):
    """opens the file and auto fixes the normals"""
    mesh = pv.read(save_file_name)
    mesh = mesh.compute_normals(auto_orient_normals=True)
    pl = pv.Plotter()
    # save with colors

    _ = pl.add_mesh(mesh, show_edges=True, rgb=True)  # show_edges=True
    save_file_name = save_file_name + "5.obj"
    pl.export_obj(save_file_name)
    # mesh.save(save_file_name)

# def pyvista_marching_cubes(pointcloud, grid_size=32):
#     grid = pv.UniformGrid(
#         dimensions=(grid_size, grid_size, grid_size),
#         origin=(0, 0, 0),
#         spacing=(1 / grid_size, 1 / grid_size, 1 / grid_size),
#     # spacing=(abs(x_min) / n * 2, abs(y_min) / n * 2, abs(z_min) / n * 2),
#     # origin=(x_min, y_min, z_min),
#     )
#     values = np.zeros((grid_size, grid_size, grid_size))
#     for i in range(grid_size):
#         for j in range(grid_size):
#             for k in range(grid_size):
#                 # check if point is in point cloud
#                 x = i / grid_size
#                 y = j / grid_size
#                 z = k / grid_size
#                 point = np.array([x, y, z])
#                 distances = np.linalg.norm(point - pointcloud.coords, axis=1)
#                 values[i, j, k] = model(torch.tensor([i / grid_size, j / grid_size, k / grid_size]).to(device)).item()
#     mesh = grid.contour([1], values, method='marching_cubes')
def marching_cubes_mesh_creation(save_file_name, colorpc, grid_size=32, pc=None):
    # load
    input_mesh = pv.read(save_file_name)
    # get the points
    points = input_mesh.points
    # get the faces
    pc = PointCloud(points, {})
    # pc.coords = points
    # pc.channels = {}
    # pc.channels['R'] = mesh.point_arrays['colors'][:, 0]
    # pc.channels['G'] = mesh.point_arrays['colors'][:, 1]
    # pc.channels['B'] = mesh.point_arrays['colors'][:, 2]
    mesh: TriMesh = marching_cubes_mesh(
        pc=pc,
        model=model,
        batch_size=4096,
        grid_size=grid_size,  # increase to 128 for resolution used in evals
        progress=True,
    )
    # Write the mesh to a PLY file to import into some other program.
    with open(save_file_name + "2.ply", 'wb') as f:
        mesh.write_ply(f)
    # convert ply to obj

    # pcd = pc_to_pointcloud(mesh.verts)
    # # estimate normals
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_delaunay(pcd)
    # mesh.plot()
    logger.info("stuff")
    # load ply file
    mesh_vista = pv.read(save_file_name + "2.ply")

    # mesh_vista = pv.PolyData(mesh.verts)
    # set faces
    mesh_vista.faces = mesh.faces
    # set normals
    mesh_vista.point_arrays['normals'] = mesh.normals
    # mesh_vista = mesh_vista.compute_normals(auto_orient_normals=True) # recompute
    surf = mesh_vista
    logger.info("texture surf")
    textured_surf = texture_surf_using_colored_pcd(surf, colorpc)
    # surf.point_data['colors'] = colors.transpose()
    # textured_surf.plot()
    textured_surf = textured_surf.rotate_x(270)
    textured_surf = ground(textured_surf)
    logger.info("save")
    # save_file_name = save_file_name + "3n.obj"
    # textured_surf.save(save_file_name, texture='colors')
    # faces = []  # .reshape(-1, 4)[:, 1:]
    # for i in range(1, len(faces), 4):
    #     faces.extend(faces[i:i + 3])
    # faces = np.array(faces).reshape(-1, 3)
    # delete every third face

    # add_color_save_meshlab(colors, normals, textured_surf.points, faces, save_file_name)
    logger.info("save obj")

    save_an_obj(textured_surf, save_file_name)



# def convert_point_cloud_to_mesh_newdelaney(filename_or_pointcloud='example_data/pc_corgi.npz', grid_size=32, save_file_name
# ='corgi_mesh.ply'):
#     if isinstance(filename_or_pointcloud, str):
#         pc = PointCloud.load(filename_or_pointcloud)
#     else:
#         pc = filename_or_pointcloud
#     #load in open3d
#     pcd = pc_to_pointcloud(pc)
#     #estimate normals
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#
#     # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_delaunay(pcd)
#     # mesh.plot()
#
#     cloud = pv.PolyData(pc.coords)
#     # cloud.plot()
#
#     # shell.plot()
#     # shell = shell.decimate(.8).extract_surface().clean()
#     # shell.plot()
#
#     # shell.plot()
#     # for alpha in tqdm(np.linspace(0.03, 0.09, 10)):
#     # #
#     #     volume = cloud.delaunay_3d(alpha=alpha)
#     #     shell = volume.extract_geometry().triangulate()
#     # #     #shell.plot()
#     # #     #take screenshot of shell
#     #     shell.plot(off_screen=True, screenshot=save_file_name + str(alpha) + "-.png")
#     volume = cloud.delaunay_3d(alpha=.06)  # todo search around this range
#     shell = volume.extract_geometry().triangulate()
#     # surf = shell.extract_surface()
#     # volume = cloud.delaunay_3d(alpha=.06)
#     # volume = volume.clean(tolerance=0.0001)
#     # volume = volume.remove_duplicate_cells()
#     # # remove points not used
#     # volume = volume.remove_unused_points()
#     # for spacing in tqdm(range(0,25)):
#     #     surf = shell.reconstruct_surface(nbr_sz=spacing)
#     #     surf = texture_surf_using_colored_pcd(surf, pc)
#     #     surf.plot(off_screen=True, screenshot=save_file_name + str(spacing) + ".png")
#     surf = shell #cloud.reconstruct_surface()
#     # close mesh
#     # surf = surf.fill_holes(hole_size=1000)
#     # remove duplicate points
#     # surf = surf.remove_duplicate_points()
#
#     # remove forests
#     # surf.plot(off_screen=True, screenshot=save_file_name + "preclean.png")
#     # surf2 = surf.clean(tolerance=0.02)
#     # surf.plot(off_screen=True, screenshot=save_file_name + "postclean.png")
#     # surf = surf.clean(tolerance=0.04)
#     # surf.plot(off_screen=True, screenshot=save_file_name + "postclean2.png")
#     # surf = surf.clean(tolerance=0.06)
#     # surf.plot(off_screen=True, screenshot=save_file_name + "postclean3.png")
#     # surf = surf.clean(tolerance=0.1)
#     # surf.plot(off_screen=True, screenshot=save_file_name + "postclean4.png")
#     # surf = surf2
#
#
#     # remove non manifold edges
#
#     # surf = surf.remove_non_manifold_edges()
#     # smooth mesh
#     # surf = surf.smooth(n_iter=10, relaxation_factor=0.1)
#     # fix normals
#     surf = surf.compute_normals(auto_orient_normals=True)
#     # check normals for being flipped
#     # todo code and flip_normals if need be
#
#     # create mtl file
#     # colors = np.array([pc.channels['R'], pc.channels['G'], pc.channels['B']])
#     # # texture = np.zeros((sphere.n_points, 3), np.uint8)
#     # get_reprojected_colors = surf.texture_map_to_plane(inplace=False)
#     # texture_coords = surf.get_array('Texture Coordinates')
#     textured_surf = texture_surf_using_colored_pcd(surf, pc)
#     # textured_surf = surf
#     # surf.point_data['colors'] = colors.transpose()
#     # textured_surf.plot()
#     textured_surf = textured_surf.rotate_x(270)
#     textured_surf = ground(textured_surf)
#
#     textured_surf.save(save_file_name, texture='colors')
#
#     colors = textured_surf.point_data['colors']
#     normals = textured_surf.point_data['Normals']
#     faces = []  # .reshape(-1, 4)[:, 1:]
#     for i in range(1, len(faces), 4):
#         faces.extend(faces[i:i + 3])
#     # faces = np.array(faces).reshape(-1, 3)
#     # delete every third face
#
#     # add_color_save_meshlab(colors, normals, textured_surf.points, faces, save_file_name)
#     # save_an_obj(textured_surf, save_file_name)
#     meshlab_process(save_file_name)
#
#     # redo loading and delaney
#     shell = pv.read(save_file_name)
#     # ensure all triangles in the mesh
#     shell = shell.triangulate()
#     # todo ensure manifoldness required for subdivision
#     # shell = shell.remove_non_manifold_edges()
#
#     # loop subdivision to get better results
#     # shell = shell.clean().subdivide(1, subfilter='butterfly')
#     shell = shell.clean().subdivide_adaptive()
#     # convert back to pc and do delaunay_3d again
#     cloud = pv.PolyData(shell.points)
#
#     # for alpha in tqdm(np.linspace(0.03, 0.09, 10)):
#     # #
#     #     volume = cloud.delaunay_3d(alpha=alpha)
#     #     shell = volume.extract_geometry().triangulate()
#     # #     #shell.plot()
#     # #     #take screenshot of shell
#     #     shell.plot(off_screen=True, screenshot=save_file_name + str(alpha) + "--.png")
#     volume = cloud.delaunay_3d(alpha=.06)  # todo search around this range
#     shell = volume.extract_geometry()
#     # shell.plot()
#     shell.save(save_file_name)
#
#     surf=shell
#
#     surf = surf.compute_normals(auto_orient_normals=True)
#     # check normals for being flipped
#     # todo code and flip_normals if need be
#
#     # create mtl file
#     # colors = np.array([pc.channels['R'], pc.channels['G'], pc.channels['B']])
#     # # texture = np.zeros((sphere.n_points, 3), np.uint8)
#     # get_reprojected_colors = surf.texture_map_to_plane(inplace=False)
#     # texture_coords = surf.get_array('Texture Coordinates')
#     textured_surf = texture_surf_using_colored_pcd(surf, pc)
#     textured_surf.save(save_file_name, texture='colors')
#
#     save_an_obj(textured_surf, save_file_name)
#     meshlab_process( save_file_name)
#     # redo loading and delaney
#     shell = pv.read(save_file_name)
#     # ensure all triangles in the mesh
#     shell = shell.triangulate()
#     # todo ensure manifoldness required for subdivision
#     # shell = shell.remove_non_manifold_edges()
#
#     # loop subdivision to get better results
#     # shell = shell.clean().subdivide(1, subfilter='butterfly')
#     shell = shell.clean().subdivide_adaptive(max_n_passes=2)
#     # convert back to pc and do delaunay_3d again
#     cloud = pv.PolyData(shell.points)
#
#     # for alpha in tqdm(np.linspace(0.03, 0.09, 10)):
#     # #
#     #     volume = cloud.delaunay_3d(alpha=alpha)
#     #     shell = volume.extract_geometry().triangulate()
#     # #     #shell.plot()
#     # #     #take screenshot of shell
#     #     shell.plot(off_screen=True, screenshot=save_file_name + str(alpha) + "--.png")
#     volume = cloud.delaunay_3d(alpha=.068)  # todo search around this range
#     shell = volume.extract_geometry()
#
#     shell.save(save_file_name)
#     surf = shell
#
#     surf = surf.compute_normals(auto_orient_normals=True)
#     # check normals for being flipped
#     # todo code and flip_normals if need be
#
#     # create mtl file
#     # colors = np.array([pc.channels['R'], pc.channels['G'], pc.channels['B']])
#     # # texture = np.zeros((sphere.n_points, 3), np.uint8)
#     # get_reprojected_colors = surf.texture_map_to_plane(inplace=False)
#     # texture_coords = surf.get_array('Texture Coordinates')
#     # repair the surface
#     # surf = surf.fill_holes()
#     # voxelize model
#     # surf = pv.voxelize(surf, density=surf.length/100, check_surface=False)
#     textured_surf = texture_surf_using_colored_pcd(surf, pc)
#     textured_surf.save(save_file_name, texture='colors')
#
#     save_an_obj(textured_surf, save_file_name)
#
#     meshlab_process(save_file_name)
#     # marching cubes over the surface
#     # shell = shell.marching_cubes()
#     # marching_cubes_mesh_creation(save_file_name, pc)
#
#     # open3d_process(colors, normals, textured_surf.points, faces, save_file_name)
#     # open3d_process(save_file_name)
#     # fix_normals(save_file_name)
#     # save_obj(surf, save_file_name)
#     # shell.save(save_file_name)


def point_in_triangle(point, triangle):
    """Returns True if the point is inside the triangle
    and returns False if it falls outside.
    - The argument *point* is a tuple with two elements
    containing the X,Y coordinates respectively.
    - The argument *triangle* is a tuple with three elements each
    element consisting of a tuple of X,Y coordinates.

    It works like this:
    Walk clockwise or counterclockwise around the triangle
    and project the point onto the segment we are crossing
    by using the dot product.
    Finally, check that the vector created is on the same side
    for each of the triangle's segments.
    """
    # Unpack arguments
    x, y = point
    ax, ay = triangle[0]
    bx, by = triangle[1]
    cx, cy = triangle[2]
    # Segment A to B
    side_1 = (x - bx) * (ay - by) - (ax - bx) * (y - by)
    # Segment B to C
    side_2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy)
    # Segment C to A
    side_3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay)
    # All the signs must be positive or all negative
    return (side_1 < 0.0) == (side_2 < 0.0) == (side_3 < 0.0)


def lerp_color(coords, triangle, colors):
    # get distances to each point
    distances = []
    for point in triangle:
        distances.append(np.linalg.norm(point - coords))
    # lerp between the four colors based on distance to points
    color = np.average(colors, weights=1 / np.array(distances), axis=0)
    return color


def map_face_to_texture_coords(face, texture_coords, size):
    texture_coord_mapped = []
    for i in range(3):
        texture_coord_mapped.append(np.round(texture_coords[face[i]] * 512.0).astype(int))
    return texture_coord_mapped


def vista_mesh_texture(vista_mesh):
    # mesh.vertices = o3d.utility.Vector3dVector(vista_object.points)

    # Add back vertex color
    # mesh.vertex_colors = o3d.utility.Vector3dVector(vista_mesh.point_data['colors'] / 255.0)

    # unwrap texture coordinates
    vista_mesh.texture_map_to_plane(inplace=True)
    texture_coords = vista_mesh.get_array('Texture Coordinates')
    # create new texture
    # texture = np.zeros((512, 512, 3), np.uint8)
    # set_texture_mask = np.zeros((512, 512), np.uint8)
    # for i in range(vista_mesh.n_points):
    #     color = vista_mesh.point_data['colors'][i]
    #     texture_coord = texture_coords[i]
    #     texture_coord[0] = min(1, texture_coord[0]) # sometimes above 1 slightly
    #     texture_coord[1] = min(1, texture_coord[1])
    #     texture[int(texture_coord[0] * 511), int(texture_coord[1] * 511)] = color
    #     set_texture_mask[int(texture_coord[0] * 511), int(texture_coord[1] * 511)] = 1

    # iterate over all faces and color all faces using lerp
    triangles_coords = []
    triangles_colors = []
    for i in range(1, vista_mesh.n_faces, 4):
        face = [vista_mesh.faces[i], vista_mesh.faces[i + 1], vista_mesh.faces[i + 2]]  # todo assuming all is trianges
        triangles_coords.append(map_face_to_texture_coords(face, texture_coords, 512))
        triangles_colors.append([vista_mesh.point_data['colors'][face[0]], vista_mesh.point_data['colors'][face[1]],
                                 vista_mesh.point_data['colors'][face[2]]])
        # triangle = []
        # colors = []
        # current_texture_coords = []
        # for point_index in face:
        #     triangle.append(vista_mesh.points[point_index])
        #     colors.append(vista_mesh.point_data['colors'][point_index])
        #     current_texture_coords.append(texture_coords[point_index])
        # # iterate over all pixels in the texture
        # for x in range(512):
        #     for y in range(512):
        #         # # if the pixel is not already set
        #         # if set_texture_mask[x, y] == 0:
        #         # if the pixel is inside the triangle
        #
        #         if point_in_triangle((x, y), current_texture_coords):
        #             # set the pixel to the lerp color
        #             texture[x, y] = lerp_color((x, y), current_texture_coords, colors)
        #             set_texture_mask[x, y] = 1
    texture = draw_triangles(triangles_coords, triangles_colors)
    # load image from disk
    # texture = cv2.imread('texture_map.jpg')

    # for i in range(len(mesh.vertices)):
    #     color1 = vista_mesh.point_data['colors'][mesh.vertices[i][0]]
    #     color2 = vista_mesh.point_data['colors'][mesh.vertices[i][1]]
    #     color3 = vista_mesh.point_data['colors'][mesh.vertices[i][2]]
    #
    #     texture_coords1 = texture_coords[mesh.vertices[i][0]]
    #     texture_coords2 = texture_coords[mesh.vertices[i][1]]
    #     texture_coords3 = texture_coords[mesh.vertices[i][2]]
    #     # get all points in triangle
    #     triangle = np.array([texture_coords1, texture_coords2, texture_coords3])
    #     # get bounding box
    #     min_x = int(np.floor(np.min(triangle[:, 0]) * 512))
    #     max_x = int(np.ceil(np.max(triangle[:, 0]) * 512))
    #     min_y = int(np.floor(np.min(triangle[:, 1]) * 512))
    #     max_y = int(np.ceil(np.max(triangle[:, 1]) * 512))
    #     # iterate over all points in bounding box
    #     for x in range(min_x, max_x):
    #         for y in range(min_y, max_y):
    #             # check if point is in triangle
    #             if point_in_triangle([x / 512, y / 512], triangle):
    #                 # lerp color
    #                 color = lerp_color([x / 512, y / 512], triangle, [color1, color2, color3])
    #                 texture[x, y] = color
    #                 set_texture_mask[x, y] = 1
    # set texture
    # mesh.texture = o3d.geometry.Image(texture)
    # o3d.visualization.draw_geometries([mesh])

    vista_mesh.textures["base"] = pv.numpy_to_texture(texture)
    # vista_mesh.textures["base"].interpolate = True
    vista_mesh.plot(texture="base", show_edges=True)
    # color the mesh texture from the vertex colors
    # skips having to do triangle to face conversion
    return vista_mesh


def vista_mesh_to_open3d_mesh_and_texture(vista_mesh):
    vista_mesh.save('temp.ply')
    mesh: open3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh('temp.ply')
    mesh.compute_vertex_normals()
    # compute texture map
    mesh.compute_triangle_uvs()
    # unwrap texture coordinates
    mesh.texture_map_to_plane(inplace=True)
    texture_coords = mesh.get_texture_coordinates()
    triangles_coords = []
    triangles_colors = []
    triangle_uvs = np.asarray(mesh.triangle_uvs)
    for i in range(mesh.triangles):
        face = [mesh.triangles[i][0], mesh.triangles[i][1], mesh.triangles[i][2]]  # todo assuming all is trianges
        uvs = ((triangle_uvs[i * 3], triangle_uvs[i * 3 + 1], triangle_uvs[i * 3 + 2]) * 512)
        triangles_coords.append(face)
        triangles_colors.append(uvs)
    texture = draw_triangles(triangles_coords, triangles_colors)
    mesh.textures = [o3d.geometry.Image(texture)]
    # plot
    o3d.visualization.draw_geometries([mesh])
    # save
    o3d.io.write_triangle_mesh('temp.ply', mesh)
    return mesh
    # create new texture


# def save_obj(vista_object, save_file_name):
#     # convert to open3d mesh
#     mesh = vista_mesh_to_open3d_mesh_and_texture(vista_object)
#     pl = pv.Plotter()
#     # save with colors
#
#     _ = pl.add_mesh(vista_object, scalars='colors', show_edges=True, rgb=True) # show_edges=True
#     save_file_name = save_file_name + ".obj"
#     pl.export_obj(save_file_name)
def save_an_obj(vista_object, save_file_name):
    # convert to open3d mesh
    pl = pv.Plotter()
    # save with colors

    _ = pl.add_mesh(vista_object, scalars='colors', show_edges=True, rgb=True)  # show_edges=True
    save_file_name = save_file_name.split('.')[0] + ".obj"
    pl.export_obj(save_file_name)


# def save_obj(vista_object, save_file_name):
#     # convert to open3d mesh
#     mesh = vista_mesh_to_open3d_mesh_and_texture(vista_object)
#     # mesh = vista_mesh_texture(vista_object)
#
#     save_file_name = save_file_name.split('.')[0] + ".obj"
#     # pyvista save
#     # pl = pv.Plotter()
#     #
#     # # save with texture
#     # _ = pl.add_mesh(mesh, texture=True) # show_edges=True
#     # # pl.e
#     # pl.export_obj(save_file_name)
#
#     o3d.io.write_triangle_mesh(save_file_name, mesh, write_vertex_colors=True) # todo only need texture not vertex colors


def rotate90(mesh):
    """Rotate a mesh 90 degrees around the x axis so y faces forward"""
    mesh.rotate_x(90)
    return mesh


def ground(mesh):
    """Move a mesh so that the lowest point is at 0"""
    mesh.translate([0, 0, -mesh.bounds[4]])
    return mesh


# def save_obj(vista_object, save_file_name):
#     # pl = pv.Plotter()
#     # # save with colors
#     #
#     # _ = pl.add_mesh(vista_object, scalars='colors', show_edges=True) # show_edges=True
#     save_file_name = save_file_name + ".obj"
#     faces = vista_object.faces
#     triangles = []
#     # convert vista 1d array of faces to triangles of (nvert, 3)
#     for i in range(0, len(faces), 4):
#         triangles.append(faces[i:i + 3])
#
#     # for face in range(len(faces, :
#     #     triangles.append([face[0], face[1], face[2]])
#     write_obj_with_colors_texture(save_file_name, vista_object.points, triangles, vista_object.point_data['colors'], None, None)
#     write_obj_with_colors_texture(save_file_name, vista_object.points, np.array(triangles),
#                                   vista_object.point_data['colors'], None, None)

def convert_point_cloud_to_mesh_oldb(filename_or_pointcloud='example_data/pc_corgi.npz', grid_size=32, save_file_name
='corgi_mesh.ply'):
    if isinstance(filename_or_pointcloud, str):
        pc = PointCloud.load(filename_or_pointcloud)
    else:
        pc = filename_or_pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.coords)
    channels = np.array([pc.channels['R'], pc.channels['G'], pc.channels['B']])
    channels = channels.transpose()
    pcd.colors = o3d.utility.Vector3dVector(channels)
    # infer normals from point cloud
    pcd.estimate_normals()
    poisson_mesh = \
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    # cropping
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    # o3d.visualization.draw_geometries([p_mesh_crop])
    o3d.io.write_triangle_mesh(save_file_name + "p_mesh_c.ply", p_mesh_crop)
    o3d.io.write_triangle_mesh(save_file_name + "bpa_mesh.ply", poisson_mesh)


if __name__ == '__main__':
    # convert_point_cloud_to_mesh()
    convert_point_cloud_to_mesh(save_file_name='corgi_mesh_4.ply')
    # convert_point_cloud_to_mesh(save_file_name='/mnt/fast/code/point-e/point_e/examples/results/bazooka3.ply.ply')
    # convert_point_cloud_to_mesh(save_file_name='chair.ply')
    # convert_point_cloud_to_mesh(filename_or_pointcloud='example_data/pc_cube_stack.npz', grid_size=32)
    # convert_point_cloud_to_mesh_old(filename_or_pointcloud='example_data/pc_cube_stack.npz', grid_size=32)
