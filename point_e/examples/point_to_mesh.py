import open3d as o3d
from PIL import Image
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud

import pyvista as pv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating SDF model...')
name = 'sdf'
model = model_from_config(MODEL_CONFIGS[name], device)
model.eval()

print('loading SDF model...')
model.load_state_dict(load_checkpoint(name, device))
# Load a point cloud we want to convert into a mesh.
def convert_point_cloud_to_mesh_old(filename_or_pointcloud='example_data/pc_corgi.npz', grid_size=32, save_file_name
                                ='corgi_mesh.ply'):
    if isinstance(filename_or_pointcloud, str):
        pc = PointCloud.load(filename_or_pointcloud)
    else:
        pc = filename_or_pointcloud

    # Plot the point cloud as a sanity check.
    # fig = plot_point_cloud(pc, grid_size=2)
    # Produce a mesh (with vertex colors)
    mesh = marching_cubes_mesh(
        pc=pc,
        model=model,
        batch_size=4096,
        grid_size=grid_size, # increase to 128 for resolution used in evals
        progress=True,
    )
    # Write the mesh to a PLY file to import into some other program.
    with open(save_file_name, 'wb') as f:
        mesh.write_ply(f)
def convert_point_cloud_to_mesh_older(filename_or_pointcloud='example_data/pc_corgi.npz', grid_size=32, save_file_name
                                ='corgi_mesh.ply'):
    if isinstance(filename_or_pointcloud, str):
        pc = PointCloud.load(filename_or_pointcloud)
    else:
        pc = filename_or_pointcloud

    cloud = pv.PolyData(pc.coords)
    # cloud.plot()

    # volume = cloud.delaunay_3d(alpha=10000)
    # shell = volume.extract_geometry()
    # shell.plot()
    surf = cloud.reconstruct_surface()
    surf.plot()
    surf.save(save_file_name)
    # shell.save(save_file_name)

def convert_point_cloud_to_mesh(filename_or_pointcloud='example_data/pc_corgi.npz', grid_size=32, save_file_name
                                ='corgi_mesh.ply'):
    if isinstance(filename_or_pointcloud, str):
        pc = PointCloud.load(filename_or_pointcloud)
    else:
        pc = filename_or_pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.coords)
    channels = [pc.channels['R'], pc.channels['G'], pc.channels['B']]
    pcd.colors = o3d.utility.Vector3dVector(channels)
    poisson_mesh = \
    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    #cropping
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    # o3d.visualization.draw_geometries([p_mesh_crop])
    o3d.io.write_triangle_mesh(save_file_name+"p_mesh_c.ply", p_mesh_crop)
    o3d.io.write_triangle_mesh(save_file_name+"bpa_mesh.ply", poisson_mesh)

if __name__ == '__main__':
    convert_point_cloud_to_mesh()
    # convert_point_cloud_to_mesh(filename_or_pointcloud='example_data/pc_cube_stack.npz', grid_size=32)
    # convert_point_cloud_to_mesh_old(filename_or_pointcloud='example_data/pc_cube_stack.npz', grid_size=32)
