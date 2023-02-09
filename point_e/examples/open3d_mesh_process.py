import open3d as o3d

# colors, normals, points, faces
def open3d_process(save_file_name):
    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(save_file_name)
    # subdivide
    #mesh = mesh.subdivide_loop(number_of_iterations=1)

    o3d.visualization.draw_geometries([mesh], point_show_normal=True)

    # mesh = mesh.subdivide_loop(number_of_iterations=2)
    # o3d.visualization.draw_geometries([mesh], point_show_normal=True)
    #
    # mesh = mesh.subdivide_loop(number_of_iterations=2)
    # o3d.visualization.draw_geometries([mesh], point_show_normal=True)

    pcd = convert_mesh_to_pointcloud(mesh)
    # pcd.compute_vertex_normals()

    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # pcd.orient_normals_consistent_tangent_plane(100)
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    for depth in range(5, 13):

        mesh_new = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, n_threads=1)[0]
        o3d.visualization.draw_geometries([mesh_new])
    # for scale  in range(1, 10):
    #     mesh_new = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([scale, scale * 2]))
    #     # mesh_new = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, scale=scale, n_threads=1)[0]
    #     o3d.visualization.draw_geometries([mesh_new])
    # for depth in range(1, 10):
    #
    #     mesh_new = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, n_threads=1)[0]
    #     o3d.visualization.draw_geometries([mesh_new])
    mesh_new = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([0.001, 0.0015, 0.002]))
    o3d.visualization.draw_geometries([mesh_new])
    mesh_new = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.005)
    o3d.visualization.draw_geometries([mesh_new])
    pass

def convert_mesh_to_pointcloud(mesh: o3d.geometry.TriangleMesh):
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals
    pcd.colors = mesh.vertex_colors
    return pcd



if __name__ == '__main__':
    save_file_name = 'results/bazooka3.obj'
    open3d_process(save_file_name)
