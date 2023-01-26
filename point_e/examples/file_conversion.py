import open3d as o3d


def convert_ply_to_obj(ply_filename, save_filename):
    """ load a ply file into open3d and save as obj """
    import open3d as o3d
    pcd = o3d.io.read_triangle_mesh(ply_filename)
    o3d.io.write_triangle_mesh(ply_filename.replace(".ply", ".obj"), pcd)


if __name__ == "__main__":
    convert_ply_to_obj("corgi_mesh_2.ply", "corgi_mesh_2_.obj")
