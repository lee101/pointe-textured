from point_e.util.point_cloud import PointCloud
import open3d as o3d
import numpy as np

def convert_point_cloud_to_pcd(filename_or_pointcloud='example_data/pc_corgi.npz'):
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

    return pcd

voxel_size = 0.02

max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph


def convert_pcd_to_point_cloud(pcd):
    coords = np.asarray(pcd.points)
    channels = np.asarray(pcd.colors)
    channels = channels.transpose()
    channels = {
        'R': channels[0],
        'G': channels[1],
        'B': channels[2],
    }
    pc = PointCloud(coords, channels)
    return pc


def align_two_point_clouds(pcd1, pcd2):
    # pcd1.paint_uniform_color([1, 0.706, 0])
    # pcd2.paint_uniform_color([0, 0.651, 0.929])
    # o3d.visualization.draw_geometries([pcd1, pcd2])

    pc1 = convert_point_cloud_to_pcd(pcd1)
    pc1.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pc2 = convert_point_cloud_to_pcd(pcd2)
    pc2.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # print("Apply point-to-point ICP")
    # reg_p2p = o3d.registration.registration_icp(
    #     pc1, pc2, 0.2, np.eye(4),
    #     o3d.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p)

    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # print("")
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    source_down = pc1.voxel_down_sample(voxel_size)
    target_down = pc2.voxel_down_sample(voxel_size)

    pcds = [source_down, target_down]

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds,
                                       max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine)
    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]

    # write mesh to file?

    # pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    # convert back to point e pointcloud
    pcd_combined = convert_pcd_to_point_cloud(pcd_combined)
    return pcd_combined
