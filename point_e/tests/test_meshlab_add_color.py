from point_e.examples.meshlab_add_color import load_mesh_into_meshlab, mesh_repair


def test_mesh_repair():
    save_path = "/mnt/fast/code/point-e/point_e/examples/results/bazooka3.obj"
    # load mesh into meshlab
    ms = load_mesh_into_meshlab(save_path)
    ms = mesh_repair(ms)
    ms.save_current_mesh(save_path.replace(".obj", "_fixed.obj"))

if __name__ == "__main__":
    test_mesh_repair()
