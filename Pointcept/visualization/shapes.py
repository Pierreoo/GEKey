"""
Visualize category point clouds in the KeyPointNet dataset
"""

import os
import glob
import open3d as o3d


def visualize_keypointnet_category(base_path="../data/keypointnet/pcds", category_id="02691156"):
    pcd_folder = os.path.join(base_path, category_id)

    if not os.path.exists(pcd_folder):
        print(f"Error: Folder not found: {pcd_folder}")
        return

    pcd_files = glob.glob(os.path.join(pcd_folder, "*.pcd"))

    if not pcd_files:
        print(f"No .pcd files found in {pcd_folder}")
        return

    print(f"Found {len(pcd_files)} airplane models")
    print("Press 'N' for next, 'Q' to quit")
    print("-" * 50)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Airplane Models", width=800, height=600)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = [0.1, 0.1, 0.1]

    current_idx = [0]
    current_pcd = [None]

    def load_next_model(vis):
        if current_pcd[0] is not None:
            vis.remove_geometry(current_pcd[0], reset_bounding_box=False)

        if current_idx[0] >= len(pcd_files):
            print("All models viewed!")
            vis.close()
            return

        pcd_path = pcd_files[current_idx[0]]
        model_id = os.path.splitext(os.path.basename(pcd_path))[0]

        pcd = o3d.io.read_point_cloud(pcd_path)

        if pcd.is_empty():
            print(f"Skipping empty: {model_id}")
            current_idx[0] += 1
            load_next_model(vis)
            return

        pcd.paint_uniform_color([0.5, 0.0, 0.5])

        print(f"[{current_idx[0] + 1}/{len(pcd_files)}] {model_id}")

        vis.add_geometry(pcd, reset_bounding_box=True)
        current_pcd[0] = pcd
        current_idx[0] += 1

    vis.register_key_callback(ord('N'), lambda vis: load_next_model(vis))
    load_next_model(vis)
    vis.run()
    vis.destroy_window()
    print("-" * 50)
    print("Visualization complete!")


if __name__ == "__main__":
    visualize_keypointnet_category("../../data/keypointnet/pcds", category_id="02691156")