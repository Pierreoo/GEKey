import os
import numpy as np
import trimesh
import glob as glob
from tqdm import tqdm
from natsort import natsorted
import open3d as o3d
from multiprocessing import Pool, cpu_count

BASEDIR = os.path.dirname(os.path.abspath(__file__))


def anime_read(filename):
    """Read .anime file and return mesh data"""
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data


def normalize_pc(pc):
    pc = pc - pc.mean(0)
    pc /= np.max(np.linalg.norm(pc, axis=-1))
    return pc / 2


def farthest_point_sample(point, npoint):
    """Farthest point sampling - returns sampled points only"""
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return point[centroids.astype(np.int32)]


def farthest_point_sample_centroids(point, npoint):
    """Farthest point sampling - returns sampled points and their indices"""
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return point[centroids.astype(np.int32)], centroids.astype(np.int32)


def process_single_sequence(args):
    """Save pcd0 + offsets as .npz - one file per sequence"""
    category, anime_path, output_dir = args

    try:
        output_path = os.path.join(output_dir, category)
        output_file = os.path.join(output_path, f"{category}.npz")

        if os.path.exists(output_file):
            return (True, category, "skipped", True)

        _, _, _, vert_data, face_data, offset_data = anime_read(anime_path)

        vert_, index_ = farthest_point_sample_centroids(vert_data, 2048)
        offsets_ = offset_data[:, index_]

        xt = np.asarray([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype='f')
        yt = np.asarray([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype='f')
        pcd0 = (yt[:3, :3] @ xt[:3, :3] @ vert_.T).T

        off = np.asarray([
            (yt[:3, :3] @ xt[:3, :3] @ offset.T).T
            for offset in offsets_
        ])

        os.makedirs(output_path, exist_ok=True)
        np.savez(output_file, pcd0=pcd0, offsets=off)

        return (True, category, len(offsets_), False)

    except Exception as e:
        import traceback
        return (False, category, str(e), traceback.format_exc())



def process_single_frame(args):
    """Save .pcd - one file per frame.

    Pseudo-labels are not computed here: the dataset derives them on its first run
    (see DeformingThingsDataset.get_data_single) so that every dataset shares one
    graph/peak implementation.
    """
    i, offset_idx, name, anime_path, pcd_dir = args

    try:
        output_pcd_path = os.path.join(pcd_dir, name)
        pcd_file = os.path.join(output_pcd_path, f"{i}_{name}.pcd")

        if os.path.exists(pcd_file):
            return (True, name, i, -1, True)

        _, _, _, vert_data, face_data, offset_data = anime_read(anime_path)
        offset = offset_data[offset_idx]

        mesh = trimesh.Trimesh(vert_data + offset, face_data)

        xt = np.asarray([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype='f')
        yt = np.asarray([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype='f')
        pcd = normalize_pc(farthest_point_sample(mesh.sample(16000), 2048))
        pcd = (yt[:3, :3] @ xt[:3, :3] @ pcd.T).T

        os.makedirs(output_pcd_path, exist_ok=True)
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        o3d.io.write_point_cloud(pcd_file, pcd_o3d)

        return (True, name, i, len(pcd), False)

    except Exception as e:
        import traceback
        return (False, name, i, str(e), traceback.format_exc(), False)



def run_parallel(tasks, worker_fn, desc, num_processes):
    with Pool(processes=num_processes) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(worker_fn, tasks, chunksize=10),
                           total=len(tasks), desc=desc):
            results.append(result)
            if not result[0]:
                print(f"\nError: {result[1:]}")
    return results


def print_summary(results, label):
    successes = sum(1 for r in results if r[0])
    skipped = sum(1 for r in results if r[0] and len(r) > 3 and r[-1] is True)
    failures = len(results) - successes
    print(f"\n[{label}] {successes - skipped} processed, {skipped} skipped, {failures} failed")


def create_all_parallel(num_processes=None):
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    print(f"Using {num_processes} processes")

    data_dir = BASEDIR + '/anime'
    corres_dir = BASEDIR + '/corres'
    pcd_dir = BASEDIR + '/pcds'

    list_paths = natsorted(glob.glob(data_dir + '/*'))

    # Sequence-level tasks (correspondences)
    seq_tasks = []
    frame_tasks = []
    for path in list_paths:
        name = path.split('/')[-1]
        anime_path = os.path.join(path, f"{name}.anime")
        seq_tasks.append((name, anime_path, corres_dir))

        nf, _, _, _, _, offset_data = anime_read(anime_path)
        for i in range(len(offset_data)):
            frame_tasks.append((i, i, name, anime_path, pcd_dir))

    print(f"Sequences: {len(seq_tasks)} | Frames: {len(frame_tasks)}")

    print("\n-- Step 1: Correspondences --")
    seq_results = run_parallel(seq_tasks, process_single_sequence, "Correspondences", num_processes)
    print_summary(seq_results, "Correspondences")

    print("\n-- Step 2: Point clouds --")
    frame_results = run_parallel(frame_tasks, process_single_frame, "Point clouds", num_processes)
    print_summary(frame_results, "Point clouds")


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    num_processes = int(os.environ.get('NUM_PROCESSES', 4))
    create_all_parallel(num_processes=num_processes)