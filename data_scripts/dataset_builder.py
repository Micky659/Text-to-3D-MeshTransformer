import torch
import trimesh
import numpy as np
import os
import json
import subprocess
from pathlib import Path
from collections import OrderedDict

from helper_funcs import get_mesh_data, augment_mesh, derive_face_edges_from_faces
from Mesh_dataset import MeshDataset


HAUSDORFF_THRESH = 0.01
ANGLE_CANDIDATES = np.linspace(1, 60, 12)  # NEW: Try decimation angles from 1° to 60°
MAX_FACE_COUNT = 800


def decimate_with_blender(in_path, out_path, angle_limit):
    subprocess.run([
        "blender", "--background", "--python", "decimate.py", "--",
        in_path, out_path, str(angle_limit)
    ], check=True)


def hausdorff_distance(m1: trimesh.Trimesh, m2: trimesh.Trimesh):
    return max(m1.hausdorff_distance(m2), m2.hausdorff_distance(m1))


def best_decimated_mesh(original_path):
    original_mesh = trimesh.load(original_path, force='mesh')
    best_path = None
    best_score = float("inf")

    for angle in ANGLE_CANDIDATES:
        tmp_path = f"/tmp/dec_{angle:.1f}_{os.path.basename(original_path)}"
        try:
            decimate_with_blender(original_path, tmp_path, angle)
            if not os.path.isfile(tmp_path):
                continue

            dec_mesh = trimesh.load(tmp_path, force='mesh')
            dist = hausdorff_distance(original_mesh, dec_mesh)
            if dist < HAUSDORFF_THRESH and dist < best_score and len(dec_mesh.faces) <= MAX_FACE_COUNT:
                best_path = tmp_path
                best_score = dist
        except Exception as e:
            print(f"Decimation failed at angle {angle}: {e}")
            continue

    return best_path or original_path  # fallback


def load_shapenet_data(dir_path, label_path, mesh_per_category, var, MAX_FILE_SIZE):
    mesh_dataset = []
    category_counts = {}

    with open(label_path, 'r') as f:
        label_list = json.load(f)
    label_dict = {
        entry['synsetId']: entry['name']
        for entry in label_list
    }

    scale_factors = np.random.uniform(0.75, 1.00, size=var)

    for synset in os.listdir(dir_path):
        cat_path = os.path.join(dir_path, synset)
        if not os.path.isdir(cat_path) or synset not in label_dict:
            continue

        category_counts[synset] = 0
        print(f"{synset} → {label_dict[synset]} has {len(os.listdir(cat_path))} instances")
        rem_ins = len(os.listdir(cat_path))
        for inst in os.listdir(cat_path):
            rem_ins = rem_ins - 1
            if category_counts[synset] >= mesh_per_category:
                break

            obj_file = os.path.join(
                cat_path, inst, 'models', 'model_normalized.obj'
            )
            if not os.path.isfile(obj_file):
                continue

            # if os.path.getsize(obj_file) > MAX_FILE_SIZE * 1024:
            #     continue

            # dec_path = best_decimated_mesh(obj_file)
            vertices, faces = get_mesh_data(obj_file)
            if len(faces) > MAX_FACE_COUNT:
                continue

            category_counts[synset] += 1
            text_label = label_dict[synset].replace(",", " ")
            faces = torch.tensor(faces.tolist(), dtype=torch.long).to('cuda')
            face_edges = derive_face_edges_from_faces(faces)

            for scale in scale_factors:
                aug_v = augment_mesh(vertices.copy(), scale)
                mesh_dataset.append({
                    'vertices': torch.tensor(aug_v.tolist(), dtype=torch.float).to('cuda'),
                    'faces':    faces,
                    'face_edges': face_edges,
                    'texts': text_label
                })
            print("Added ", len(scale_factors), "variations of ", text_label, " Remaining ins: ", rem_ins)

    print('=' * 30)
    print("Models per category:")
    for synset, cnt in category_counts.items():
        print(f"{synset} ({label_dict[synset]}): {cnt}")
    print(f"Total selected: {sum(category_counts.values())}")

    return mesh_dataset


if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    shapenet_dir = project_root / 'Data/ShapeNetCore.v2'  # TODO: set this
    labels_dir = project_root / 'Data/ShapeNetCore.v2/taxonomy.json'    # TODO: set this

    working_dir = project_root / "Data"
    working_dir.mkdir(exist_ok=True, parents=True)
    dataset_path = working_dir / "meshData.npz"

    # if not os.path.isfile(dataset_path):

    # data = load_shapenet_data(shapenet_dir, labels_dir, 500, 3, 5048)
    # dataset = MeshDataset(data)
    # dataset.generate_face_edges()
    # dataset.save(dataset_path+"_1")

    data = load_shapenet_data(shapenet_dir, labels_dir, 500, 2, 500)
    dataset = MeshDataset(data)
    dataset.generate_face_edges()
    dataset.save(working_dir / "meshData_15_500.npz")

    # data = load_shapenet_data(shapenet_dir, labels_dir, 500, 7, 1024)
    # dataset = MeshDataset(data)
    # dataset.generate_face_edges()
    # dataset.save(working_dir / "meshData_7_1024.npz")
