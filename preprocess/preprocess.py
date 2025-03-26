import pymeshlab as ml
import numpy as np
import glob
import argparse
import sys
import os
import shutil

### preprocessing for scan data with real-life noise

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from util.mesh import Mesh

def get_parser():
    parser = argparse.ArgumentParser(description='preprocessing')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('--step', type=int, default=30)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print('{:12s}: {}'.format(k, v))
    
    return args

def smooth(ms, s_file, step):
    ms.apply_filter("laplacian_smooth", stepsmoothnum=step, cotangentweight=False)
    ms.save_current_mesh(s_file)

def normalize(ms, n_file, s_file, g_file=None):
    ms.apply_filter("transform_scale_normalize", scalecenter="barycenter", unitflag=True, alllayers=True)
    ms.apply_filter("transform_translate_center_set_origin", traslmethod="Center on Layer BBox", alllayers=True)
    if ms.number_meshes() == 2:
        ms.set_current_mesh(0)
        ms.save_current_mesh(s_file)
        ms.set_current_mesh(1)
        ms.save_current_mesh(n_file)
    elif ms.number_meshes() == 3:
        ms.set_current_mesh(0)
        ms.save_current_mesh(s_file)
        ms.set_current_mesh(1)
        ms.save_current_mesh(g_file)
        ms.set_current_mesh(2)
        ms.save_current_mesh(n_file)

def main():
    args = get_parser()
    
    input_files = glob.glob(args.input + "/*.obj")
    if not input_files:
        print("No OBJ files found in input directory.")
        sys.exit(1)
    
    orig_file = input_files[0]  
    mesh_name = os.path.splitext(os.path.basename(orig_file))[0]
    output_dir = os.path.join(args.input, "4real")
    os.makedirs(output_dir, exist_ok=True)
    
    n_file = os.path.join(output_dir, mesh_name + "_noise.obj")
    s_file = os.path.join(output_dir, mesh_name + "_smooth.obj")
    g_file = os.path.join(args.input, mesh_name + "_gt.obj")
    
    shutil.copy(orig_file, n_file)  # copy original as noise mesh
    
    ms = ml.MeshSet()
    ms.load_new_mesh(n_file)
    smooth(ms, s_file, step=args.step)
    
    if os.path.exists(g_file):
        ms.load_new_mesh(g_file)  # meshset: [smooth, gt]
        ms.load_new_mesh(n_file)  # meshset: [smooth, gt, noise]
        normalize(ms, n_file, s_file, g_file)
        g_mesh = Mesh(g_file)
    else:
        ms.load_new_mesh(n_file)  # meshset: [smooth, noise]
        normalize(ms, n_file, s_file)
    
    n_mesh = Mesh(n_file)
    s_mesh = Mesh(s_file)
    
    edge_vec = n_mesh.vs[n_mesh.edges][:, 0, :] - n_mesh.vs[n_mesh.edges][:, 1, :]
    ave_len = np.sum(np.linalg.norm(edge_vec, axis=1)) / n_mesh.edges.shape[0]
    
    n_mesh.vs /= ave_len
    s_mesh.vs /= ave_len
    
    n_mesh.save(n_file)
    s_mesh.save(s_file)
    
    if os.path.exists(g_file):
        g_mesh.vs /= ave_len
        g_mesh.save(g_file)

if __name__ == "__main__":
    main()