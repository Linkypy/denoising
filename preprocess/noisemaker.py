import pymeshlab as ml
import numpy as np
import shutil
import os
import argparse
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from util.mesh import Mesh
import util.loss as Loss

### preprocess for synthetic data

def get_parser():
    parser = argparse.ArgumentParser(description='create datasets(noisy mesh & smoothed mesh) from a single clean mesh')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('--noise', type=str, default="gaussian")
    parser.add_argument('--glevel', type=float, default="1.0")
    parser.add_argument('--plevel', type=float, default="2.0")
    parser.add_argument('--lam', type=float, default="32")
    parser.add_argument('--sppercent', type=float, default="0.6")
    parser.add_argument('--step', type=int, default=30)
    parser.add_argument('--spmagnitude', type=float, default="1.0")
    parser.add_argument('--spfmagnitude', type=float, default="0.05")
    args = parser.parse_args()

    for k, v in vars(args).items():
        print('{:12s}: {}'.format(k, v))

    return args

def smooth(ms, step):
    ms.apply_filter("laplacian_smooth", stepsmoothnum=step, cotangentweight=False)

def normalize(ms):
    ms.apply_filter("transform_scale_normalize", scalecenter= "barycenter", unitflag=True)
    ms.apply_filter("transform_translate_center_set_origin", traslmethod="Center on Layer BBox")

def edge_based_scaling(mesh):
    # dimensions are (Number of vertices, number of vertices, dims of vertice)
    # or rather (N,2,3) where 2 is one pair of an edge tuple
    edge_vec = mesh.vs[mesh.edges][:, 0, :] - mesh.vs[mesh.edges][:, 1, :]
    ave_len = np.sum(np.linalg.norm(edge_vec, axis=1)) / mesh.edges.shape[0]
    mesh.vs /= ave_len
    return mesh

def gausian_noise(mesh, noise_level):
    np.random.seed(314)
    noise = np.random.normal(loc=0, scale=noise_level, size=(len(mesh.vs), 1))
    mesh.vs += mesh.vn * noise
    return mesh

def poisson_noise(mesh, lam, plevel):
    noise = plevel * np.random.poisson(lam=lam, size=(len(mesh.vs), 1)) / lam
    mesh.vs += mesh.vn * noise
    return mesh

def salt_pepper_noise(mesh, magnitude, percent):
    # select members
    n = mesh.vs.shape[0]
    idx = np.random.choice(n, int(percent * mesh.vs.shape[0]))
    n = len(idx)//2
    mesh.vs[idx[:n]] += magnitude
    mesh.vs[idx[n:]] -= magnitude
    return mesh

def salt_pepper_surface_noise(mesh, magnitude, percent):
    # select members
    n = mesh.vs.shape[0]
    idx = np.random.choice(n, int(percent * mesh.vs.shape[0]))
    n = len(idx)//2
    mesh.vs[idx[:n]] += mesh.vs[idx[:n]]*magnitude
    mesh.vs[idx[n:]] -= mesh.vs[idx[n:]]*magnitude
    return mesh

def mix_poisson_gauss_noise(mesh, lam, poisson_level, gaussian_level):
    tmesh = gausian_noise(mesh,gaussian_level)
    return poisson_noise(tmesh, lam, poisson_level)

def periodic_sin_noise(mesh, frequency=10, amplitude=1.0):
    # Create a sinus pattern based on vertex indices
    sin_noise = amplitude * np.sin(frequency * np.linspace(0, 2 * np.pi, len(mesh.vs)).reshape(-1, 1))
    mesh.vs += mesh.vn * sin_noise
    return mesh

def space_periodic_sin_noise(mesh, frequency=1, amplitude=1.):
    sin_noise = np.zeros(mesh.vs.shape)
    sin_noise[:,0] = amplitude*np.sin(frequency*mesh.vs[:,0])
    sin_noise[:,1] = amplitude*np.sin(frequency*mesh.vs[:,1])
    sin_noise[:,2] = amplitude*np.sin(frequency*mesh.vs[:,2])
    mesh.vs += mesh.vn * sin_noise
    return mesh

def main():
    args = get_parser()

    ms = ml.MeshSet()
    root_dir = os.path.dirname(args.input)
    mesh_name = os.path.splitext(os.path.basename(args.input))[0]

    #Create directory for ground truth
    gt_dir = os.path.join(root_dir, "original")
    os.makedirs(gt_dir, exist_ok=True)

    #Define ground truth file 
    g_file = os.path.join(gt_dir, f"{mesh_name}.obj") 

    #Save preprocessed ground truth
    ms.load_new_mesh(args.input)
    normalize(ms)
    ms.save_current_mesh(g_file)
    g_mesh = Mesh(g_file)
    g_mesh = edge_based_scaling(g_mesh)
    g_mesh.compute_face_normals()
    g_mesh.save(g_file)

    #Define noise directories
    noise_types = ["gaussian", "poisson", "saltpepper","saltpeppersurface", "mix", "sin","spacesin"]
    noise_dirs = {noise: os.path.join(root_dir, noise) for noise in noise_types}
    for noise_dir in noise_dirs.values():
        os.makedirs(noise_dir, exist_ok=True)

    #Process each noise type
    noise_functions = [
        ("gaussian", lambda mesh: gausian_noise(mesh, args.glevel)),
        ("poisson", lambda mesh: poisson_noise(mesh, args.lam, args.plevel)),
        ("saltpepper", lambda mesh: salt_pepper_noise(mesh, args.spmagnitude, args.sppercent)),
        ("saltpeppersurface", lambda mesh: salt_pepper_surface_noise(mesh, args.spfmagnitude, args.sppercent)),
        ("mix", lambda mesh: mix_poisson_gauss_noise(mesh, args.lam, args.plevel, args.glevel)),
        ("sin", lambda mesh: periodic_sin_noise(mesh)),
        ("spacesin", lambda mesh: space_periodic_sin_noise(mesh))
    ]

    for noise_name, noise_func in noise_functions:
        noise_dir = noise_dirs[noise_name]

        #Define filenames
        noisy_file = os.path.join(noise_dir, f"{mesh_name}_noise.obj")
        smooth_file = os.path.join(noise_dir, f"{mesh_name}_smooth.obj")
        gt_file_in_noise_dir = os.path.join(noise_dir, f"{mesh_name}_gt.obj")

        #Create noisy mesh
        noisy_mesh = Mesh(g_file)
        noisy_mesh = noise_func(noisy_mesh)
        noisy_mesh.compute_face_normals()
        noisy_mesh.save(noisy_file)

        #Load and smooth noisy mesh
        ms.load_new_mesh(noisy_file)
        smooth(ms, step=args.step)
        ms.save_current_mesh(smooth_file)

        #Copy ground truth file
        if not os.path.exists(gt_file_in_noise_dir):
            shutil.copy(g_file, gt_file_in_noise_dir)



if __name__ == "__main__":
    main()
