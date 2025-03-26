import pyvista as pv
import open3d as o3d
import numpy as np
import os
import argparse
import uuid
from math import ceil

class ScreenshotHandler:
    def __init__(self, plotter, path_no_ext):
        self.plotter = plotter
        self.path_no_ext = path_no_ext
        self.number = 0
        self.id = uuid.uuid4()
        
    def __call__(self):
        path = f"{self.path_no_ext}/{self.id}_{self.number:03d}.png"
        if not os.path.exists(self.path_no_ext):
            os.makedirs(self.path_no_ext)
        self.plotter.screenshot(filename=path)
        self.number += 1

def mad(norm1, norm2):
    inner = np.sum(norm1 * norm2, 1)
    sad = np.rad2deg(np.arccos(np.clip(inner, -1.0, 1.0)))

    return sad

def find_closest(l, k):
    return l[min(range(len(l)), key=lambda i : abs(l[i]-k))]

def find_best_model(path,stagnate_p=-1):
    fs = [int(file.split("_")[0]) for file in os.listdir(path) if file[-4:]==".obj"]
    # not right but unlikely to fail
    am = 0
    for epoch in fs:
        if epoch !=10 and epoch % 100:
            am = epoch
            break
    ## TODO BETTER WAY TO DEAL WITH BEST MODEL
    if am==0:
        am = max(fs)
    if stagnate_p != -1 and am == 0:
        am = max(fs) - stagnate_p
        if os.path.exists(os.path.join(path,f"{am}_ddmp.obj")):
            am = find_closest(fs,am)

    return f"{am}_ddmp.obj"

def visualize_diffs(root_input_dir,save_path,config):
    obj_name = root_input_dir.split(r"/")[-1]
    # path to ground truth mesh
    gt_path = os.path.join(root_input_dir,f"{obj_name}.obj")
    assert os.path.exists(gt_path), f"The following file should exist {gt_path}"
    # read mesh compute the face normals
    gt_mesh = o3d.io.read_triangle_mesh(gt_path)
    gt_mesh.compute_triangle_normals()
    gt_mesh = np.asarray(gt_mesh.triangle_normals)
    files = []
    for noise_folder in os.listdir(root_input_dir):      
        noise_folder_path = os.path.join(root_input_dir,noise_folder)
        if not os.path.isdir(noise_folder_path) or noise_folder=="original":
            continue
        noise_folder_path_output = os.path.join(noise_folder_path, "output")
        # find best model with consideration of stagnate_p
        best_model = find_best_model(noise_folder_path_output,-1)
        files.append((
                os.path.join(noise_folder_path_output,best_model),
                noise_folder,
                best_model.split("_")[0]
            )
        )

    # angles are the angular differences between both models
    angles = []
    for i in range(len(files)):
        obj_mesh = o3d.io.read_triangle_mesh(files[i][0])
        obj_mesh.compute_triangle_normals()
        angles.append(mad(np.asarray(obj_mesh.triangle_normals),gt_mesh))

    angles = np.array(angles)

    # Create a Lookup Table (LUT) for the 'jet' colormap
    lut = pv.LookupTable()
    lut.apply_cmap("jet")  # Apply the 'jet' colormap
    lut.scalar_range = (0, angles.max())  # Define the color scale range
    print("Max Angle: ",angles.max())
    c = 4 # number of columns
    pl = pv.Plotter(shape=(ceil((len(files)+1)/c),c),border=False)
    
    scalar_bar_args = {
    "n_labels": 3,  # Number of labels
    "fmt": "%0.0f°",  # Format (one decimal place)
    "color": "black",  # Font color
    }
    
    for i in range(len(files)):       
        pl.subplot((i+1)//c,(i+1)%c)
        pl.add_mesh(pv.read(files[i][0]), cmap=lut, scalars=angles[i], show_scalar_bar=True,scalar_bar_args=scalar_bar_args)
        pl.add_text(
            text=f"{files[i][1]} #{files[i][2]}\n{angles[i].mean():.2f} °",
            font_size=12,
            color="black",
            shadow=True
        )

    pl.subplot(0,0)
    pl.add_mesh(pv.read(gt_path), color="lightblue")

    handler = ScreenshotHandler(pl, save_path)
    pl.add_key_event('s', handler)
    description = "`s`=screenshot\n`v`=toggle visibility"
    actor = pl.add_text(description,font_size=12)

    def toggleVis():
        actor.set_text("upper_left","")
    pl.add_key_event('v', toggleVis)

    
    pl.link_views()
    pl.show()

def get_parser():
    parser = argparse.ArgumentParser(description='Visualiser for Mean Angular Differences on the Meshes')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default="./diffs")
    args = parser.parse_args()
    
    for k, v in vars(args).items():
        print('{:12s}: {}'.format(k, v))
    
    return args

def main():
    # get arguments as a dictionary
    args = get_parser()
    # sanitize path
    path = os.path.normpath(args.input).replace("\\","/")
    oPath = os.path.normpath(args.output).replace("\\","/")
    visualize_diffs(path,oPath,None)

if __name__ == "__main__":
    main()
