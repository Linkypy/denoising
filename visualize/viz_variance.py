import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import os
import glob
import argparse

# for finding the best model with early stopping
# early model usually has a mumber not divisble by 100
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

# read in data from loss_history.csv
def read_in_obj(root_input_dir, reconstructed_models_csvs):
    for noise_folder in os.listdir(root_input_dir):
            noise_folder_path = os.path.join(root_input_dir, noise_folder)
            
            # Only look at folders that are not named "original"
            if not os.path.isdir(noise_folder_path) or noise_folder == "original":
                continue
            
            # Unterordner-spezifischen Output-Pfad erstellen
            output_folder = os.path.join(noise_folder_path, "output")
            csv_model_path  = os.path.join(output_folder, f"{noise_folder}_loss_history.csv")
            # check if denoised objs exist as well as the csv
            assert os.path.exists(output_folder) and len(glob.glob(os.path.join(output_folder,"*_ddmp.obj")))>0 \
            and os.path.exists(csv_model_path), f"Missing .objs or csv in {output_folder}"
            best_model_name =  find_best_model(output_folder)
            #best_model_path = os.path.join(output_folder, find_best_model(output_folder))
            
    
            reconstructed_models_csvs.append((noise_folder,csv_model_path, int(best_model_name.split("_")[0])))      

# load data
def load_data(reconstructed_models_csvs,data_matrices):
    for noise_path in reconstructed_models_csvs:
        data = np.genfromtxt(noise_path[1], dtype=float,delimiter=",",skip_header=1)
        if data.shape[1] >=5: # 5 because 5 cols
            data_matrices.append(data)

# the big diagram. Plot for all models the early stopping performance
def plot_model_ES(obj_folder,reconstructed_models_csvs,data_matrices,save_path):
    cols = 3
    # for calculation of grid with plots
    rows = int(np.ceil(len(data_matrices)/cols))
    fig, ax = plt.subplots(rows,cols,figsize=(10,8))
    ax = ax.ravel()
    
    for i,data in enumerate(data_matrices):
        ax[i].plot(data[:,0],data[:,3],label="PSNR")
        ax[i].plot(data[:,0],data[:,4],label="Variance")
        ax[i].plot(data[:,0],data[:,2],label="MAD")
        ax[i].set_title(reconstructed_models_csvs[i][0])
        min_iter = reconstructed_models_csvs[i][2]
        ax[i].scatter([min_iter], [data[min_iter,4]],marker="x",s=18.,color="black",label=f"Min #iter {min_iter}")
        ax[i].legend(loc="upper right")

        # so that big line does not shift the view limit and squish all other graphs
        if data[:,4].max() > 1.4*data[:,3].max():
            #yticks = np.linspace(0,data[:,3].max(),6)
            #ax[i].set_yticks(yticks)
            ax[i].set_ylim(0,data[:,3].max()*1.1)
            
        
    for i in range(len(data_matrices),cols*rows):
        ax[i].axis("off")
    plt.suptitle(obj_folder,weight="bold")
    
    plt.tight_layout()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path,f"{obj_folder}_ES_visual.png"))

    plt.figure(figsize=(10,5))
    c = mpl.colormaps["tab20"].colors
    for i,data in enumerate(data_matrices):
        
        plt.title(obj_folder,weight="bold")
        min_iter = reconstructed_models_csvs[i][2] # reconstructed_model_csvs is the read in csv file
        if not i: # plot label for first iteration
            plt.scatter([data[min_iter-1,0]], [data[min_iter-1,2]],marker="x",color="black",zorder=-0.5,label="Early Stopping iter")
        else:
            plt.scatter([data[min_iter-1,0]], [data[min_iter-1,2]],marker="x",color="black",zorder=-0.5)
        
        plt.plot(data[:,0],data[:,2],color=c[i],label=reconstructed_models_csvs[i][0],zorder=-1.0)
    plt.legend()
    plt.grid(True)   
    plt.xlabel("#iterations")
    plt.ylabel("MAD Scores")
    plt.savefig(os.path.join(save_path,f"{obj_folder}_stop.png"))
    plt.show()

def plot_full_iter(obj_folder,ES_reconstructed_models_csvs,ES_data_matrices,regular_reconstructed_models_csvs,regular_data_matrices):
    plt.figure(figsize=(10,5))
    c = mpl.colormaps["tab20"].colors
    for i,(data,data_r) in enumerate(zip(ES_data_matrices,regular_data_matrices)):
        plt.plot(data[:,0],data[:,2],color=c[i],label=ES_reconstructed_models_csvs[i][0],zorder=-1.0)
        plt.title(obj_folder,weight="bold")
        min_iter = ES_reconstructed_models_csvs[i][2]
        plt.scatter([min_iter], [data[min_iter,2]],marker="x",color="black",zorder=-0.5)
        n = len(data[:,0]) -1
        plt.plot(data_r[n:,0],data_r[n:,2],color=c[i],linestyle="--",zorder=-1.0)

    plt.ylabel("MAD Scores")
    plt.xlabel("#iterations")
    plt.legend()
    plt.grid(True)    
    plt.savefig(os.path.join(save_path,f"{obj_folder}_full_iter.png"))
    plt.show()

def plot_all_modelsES(root_input_dir, save_path):

     for obj_folder in os.listdir(root_input_dir):
        obj_folder_path = os.path.join(root_input_dir,obj_folder)
        if not os.path.isdir(obj_folder_path):        
            continue
        reconstructed_models_csvs = []
        data_matrices = []
        read_in_obj(obj_folder_path, reconstructed_models_csvs)
        load_data(reconstructed_models_csvs, data_matrices)
        if len(data_matrices): # atleast one entry
            plot_model_ES(obj_folder,reconstructed_models_csvs, data_matrices,save_path)

def get_parser():
    parser = argparse.ArgumentParser(description='Visualiser for Variances and PSNR')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default="./variances")
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
    plot_all_modelsES(path,oPath)


if __name__=="__main__":
    main()

    
    
        
