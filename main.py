import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from tqdm import tqdm
import viser
import viser.transforms as tf
import util.loss as Loss
import util.models as Models
import util.datamaker as Datamaker
from util.mesh import Mesh
from util.networks import PosNet, NormalNet
import csv
import glob
from util.early_stopping import EarlyStopping

def get_parser():
    parser = argparse.ArgumentParser(description="Dual Deep Mesh Prior")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument('-es', '--early_stopping', type=bool, required=True)
    parser.add_argument('--forget_a', type=float, default=0.1) #early stopping variable
    parser.add_argument('--stagnate_p', type=int, default=100) #early stopping variable
    parser.add_argument("--pos_lr", type=float, default=0.01)
    parser.add_argument("--norm_lr", type=float, default=0.01)
    parser.add_argument("--norm_optim", type=str, default="Adam")
    parser.add_argument("--iter", type=int, default=1000)
    parser.add_argument("--k1", type=float, default=3.0)
    parser.add_argument("--k2", type=float, default=4.0)
    parser.add_argument("--k3", type=float, default=4.0)
    parser.add_argument("--k4", type=float, default=4.0)
    parser.add_argument("--k5", type=float, default=1.0)
    parser.add_argument("--grad_crip", type=float, default=0.8)
    parser.add_argument("--bnfloop", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--viewer", action="store_true", default=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print("{:12s}: {}".format(k, v))
    
    return args

def main():
    args = get_parser()
    
    if args.viewer:
        server = viser.ViserServer(port=args.port)

    # Iterate directories
    root_input_dir = args.input  
    for noise_folder in os.listdir(root_input_dir):
        noise_folder_path = os.path.join(root_input_dir, noise_folder)
        
        # Only look at folders that are not named "original"
        if not os.path.isdir(noise_folder_path) or noise_folder == "original":
            continue

        # Check required files in the noise folder
        required_files = ['_noise.obj', '_smooth.obj']
        available_files = glob.glob(os.path.join(noise_folder_path, '*.obj'))
        missing_files = [file for file in required_files if not any(file in f for f in available_files)]  

        if missing_files:
            print(f"Error: Missing required files in {noise_folder_path}: {', '.join(missing_files)}")
            continue
        print("All required files found. Proceeding with training.")
        print("Folder" + noise_folder_path)
        # Subfolder output
        output_folder = os.path.join(noise_folder_path, "output")
        os.makedirs(output_folder, exist_ok=True)

        # create dataset 
        mesh_dic, dataset = Datamaker.create_dataset(noise_folder_path)
        mesh_name = mesh_dic["mesh_name"]
        gt_mesh, n_mesh, o1_mesh = mesh_dic["gt_mesh"], mesh_dic["n_mesh"], mesh_dic["o1_mesh"]

        # create model instance 
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        posnet = PosNet(device).to(device)
        normnet = NormalNet(device).to(device)
        optimizer_pos = torch.optim.Adam(posnet.parameters(), lr=args.pos_lr)
        optimizer_norm = torch.optim.Adam(normnet.parameters(), lr=args.norm_lr)
        
        os.makedirs("datasets/" + mesh_name + "/output", exist_ok=True)

        # initial condition
        if gt_mesh != None:
            init_mad = mad_value = Loss.mad(n_mesh.fn, gt_mesh.fn)
            print("initial_mad: {:.3f}".format(init_mad))
        else:
            init_mad = 0.0

        scale = 2 / np.max(n_mesh.vs)
        if args.viewer:
            print("\n\033[42m Viewer at: http://localhost:{} \033[0m\n".format(args.port))
            with server.gui.add_folder("Training"):
                server.scene.add_mesh_simple(
                    name="/input",
                    vertices=n_mesh.vs * scale,
                    faces=n_mesh.faces,
                    flat_shading=True,
                    visible=False,
                )
                gui_counter = server.gui.add_number(
                    "Epoch",
                    initial_value=0,
                    disabled=True,
                )
                gui_mad = server.gui.add_number(
                    "MAD",
                    initial_value=init_mad,
                    disabled=True,
                )

        csv_file_path = os.path.join(output_folder, f"{noise_folder}_loss_history.csv")
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            #write header
            if gt_mesh != None:
                csv_writer.writerow(["Epoch", "Loss", "MAD"])
            else:
                csv_writer.writerow(["Epoch", "Loss"])

        early_stopper = EarlyStopping(args.forget_a, args.stagnate_p)

        # Training loop 
        with tqdm(total=args.iter) as pbar:
            for epoch in range(1, args.iter+1):
                posnet.train()
                normnet.train()
                optimizer_pos.zero_grad()
                optimizer_norm.zero_grad()

                pos = posnet(dataset)
                loss_pos1 = Loss.pos_rec_loss(pos, n_mesh.vs)
                loss_pos2 = Loss.mesh_laplacian_loss(pos, n_mesh)

                norm = normnet(dataset)
                loss_norm1 = Loss.norm_rec_loss(norm, n_mesh.fn)
                loss_norm2, _ = Loss.fn_bnf_loss(pos, norm, n_mesh, loop=args.bnfloop)
                
                if epoch <= 100:
                    loss_norm2 = loss_norm2 * 0.0

                loss_pos3 = Loss.pos_norm_loss(pos, norm, n_mesh)

                loss = args.k1 * loss_pos1 + args.k2 * loss_pos2 + args.k3 * loss_norm1 + args.k4 * loss_norm2 + args.k5 * loss_pos3
                loss.backward()
                nn.utils.clip_grad_norm_(normnet.parameters(), args.grad_crip)
                optimizer_pos.step()
                optimizer_norm.step()

                pbar.set_description("Epoch {}".format(epoch))
                pbar.set_postfix({"loss": loss.item()})
                
                vs_update = False

                if epoch % 10 == 0:
                    new_pos = pos.to("cpu").detach().numpy().copy()
                    o1_mesh.vs = new_pos
                    Mesh.compute_face_normals(o1_mesh)
                    Mesh.compute_vert_normals(o1_mesh)

                    if gt_mesh != None:
                        mad_value = Loss.mad(o1_mesh.fn, gt_mesh.fn)
                    else:
                        mad_value = 0.0

                    if epoch == 10 or epoch % 100 == 0:
                        o_path = os.path.join(output_folder, f"{epoch}_ddmp.obj")
                        Mesh.save(o1_mesh, o_path)
                    
                    if args.viewer:
                        server.scene.add_mesh_simple(
                            name="/output",
                            vertices=o1_mesh.vs * scale,
                            faces=o1_mesh.faces,
                            flat_shading=True,    
                        )
                        gui_counter.value = epoch
                        gui_mad.value = mad_value

                    if vs_update:
                        updated_pos = Models.vertex_updating(pos, norm, o1_mesh, loop=15)
                        o1_mesh.vs = updated_pos.to("cpu").detach().numpy().copy()
                        Mesh.compute_face_normals(o1_mesh)
                        #updated_mad = Loss.mad(o1_mesh.fn, gt_mesh.fn)
                        u_path = o_path = os.path.join(output_folder, f"{epoch}_ddmp.obj")
                        Mesh.save(o1_mesh, u_path)

                    # save loss in CSV
                    with open(csv_file_path, mode='a', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        if gt_mesh != None:
                            new_pos = pos.to("cpu").detach().numpy().copy()
                            o1_mesh.vs = new_pos
                            Mesh.compute_face_normals(o1_mesh)
                            Mesh.compute_vert_normals(o1_mesh)
                            csv_writer.writerow([epoch, loss.item(),Loss.mad(o1_mesh.fn, gt_mesh.fn)])
                        else:
                            csv_writer.writerow([epoch, loss.item()])

                # Early stopping calculation 
                if args.early_stopping:
                    if early_stopper.early_stopping_algorithm(o1_mesh, pos, output_folder, epoch, Mesh):
                        break
                pbar.update(1)

        print("final_mad: {:.3f}".format(mad_value)) #just works for synthetic data with ground truth or else MAD value will be 0.0


if __name__ == "__main__":
    main()