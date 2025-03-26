import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
import csv
import glob

import util.loss as Loss
import util.datamaker as Datamaker
from util.mesh import Mesh
from util.networks import PosNet, NormalNet
from util.early_stopping import EarlyStopping

def get_parser():
    #create add argument --earlystopping type bool
    parser = argparse.ArgumentParser(description='Dual Deep Mesh Prior')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-es', '--early_stopping', type=bool, required=True)
    parser.add_argument('--forget_a', type=float, default=0.1) #early stopping variable, nicht rausschmeißen
    parser.add_argument('--stagnate_p', type=int, default=100) #early stopping variable
    parser.add_argument('--pos_lr', type=float, default=0.01)
    parser.add_argument('--norm_lr', type=float, default=0.01)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument("--k1", type=float, default=3.0)
    parser.add_argument("--k2", type=float, default=4.0)
    parser.add_argument("--k3", type=float, default=4.0)
    parser.add_argument("--k4", type=float, default=4.0)
    parser.add_argument("--k5", type=float, default=1.0)
    parser.add_argument('--grad_crip', type=float, default=0.8)
    parser.add_argument('--bnfloop', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print('{:12s}: {}'.format(k, v))
    
    return args

def main():
    args = get_parser()
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

        # Subfolder for output
        output_folder = os.path.join(noise_folder_path, "output")
        os.makedirs(output_folder, exist_ok=True)

        # create data
        mesh_dic, dataset = Datamaker.create_dataset(noise_folder_path)
        gt_mesh, n_mesh, o1_mesh = mesh_dic["gt_mesh"], mesh_dic["n_mesh"], mesh_dic["o1_mesh"]


        # Initialize models and optimizers
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        posnet = PosNet(device).to(device)
        normnet = NormalNet(device).to(device)
        optimizer_pos = torch.optim.Adam(posnet.parameters(), lr=args.pos_lr)
        optimizer_norm = torch.optim.Adam(normnet.parameters(), lr=args.norm_lr)

        #Create CSV data
        csv_file_path = os.path.join(output_folder, f"{noise_folder}_loss_history.csv")
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            #write header
            if gt_mesh != None:
                csv_writer.writerow(["Epoch", "Loss", "MAD", "PSNR","EMV"])
            else:
                csv_writer.writerow(["Epoch", "Loss"])

        early_stopper = EarlyStopping(args.forget_a, args.stagnate_p)
        
        # Training Loop
        with tqdm(total=args.iter, desc=f"Training {noise_folder}") as pbar:
            for epoch in range(1, args.iter + 1):
                posnet.train()
                normnet.train()
                optimizer_pos.zero_grad()
                optimizer_norm.zero_grad()

                #Forward pass and loss calculation
                pos = posnet(dataset)
                loss_pos1 = Loss.pos_rec_loss(pos, n_mesh.vs)
                loss_pos2 = Loss.mesh_laplacian_loss(pos, n_mesh)

                norm = normnet(dataset)
                loss_norm1 = Loss.norm_rec_loss(norm, n_mesh.fn)
                loss_norm2, new_fn = Loss.fn_bnf_loss(pos, norm, n_mesh, loop=args.bnfloop)

                if epoch <= 100:
                    loss_norm2 = loss_norm2 * 0.0

                loss_pos3 = Loss.pos_norm_loss(pos, norm, n_mesh)

                loss = (
                    args.k1 * loss_pos1
                    + args.k2 * loss_pos2
                    + args.k3 * loss_norm1
                    + args.k4 * loss_norm2
                    + args.k5 * loss_pos3
                )
                loss.backward()
                nn.utils.clip_grad_norm_(normnet.parameters(), args.grad_crip)
                optimizer_pos.step()
                optimizer_norm.step()

                #update progress
                pbar.set_postfix({"loss": loss.item()})

                # Calculate MAD value
                # MAD = Mean Angular Deviation
                # evaluate how closely the predicted/optimized normals align with ground-truth normals
                # lower MAD indicates that  model has successfully reconstructed the geometry and normals of the original mesh
                o1_mesh.vs = pos.to("cpu").detach().numpy().copy()
                Mesh.compute_face_normals(o1_mesh)

                # save loss in CSV
                with open(csv_file_path, mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    if gt_mesh != None:
                        new_pos = pos.to("cpu").detach().numpy().copy()
                        o1_mesh.vs = new_pos
                        Mesh.compute_face_normals(o1_mesh)
                        Mesh.compute_vert_normals(o1_mesh)
                        csv_writer.writerow([epoch, loss.item(),Loss.mad(o1_mesh.fn, gt_mesh.fn),Loss.psnr(gt_mesh,o1_mesh),early_stopper.emv])
                    else:
                        csv_writer.writerow([epoch, loss.item()])

                #save intermediate steps
                if epoch == 10 or epoch % 100 == 0:
                    o1_mesh.vs = pos.to('cpu').detach().numpy().copy()
                    o_path = os.path.join(output_folder, f"{epoch}_ddmp.obj")
                    Mesh.save(o1_mesh, o_path)

                # Early stopping calculation here
                if args.early_stopping:
                    if early_stopper.early_stopping_algorithm(o1_mesh, pos, output_folder, epoch, Mesh):
                        break
                
                pbar.update(1)
    
if __name__ == "__main__":
    main()
