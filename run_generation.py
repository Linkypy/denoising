import subprocess
from os import listdir

cad_models = set(["block","fandisk","nut","part","pyramid","trim-star"])
non_cad_params = ["--k1","3", "--k2", "4", "--k3", "4", "--k4", "4", "--k5", "1"]
cad_params = ["--k1", "3", "--k2", "0", "--k3", "3", "--k4", "4", "--k5", "2"]


for file in listdir("./datasets/"):
    subprocess.run(["python","preprocess/noisemaker.py", "-i", f"datasets/{file}/{file}.obj"])
    subprocess.run(["python","main4real.py", "-i", f"datasets/{file}/"] + (cad_params if file in cad_models else non_cad_params))
