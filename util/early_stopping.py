import os
import numpy as np

class EarlyStopping:
    def __init__(self, forget_a, stagnate_p):
        self.forget_a = forget_a
        self.stagnate_p = stagnate_p
        self.stagnate_counter = 0
        self.ema = 0
        self.emv = 0
        self.emv_min = float("inf")
        self.x_opt = None

    def early_stopping_algorithm(self, o1_mesh, pos, output_folder, epoch, Mesh):
        self.stagnate_counter += 1
        
        
        # ema and emv will be added/subtracted from np.array
        self.emv = (1 - self.forget_a) * self.emv + self.forget_a * (1 * self.forget_a) * np.linalg.norm(o1_mesh.vs - self.ema,ord=2)
        self.ema = (1 - self.forget_a) * self.ema + self.forget_a * o1_mesh.vs
        
        if self.emv < self.emv_min: # save new min and make copy
            self.emv_min = self.emv
            self.x_opt = o1_mesh
            self.stagnate_counter = 0

        # when quitting save final mesh
        if self.stagnate_counter == self.stagnate_p:
            o1_mesh.vs = pos.to('cpu').detach().numpy().copy()
            o_path = os.path.join(output_folder, f"{epoch}_ddmp.obj")
            Mesh.save(self.x_opt, o_path)
            return True  # Signal to stop training
        return False  # Continue training
