from numpy import genfromtxt,zeros,argmin
import os
import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import textwrap


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


#################################
# Data storage
#################################

class Model:
    def __init__(self,name="nonamegiven"):
        self._name = name
        self._mads = {}
        self._losses = {}
        """
        noises = [
            "gaussian",
            "mix",
            "poisson",
            "saltpepper",
            "saltpeppersurface",
            "sin",
            "spacesin"
        ]
        for noise in noises:
            self._mads[noise] = None
            self._losses[noise] = None
        """
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name

    @property
    def losses(self):
        return self._losses
    
    @losses.setter
    def losses(self,losses):
        self._losses = losses

    @property
    def mads(self):
        return self._mads
    
    @mads.setter
    def mads(self,mads):
        self._mads = mads

    def addCSV(self, key, training_result):
        if training_result.shape[1] >= 3:
            self._mads[key] = training_result[:,2]
            self._losses[key] = training_result[:,1]
        elif training_result.shape[1] == 2:
            self._losses[key] = training_result[:,1]
        else:
            raise Exception("A csv did not have losses or mads")

    def meanMads(self):
        # find minimum mad, because we can only gurantee
        # proper mean calculation for that number of iterations
        n = float("inf")
        for mad in self._mads.values():
            n = min(n,len(mad)) if len(mad) != 0 else n

        if n != float("inf"):
            mean_mad = zeros(n)
            i = 0 # if not all measurements are given, we need to check for that
            for mad in self._mads.values():
                if len(mad) != 0:
                    mean_mad += mad[:n]
                    i += 1
            mean_mad /= i   
        else:
            mean_mad = []
        return mean_mad
    
    def meanLoss(self):
                # find minimum mad, because we can only gurantee
        # proper mean calculation for that number of iterations
        n = float("inf")
        for loss in self._losses.values():
            n = min(n,len(loss)) if len(loss) != 0 else n
        if n != float("inf"):
            mean_loss = zeros(n,dtype=float)
            i = 0 # if not all measurements are given, we need to check for that
            for loss in self._losses.values():
                if len(loss) != 0:
                    mean_loss += loss[:n]
                    i+=1
            mean_loss /= i
        else:
            mean_loss = []
        return mean_loss
            

########################
# Plotting
########################
"""
line = a data row
noise type = gaussian/poisson/mix/... noise
model = 3D-Mesh of an object
"""


"""
Plot lines for all models and all noise types.
Just for overview for the observer.
"""
plot_params = {
    "linewidth" : 0.8,
    "color" : mpl.colormaps["tab20"].colors,
    "markersize" : 17.,
    "figsize" : (10,5),
}


def plot_all_lines(ax1,csvs,access_method):
    for i,data_class in enumerate(csvs):
        for metric in access_method(data_class).values():
            if metric is None:
                continue
            ax1.plot(metric,color=plot_params["color"][i], linewidth=plot_params["linewidth"], label=data_class.name)

"""
Plot the average lines for all models.
Just for overview for the observer.
"""
def plot_average_lines(ax1,csvs,mode):
    for i,data_class in enumerate(csvs):       
        if mode=="loss":
            ax1.plot(data_class.meanLoss(),color=plot_params["color"][i], linewidth=plot_params["linewidth"], label=data_class.name)
        elif mode=="mad":
            ax1.plot(data_class.meanMads(),color=plot_params["color"][i], linewidth=plot_params["linewidth"], label=data_class.name)

"""
Plot for a noise type the line for all models.
"""
def plot_noise(ax1,ax2,csvs,mode,noise,access_method):
    minimums = []
    for i,data_class in enumerate(csvs):
        try:
            ax1.plot(access_method(data_class)[noise],color=plot_params["color"][i], linewidth=plot_params["linewidth"], label=data_class.name)
            minimums.append((data_class.name, argmin(access_method(data_class)[noise]), round(access_method(data_class)[noise].min(),2)))
            ax1.scatter(minimums[-1][1], minimums[-1][2],marker="x",s=plot_params["markersize"],color="black")
        except KeyError:
            print(f"Model {data_class.name} does not have {mode} for {noise}")
    
    # order results via their minimum metric
    minimums = sorted(minimums, key=lambda x : x[-1])

    if minimums != []:
        plot_table(ax2,minimums)
"""
For a model plot all the noise lines.
"""
def plot_model_all_lines(ax1,ax2,data_class,access_method):

    minimums = [(noise,argmin(metric), round(metric.min(),2)) for (noise,metric) in access_method(data_class).items()]
    #for i,(noise,metric) in enumerate(access_method(data_class).items()):
    for i,(noise,metric) in enumerate(access_method(data_class).items()):
        ax1.plot(metric, color=plot_params["color"][i], linewidth=plot_params["linewidth"], label=noise)
        ax1.scatter(minimums[i][1], minimums[i][2],marker="x", s=plot_params["markersize"],color="black")
    # order results via their minimum metric
    minimums = sorted(minimums, key = lambda x : x[-1])
    if minimums != []:
        plot_table(ax2,minimums)




"""
Create Plot description
"""
def plot_description(output, fig, ax1, ax2, title, ylabel,show):
    plt.sca(ax1)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    ax1.legend(by_label.values(), by_label.keys(),ncols=2)
    ax1.set_title(title, weight="bold")
    ax1.set_xlabel("# iteration")
    ax1.set_ylabel(ylabel)
    ax1.grid(True)

    if ax2:
        ax2.set_title(f"Ascending {ylabel}", weight="bold")

    fig.tight_layout()

    if not show:
        plt.savefig(output)
        plt.close()
    else:
        plt.show()

def plot_table(ax2,minimums):
        ax2.table(
            cellText = minimums,
            colLabels=["noise","ArgMin-iter","Min-Score"],
            loc="best",
            colWidths=[5./12.,4./12.,3./12.]
        )

# create figure
def cf(make_grid=False):
    # Two columns, wider for the plot
    if make_grid:
        grid = mpl.gridspec.GridSpec(1, 2, width_ratios=[2, 1])  
        fig = plt.figure(dpi=250,figsize=plot_params["figsize"])
        ax1 = fig.add_subplot(grid[0,0]) # plotting axis
        ax2 = fig.add_subplot(grid[0,1]) # text descriptor axis
        ax2.axis("off")

        return fig, ax1, ax2
    else:
        fig, ax1 = plt.subplots(dpi=150,figsize=plot_params["figsize"])
        return fig, ax1
    

def plot_metrics(csvs,args,mode,ylabel):
    if mode=="loss":
        access_method = lambda x : x.losses
    elif mode=="mad":
        access_method = lambda x : x.mads 
    else:
        Exception(f"Plot all lines invalid mode \"{mode}\"")
    
    wrap = lambda x,w : '\n'.join(textwrap.wrap(x,width=w))

    # cross model
    fig, ax1 = cf()
    plot_all_lines(ax1, csvs, access_method)
    plot_description(os.path.join(args.output, f"{mode}_all_cross.png"), fig, ax1, None, 
                     wrap(f"Different {ylabel} for all models and their noises", 40),
                        ylabel,args.show)
    
    # cross model average
    fig, ax1 = cf()
    plot_average_lines(ax1, csvs,mode)
    plot_description(os.path.join(args.output,f"{mode}_average_cross.png"), fig, ax1, None,
                       wrap(f"Different average {ylabel} for all models", 40),ylabel, args.show)

    # for every noise kind
    noises = set([])
    [noises.add(noise) for data_class in csvs for noise in data_class.losses.keys()]
    
    for noise in noises:
        fig, ax1, ax2 = cf(make_grid=True)
        plot_noise(ax1, ax2, csvs,mode,noise,access_method)
        plot_description(os.path.join(args.output,f"{mode}_{noise}_cross.png"), fig, ax1, ax2, 
                          wrap(f"Different {ylabel} for models for {noise} noise", 40),ylabel,args.show)
    

    for data_class in csvs:
        fig, ax1, ax2 = cf(make_grid=True)
        plot_model_all_lines(ax1, ax2, data_class, access_method)
        plot_description(os.path.join(args.output,data_class.name,f"{data_class.name}_{mode}.png"), fig, ax1, ax2, 
                         wrap(f"Different {ylabel} for the {data_class.name} model", 40),ylabel,args.show)


def create_plots(csvs,args):
    plot_metrics(csvs,args,"mad", "Mads")
    plot_metrics(csvs,args,"loss", "Losses")



########################
# main
########################


def get_parser():
    parser = argparse.ArgumentParser(description='create datasets(noisy mesh & smoothed mesh) from a single clean mesh')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default="metrics")
    parser.add_argument("-s", "--show",type=bool,default=False)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print('{:12s}: {}'.format(k, v))

    return args

def main():
    args = get_parser()


    csvs = []
    # read in directories and csvs
    root_input_dir = args.input  
    for obj_folder in os.listdir(root_input_dir):
        obj_folder_path = os.path.join(root_input_dir,obj_folder)
        if not os.path.isdir(obj_folder_path): # skip non directories
                continue
        # create data collection
        data_class = Model(obj_folder)
        for noise_folder in os.listdir(obj_folder_path):
            noise_folder_path = os.path.join(root_input_dir, obj_folder, noise_folder)
            # Only look at folders that are not named "original"
            if not os.path.isdir(noise_folder_path) or noise_folder == "original":
                continue
            
            csv_path = os.path.join(noise_folder_path,"output",f"{noise_folder}_loss_history.csv")
            if not os.path.exists(csv_path):
                print(f"Tried to access: {csv_path}\nCSV file with losses or entire directory missing.")
                continue
            # get header
            with open(csv_path,'r') as s:
                header = s.readline().split(',')
            # remove header from reading
            data = genfromtxt(csv_path,delimiter=",",skip_header=1)
            if "MAD" not in header: # prevent EMV accidentally becoming third colum and being interpreted as MAD
                data = data[:,:2]
            try:
                data_class.addCSV(noise_folder, data)
            except KeyError as e:
                raise KeyError(f"{str(e)} at {csv_path}")

        csvs.append(data_class)
    # create nessecary directories
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    for csv in csvs:
        path_model_folder = os.path.join(args.output,csv.name)
        if not os.path.exists(path_model_folder):
            os.mkdir(path_model_folder)

    # create plots
    create_plots(csvs,args)

    

if __name__ == "__main__":
    main()
