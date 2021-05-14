
#############################################
Installation of required environment packages.
#############################################

python3.6
tensorflow==1.15
#stable-baselines (install modified version from link below, this version includes a custom 3D CNN Policy for this project.
3D CNN is not otherwise available in stable-baselines) 
pip install git+https://github.com/bogglem/stable-baselines.git
sklearn

matplotlib (comes with stable baselines)
pandas (comes with stable baselines)
numpy (comes with tensorflow)

the MineSequencing module can be found at;
https://github.com/bogglem/MineSequencing.git


############################################
Training Parameter Setup.
############################################

Training parameters are imported from ./jobarrays/****.csv files.
In these parameter csv files, you can adjust the following parameters;

LR - Learning Rate. An effective value appears to be in the order of 0.0001.
--------
gamma - Discount Factor. 
Longer sequences require higher discount factor (close to 1) to ensure relevance of late actions is carried forward.
Poor selection of discount factor caused issues with propagating signals from later actions. This is a subject of the research.
--------
batch_size - varying batch size has an effect on the stability and convergence of the learning curve, similar to learning rate.
A batch sie around 64-128 gives reasonable results.
--------
runtime - training time (in seconds). 
--------
cutoffpenalty - a value that describes how much penalty will be given if the episode termination action is taken early. 
The longer the episode runs, the lower the cutoff penalty. The penalty is also related to the quantity of unmined ore remaining in the environment.
This parameter is a focus of research and development.
--------
rg_prob - probabilty that a new random environment will be generated in a new training episode.
If rg_prob is set to zero, this is the special case where training will only occur in 1 environment. 
In this case, the environment will be saved in the output folder.
--------
turnspc - max number of actions to be taken each episode (as a percentage of the block model size) eg. Max number of Actions = turnspc*block model size
--------
trialv - a text string used to describe the trial. Will be put on output files for easy identification.
--------

############################################
Training.
############################################

1. Activate python environment.

2. Navigate to MineSequencing folder.

3. Modify Train_RG_CNNA2C_drstrange.py to your desired parameters. (Several "low level" parameters are in the code, not the csv files)
You can change the importing parameter file via inputarray variable on line45
You can change the environment block model dimensions on line63-65.
You can change the GPU devices used on line39.
You can change the number of parallel processes using the num_cpu variable on line127.

4. execute Train_RG_CNNA2C_drstrange.py
The best agent will be periodically saved to the "output" folder along with learning curve and evaluation data. 
In the case where rg_prob = 0.0, the environment will also be saved.


############################################
Implementation and inspection of agent.
############################################


Once there is a Best_model.zip file available, it can be reloaded, implemented and qualitatively inspected via the render functions.

1. Modify CNNA2C_implement.py parameters on lines29-40 to match the desired folder and file where the best model is located. 
Note that the parameters must be exactly the same as used for training. 
------
When rg_prob is set to 0.0 the code will load the saved environment for this case.
------

2. execute CNNA2C_implement.py to see the agent acting on the environment in matplotlib windows. 
This is best viewed in ide software such as spyder.


