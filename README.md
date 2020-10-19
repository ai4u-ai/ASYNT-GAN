# ASYNT-GAN
This is the code for the paper De Novo Drug Design using Artificial Intelligence ASYNT-GAN
Computer-assisted de novo design of natural product mimetics offers a viable strategy to reduce synthetic efforts and obtain natural-product-inspired bioactive small molecules but suffers from several limitations. Deep Learning techniques can help address these shortcomings. We propose the generation of synthetic molecule structures that optimizes the binding affinity to a target. To achieve this, we leverage on important advancements in Deep Learning. Our approach generalizes to systems beyond the source system and achieves generation of complete structures that optimize the binding to a target unseen during training. Translating the input sub-systems into the latent space permits the ability to search for similar structures and the sampling from the latent space for generation.


###### Install TF.GRAPHICS

```
git clone https://github.com/tensorflow/graphics.git
cd graphics
```
Windows:

Download  and install OpenEXR
https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr  

`pip install path_to_whl`

Install tf graphics
`pip install -e . --user`

###### Install Pymol

https://www.lfd.uci.edu/~gohlke/pythonlibs/#pymol-open-source

###### PREPARE DATA
Create and activate an anaconda env.
Install the requirements 
`pip install -r requirements.txt`

The file rcsb_pdb_ids_20200628065205.txt holds ids of pdb files related to covid-19 pandemic.
The convert_to_wrl_files_pymol.py will fetch all the pdb files and split them into ligands and proteins, centralize them and create .wrl files for each ligand and chain in a particular protein.
The convert_to_ply_blender.py will iterate over the files in the data/to_convert folder and create ply files in the converted folder

###### Train
The ASYNT-GAN.py  will sample from converted files train and save the model checkpoints to tf_ckpts_v2 folder and tensorboard logs to logs3d folder.
to render tensorboard 
`tensorboard --logdir logs3d`
The port url and port will be shown in the cmd