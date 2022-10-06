# DC3

Code for the paper: "[A data-centric approach for improving ambiguous labels with combined semi-supervised classification and clustering](https://arxiv.org/abs/2106.16209)" by Lars Schmarje,  Monty Santarossa, Simon-Martin Schröder, Claudius Zelenka, Rainer Kiko, Jenny Stracke, Nina Volkmann and Reinhard Koch

Datasets available [here](https://zenodo.org/record/5550917)

The code is extended / forked version of the official FixMatch repository. The full original README is given in `deprecated_README.md`:

## Setup

All code execution is designed to be run within docker. A working version of the latest Docker with GPU Support is expected.

create an image with docker file

    docker build -t dc3 .
    
All following commands should be executed within the created docker image.
You can create an appropriate container with

```
docker run -it --rm --gpus all --name dc3 --shm-size 16G -v <DIRECTORY_SOURCECODE>:/src -w /src  -v <DIRECTORY_DATASETS>:/data-ssd -v <DIRECTORY_LOGS>:/data1  dc3 bash
```

You need to update the paths to the three directories to appropriate locations on your device.


You need to download the datasets from [Zenodo](https://doi.org/10.5281/zenodo.5550916) and extract them into `<DIRECTORY_DATASETS>`.
You should now have 4 subfolders within the specified directory.
Create a fifth folder `fixmatch` inside this directory.

You need to import and preprocess these datasets now with `./generate_datasets.sh`. Remember to run it within the docker container you created above.
The generated files will be stored under the created folder `fixmatch`.
By default the GPU with the index 0 is used to run these elements. If you have no GPU or an unsupported device this might lead to errors.

## Running

You can run the experiments within the docker container with the commands below
```

    # Before you can start anything you need to export some enviroment variables for the fixmatch repository source code
    export ML_DATA=/data-ssd/fixmatch
    export PYTHONPATH=$PYTHONPATH:.


    # original source code SSL method e.g. mean teacher
    # this uses additional weighted cross entropy for unbalanced datasets
    
    python mean_teacher.py --filters 32 --dataset plankton.1@-1-10 --train_dir /data1/   --train_kimg 1024  --wou 0   --wol 0  --lr 0.01 --ceinv_labels 0  --arch resnet --augment d.d.d --combined_output 0  --wf 0  --ws 0   --prior_fuzzy_distribution 0 --prob_gradient 0 --use_loss_rescale 0 --use_soft_prob 0 --use_weighted_xe 1
    
    # dc3 in combination with SSL e.g. mean teacher
    # the parameters are explained in `libml/overclustering_training.py`
    
    python mean_teacher.py --filters 32 --dataset  plankton.1@-1-10 --train_dir /data1/   --train_kimg 1024 --wou 10   --wol 10  --lr 0.01 --ceinv_labels 1 --arch resnet --augment d.d.d --combined_output 4  --wf 0.1 --ws 0.1  --prob_gradient 0 --use_loss_rescale 0 --use_soft_prob 1 --prior_fuzzy_distribution 0.6 --use_weighted_xe 1
    
    # useable datasets, the last part (.1@-1-10) is a left over of the original fixmatch repository and can be ignored
    plankton.1@-1-10 miceBone.1@-1-10 turkey.1@-1-10 cifar10h.1@-1-10
    
    
    # supported methods, the other models could be added but are currently not modified
    ce (supervised baseline), fixmatch, mean_teacher, pseudo_label, pi_model
```

Additional notes for running the experiments:

- All experiments were run on a RTX 2080 Ti with 12 GB VRAM
- For MiceBone data were we used 4 GPUs and used the following different arguments `--train_kimg 256  --batch 8 --report_kimg 8 --filters 8 ` 
- For FixMatch we used for all datasets 6 GPUs in parallel.
- For the method pseudo_label the augmentation is d.d not d.d.d


## Citing this work

```bibtex
@Article{schmarje2022dc3,
AUTHOR = {Schmarje, Lars and Santarossa, Monty and Schröder, Simon-Martin and Zelenka, Claudius and Kiko, Rainer and Stracke, Jenny and Volkmann, Nina and Koch, Reinhard},
TITLE = {A data-centric approach for improving ambiguous labels with combined semi-supervised classification and clustering},
JOURNAL = {Proceedings of the European Conference on Computer Vision (ECCV)},
YEAR = {2022},
}
```
