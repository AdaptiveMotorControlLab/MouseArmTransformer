# Utilities for lifting 2D DLC keypoints into 3D mouse-arm space

This directory contains the `MouseArmTransformer` module along with prototyping code and notebooks.
The relevant files are contained in the `MouseArmTransformer` directory. Right now, it is possible
to do a minimal install for loading model weights and performing inference.

![3d-dlc](https://github.com/AdaptiveMotorControlLab/MouseArmTransformer/assets/28102185/cce98f67-9ef8-48fb-b1c8-6cc34aafdd1f)
**Figure caption:**
**A:** Example 2D images (as in DeWolf, Schneider et al. Figure 1) of DeepLabCut keypoints on both camera views. 
**B:** Using 3D DeepLabCut (Nath, Mathis et al. 2019) we triangulated data, then manually labeled frames for GT. This was then all passed to a transformer (see below), and video inference was done with a single camera (Camera 1) to directly predict 3D.
**C:** Example lifted 3D pose, showing a sequence of reaching; black is early, red is late (matching the view in Camera 1), into the 3D MuJoCo spaces
**D:** Quantification of errors in 3 example sessions (vs. GT) in MuJoCo space. Note, the forearm is 1.33572 cm in our model, and 1 cm equals 1 MuJoCo unit.


## Building the python wheels

You can build a python package by running:

```
make build
```

You can check the contents of that build by running:

```
make test_contents
```

The wheels and source distributions are located in the `dist/` directory, which will look
sth like this:

```
dist/                                           
dist/MouseArmTransformer-0.1.0-py3-none-any.whl 
dist/MouseArmTransformer-0.1.0.tar.gz           
```

## Testing the code

If you want to run tests (in `tests/`) in a fully-configured docker environment, simply run:

```
make test_docker
```

which will build the container, the package, and then run the tests.

If you have a suitable python environment set up, you can also run:

```
make test
```

to run the tests.


# The transformer model

For estimating 3D coordinates from the two available camera views we first use simple triangulation to obtain 3D estimates, then correct outliers in a GUI to obtain 3D ground truth. The GUI is located in mausspaun/visualization/gui. For usage refer to the gui.md file within this folder. 

The transformer model takes as input the 2D DLC extracted coordinates from one camera and extracts the 3D coordinates through a linear output layer. 
The input to the transformer consists of sequences (of length T=2) of 2D coordinates representing the marker positions (see the Dataloader classes within data.py). Each marker is represented by an x and y position. We then encode the flattened joint coordinates using a transformer layer and then project the encoder output to a three-dimensional space using a fully connected layer. We train the model using four different losses and their corresponding weights:
- Triangulation loss: The mean-squared-error (MSE) between the model output and the noisy triangulated data (weight: 1)
- Continuity loss: The MSE between successive timesteps, which ensures temporal smoothness between consecutive output frames (weight: 25)
- Connectivity loss: The mean squared prediction error (MSPE) between markers, using a skeleton model of the mouse. This encourages the model to preserve the geometric relationship between joints during model training (weight: 1)
- Ground truth loss: Additionally we use the MSE between the model output and our ground truth 3D predictions  (see above) to provide the model with noise-free triangulation results. This loss was only active within batches that contained ground truth frames (weight: 0.0001)

We train the model using the above-described loss terms across all sessions for which two cameras are available. We then use the resulting weights to generate 3D predictions for sessions with only one camera. For training specifics refere to the training.py file within this folder.

![Screen Shot 2024-07-18 at 12 58 14 PM](https://github.com/user-attachments/assets/7ebfa345-d634-4f89-bd51-d592b24c6a3c)


## Versioning

To update the version, **please update
the version number** of this code. This can be done in `MouseArmTransformer/__init__.py`. 

You also need to edit the `Makefile` and change this part

```
tests/contents.tgz.lst:                                                              
        tar tf dist/MouseArmTransformer-0.0.1.tar.gz | sort > tests/contents.tgz.lst 
```

to reflect the new version information for the tests to pass.

## Who to contact for questions

- Mackenzie (@MMathisLab) oversees the project and trained the 2D DeepLabCut models.
- Markus (@CYHSM) wrote the majority of the `MouseArmTransformer` contents, and trained the 3D model.
  He also wrote the explorative notebooks and validated the model.
- Wesley (@wesleymth) contributed code inputs and 3D ground truth labels.
- Travis (studywolf) contributed 3D labels.
- Steffen (@stes) packaged the code and provided code review.

**email:** mackenzie.mathis@epfl.ch


## Citation 

If you use this code or ideas, please cite our work 🤗

Frey, M., Monteith-Finas, W., DeWolf, T., Schneider, S., & Mathis, M. W. (2024). MouseArmTransformer: a transformer-based 3D lifting module for an adult mouse arm (0.1.0). Zenodo. https://doi.org/10.5281/zenodo.12673173
