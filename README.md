# Utilities for lifting 2D DLC keypoints into 3D mouse-arm space

This directory contains the `MouseArmTransformer` module along with prototyping code and notebooks.
The relevant files are contained in the `MouseArmTransformer` directory. Right now, it is possible
to do a minimal install for loading model weights and performing inference.

TODO: Extend the package to support training in python environments that contain the `mousearm` python
package.

## Building the python wheels

For integration, e.g. into datajoint, you can build a python package by running

```
make build
```

You can check the contents of that build by running

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

If you want to run tests (in `tests/`) in a fully-configured docker environment, simply run

```
make test_docker
```

which will build the container, the package, and then run the tests.

If you have a suitable python environment set up, you can also run

```
make test
```

to run the tests.

## Contributing

If you change pieces of the code and these should be merged to the main branch, **please update
the version number** of this code. This can be done in `MouseArmTransformer/__init__.py`. 

You also need to edit the `Makefile` and change this part

```
tests/contents.tgz.lst:                                                              
        tar tf dist/MouseArmTransformer-0.0.1.tar.gz | sort > tests/contents.tgz.lst 
```

to reflect the new version information for the tests to pass. The new wheel can then be shipped
to the datajoint pipeline, currently to this directory (please change the branch to the latest):

```
https://github.com/MMathisLab/DataJoint_mathis/tree/stes/new-dlc-and-alignment/docker/mousearm-docker
```


## Who to contact for questions

- Markus (@CYHSM) wrote the `MouseArmTransformer` contents, and trained the model.
  He also wrote the explorative notebooks etc. and validated the model.
- Steffen (@stes) packaged the code for integration into datajoint. More details on this
  are also written up here: https://github.com/MMathisLab/DataJoint_mathis/pull/87

# The transformer model

For estimating 3D coordinates from the two available camera views we first use simple triangulation to obtain 3D estimates, then correct outliers in a GUI to obtain 3D ground truth. The GUI is located in mausspaun/visualization/gui. For usage refer to the gui.md file within this folder. 

The transformer model takes as input the 2D DLC extracted coordinates from one camera and extracts the 3D coordinates through a linear output layer. 
The input to the transformer consists of sequences (of length T=2) of 2D coordinates representing the marker positions (see the Dataloader classes within data.py). Each marker is represented by an x and y position. We then encode the flattened joint coordinates using a transformer layer and then project the encoder output to a three-dimensional space using a fully connected layer. We train the model using four different losses and their corresponding weights:
- Triangulation loss: The mean-squared-error (MSE) between the model output and the noisy triangulated data (weight: 1)
- Continuity loss: The MSE between successive timesteps, which ensures temporal smoothness between consecutive output frames (weight: 25)
- Connectivity loss: The mean squared prediction error (MSPE) between markers, using a skeleton model of the mouse. This encourages the model to preserve the geometric relationship between joints during model training (weight: 1)
- Ground truth loss: Additionally we use the MSE between the model output and our ground truth 3D predictions  (see above) to provide the model with noise-free triangulation results. This loss was only active within batches that contained ground truth frames (weight: 0.0001)

We train the model using the above-described loss terms across all sessions for which two cameras are available. We then use the resulting weights to generate 3D predictions for sessions with only one camera. For training specifics refere to the training.py file within this folder.
