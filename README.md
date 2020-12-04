Super resolution! Converts 480P videos to 1440P.

# Setup

Dependencies: OpenCV, protobuf, tensorflow.

# System requirements

I'm using a desktop with i7-9700K, 32G RAM and RTX 3090 (24G GPU RAM).

Hyperparams were tuned for this machine to make full use of it. Reducing the batch size / number of threads / depth of the neural net could potentially make the code work on weaker machines. 

If you have a better workstation, please feel free to do whatever the f**k you want. 

# Usage

## Build python proto

```
export SRC_DIR=/home/darkthecross/Documents/super_resolution/src/proto
export DST_DIR=/home/darkthecross/Documents/super_resolution/src
protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/training_example.proto
```
## Extract training examples

Python was too slow, thus we have the cpp version:

```
mkdir build && cd build
cmake ..
make
./generate_training_examples "/home/darkthecross/Videos/1080*"
```

Warning: this multithreaded may take more than 20G RAM to run.

## Training

### Single Frame

Yes, single frame super resolution.

[!img](edsr.png)

This gets us a MSE of 7.71 and MAE of 1.48. 

Purely eye-balling, the neural network performs much better than nearest pixel interpolation, a little bit better than cubic interpolation:

[!img](imgs/eval_1.png)

[!img](imgs/eval_2.png)

My theory is that, we could achieve better error using multi frame as inputs.

### Multi frame

[!img](edsr_multi.png)

Turned out that we only got to a MSD of 12.05 and MAE of 1.91, worse than single frame. Fine.