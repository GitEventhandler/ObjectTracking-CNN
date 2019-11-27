# ObjectTracking_CNN
A KCF tracker implementation which uses cnn feature as core.

# Inspired
This project was inspired from [tiago-cerveira/KCF-Tracker-CNN-Features].But change the backend from theano + lasagne to pytorch.And add muti-target tracing function.

# Details
* Support Muti-Target tracking
* Support tripwire and restricted zone checking
* Support CUDA accleration
* Use PyTorch as cnn backend

[tiago-cerveira/KCF-Tracker-CNN-Features]: https://github.com/tiago-cerveira/KCF-Tracker-CNN-Features

# Requirement

## Runtime
* Anaconda: Python 3.7.5

## Library Requirement

### From Python pip
* numpy == any
* opencv == 3.4.5.20

### From Anaconda 
* PyTorch == any

### CUDA (Optional)
* CUDA == 9.1 
* cuDNN == 9.1

# Useage
## Run main
```
python main.py <path of the video file>
```

## Use KCFMultiTracker
```
tracker = KCFMultiTracker(True, True)
tracker.init(rois, frist_frame)
while not_ended:
    success, result_rois = tracker.update(next_frame)
    next_frame = get_next_frame()
```