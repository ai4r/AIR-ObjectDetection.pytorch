# Personal Belongings Detection and Recognition Module [AIR Project]

This is an implementation of Personal Belongings Detection and Recognition Module in AIR Project.
The module has two main parts, object detector and instance classifier.
The object detector is based on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) and the instance classifier consists of one fc layer. 

### Environment
Follow the environment in [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).

### Installation

1. Clone this repository.
    ```bash
    git clone https://github.com/ai4r/AIR-ObjectDetection.pytorch 
    ```

2. Clone [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) and install on faster-rcnn.pytorch folder.

3. Copy faster-rcnn.pytorch/cfgs folder to models/

4. Download [the model file](https://www.dropbox.com/s/be0isevd22eikqb/faster_rcnn_1_10_14657.pth?dl=0) from [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).
   
   
   
### Run
#### Object instance registration

1. Run the demo code.
   ```bash
   python demo.py
   ```
2. Press key 'r' and type two information, category_name and instance_name. 
3. Show the object instance until predefined number of images are captured. 
   
#### Object instance detection and recognition
1. Run the demo code.
   ```bash
   python demo.py
   ```
2. Press key 'd' and see the result.

### License

See LICENSE.md