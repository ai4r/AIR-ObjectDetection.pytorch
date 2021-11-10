# Personal Belongings Detection and Recognition Module [AIR Project]

This is an implementation of Personal Belongings Detection and Recognition Module in AIR Project.
The module has two main parts, object detector and instance classifier.
The object detector is based on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), the attention module came from 
 [CBAM and BAM](https://github.com/Jongchan/attention-module), and the instance classifier consists of one fc layer. 

### Installation

1. Clone this repository.
    ```bash
    git clone https://github.com/ai4r/AIR-ObjectDetection.pytorch
    cd AIR-ObjectDetection.pytorch
    pip install -r requirements.txt
    ```

2. create a model folder
    ```bash
    mkdir models/InstModel
    ```
    
3. download and install the font file
    ```bash
    wget http://cdn.naver.com/naver/NanumFont/fontfiles/NanumFont_TTF_ALL.zip
    unzip NanumFont_TTF_ALL.zip -d NanumFont
    rm -f NanumFont_TTF_ALL.zip
    mv NanumFont /usr/share/fonts/
    fc-cache -f -v
    ```

4. Download [the model files](https://drive.google.com/drive/folders/1aKOKMjdFcnGWdZo_VywG9pwlRyiUCodc) and move to models folder.
   
   
### Run
#### Object detection only

1. Run the demo code.
   ```bash
   python demo_detector.py
   ```
2. The program will load images from the test_input_image folder, then save results images to the test_output_image folder.

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
