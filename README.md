# Video-Segment-Classification-Using-Simple-CNNs-with-OpenCV-and-Pytorch

Video used for this project - https://www.youtube.com/watch?v=elIYmH2sR3E (India vs Pakistan 2007 4th Odi Gwalior - Full Highlights)


Video segment scene classification using CNNs with OpenCV and Pytorch. Project revolves around segmenting all the bowling scenes from a cricket match using Deep learning techniques.

The basic approach is to extract frames out of a video and to build a CNN classifier using the extracted data that can converge on classify a bowling scene from a normal one. To use the same model to run inference on a frames extracted from a new video to later stitch all the frames that were predicted to be from a bowling scene.

Requirements
OpenCv 4.1
Pytorch 1.2

File Wise usage- 
**Change the path in all of these scripts according to your data and then run them, They shall all work just fine.**

vidtoframes.py - Use this script to extract all the frames out of a video of your choice.

renamefiles.py - A convenience script that will rename and sort all the files accordingly.

classifiermodel.py - Run this to train a densenet121 pretrained model on your own data. You can change the name of the model accordingly.

framestovids.py - Run this to stitch all the frames in a folder and make a video output out of it.

**You can also use the pretrained models from the models folder.**

Result output of Vgg13 with around 90% accuracy - https://youtu.be/MXRJtJRL-Dk