Blog about YOLO: http://machinethink.net/blog/object-detection-with-yolo/
YOLO paper: https://arxiv.org/pdf/1506.02640.pdf
YOLO9000: https://arxiv.org/pdf/1612.08242.pdf

## How to do Object Detection

### Option 1: Sliding Window
* Train an image classifier
* Divide your image into pieces (these probably need to be of varying size?)
* Scan each piece with the classifier
* Pick out the pieces where the classifier has high confidence of something.
* Call that region the bounding box of whatever the classifier has high confidence in.

* Pretty slow. Lots of boxes to scan

### Option 2: Region Proposal
* Train an image classifier
* Run something to propose "interesting" regions
* Scan those regions with your classifier
* Pick out regions with high probability of being a class
