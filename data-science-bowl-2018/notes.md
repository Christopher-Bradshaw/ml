# Notes of what I have learned/what I am assuming.


## Data
* The data is either in RGB or RGBA format
    * I think that in all cases
        * The A is 255 everywhere
        * TODO: check on data load
    * In the majority of cases
        * R=G=B therefore it is greyscale
        * **I am converting everything to greyscale at the moment. Simplicity + not sure there is enough color images to train on?**
        * TODO: consider converting to greyscale using something better than mean
* The data is in a wide range of image sizes.
    * A side ranges from 161 pixels -> 1000+
* The masks never overlap, but they can be adjacent.
    * **Not a matter of finding regions, but finding individual nuclei**


## Nuclei
* Range in size from 15 to 10 000 pixels
    * Small ones are much more common. Actually looks like the SMF!


## Free flowing, probably wrong thoughts about how to do this
* Can create lots of new data by:
    * Shifting, fuzzing, slicing into different regions
* Need to read a couple of papers on how to do detection and bounding
