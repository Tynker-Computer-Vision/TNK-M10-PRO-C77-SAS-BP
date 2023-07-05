Save the Caption in a file
==========================


In this activity, you will save all the captions in a file.


<img src= "https://s3.amazonaws.com/media-p.slid.es/uploads/1525749/images/10596282/ezgif.com-optimize.gif" width = "480" height = "220">


Follow the given steps to complete this activity:


1. Save the caption.


* Open the file main.py.


* Create an empty list for storing  the predicted captions.


    `predictedCaptions = []`


* Append the `caption` to `predictedCaptions` list inside the if condition.


    `predictedCaptions.append(caption)`


* Open a file called `Automated_Captions.txt` in `write` mode.


    `file = open('Automated_Captions.txt', 'w')`


* Write the `predictedCaptions` list to the file.


    `file.write(str(predictedCaptions))`


* Close the file.


    `file.close()`


* Save and run the code to check the output.
