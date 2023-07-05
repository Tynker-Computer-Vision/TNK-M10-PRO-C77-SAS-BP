Clean the Caption Dataset
=========================

In this activity, you will learn to process and clean the dataset.


<img src= "https://s3.amazonaws.com/media-p.slid.es/uploads/2071954/images/10589114/SLide_33.gif" width = "880" height = "220">



Follow the given steps to complete this activity:


1. Clean the dataset


* Open the file main.py.


* Loop throught each key and captions in the mapping


    `for key, captions in mapping.items():`


* Loop through each `caption` in `captions` for a given image using an for loop.


    `for i in range(len(captions)):`


* Assign one caption at a time into the variable caption.


    `caption = captions[i]`


* Convert the caption to lowercase using `lower()`.


    `caption = caption.lower()`


* Replace the `digits, special chars` with empty spaces.


    `caption = caption.replace('[^a-z]', '')`


* Delete additional spaces.


    `caption = caption.replace('\s+', ' ')`


* Concatenate`startseq` and `endseq` tags to the beginning and end of the caption text.


    `caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'`


* Store the cleaned `caption` in the captions list.


    captions[i] = caption


* Print the cleaned element from the `mapping` dictonary.


    `print("::::::::::::::::::::::::Mapping after cleaning::::::::::::::::::::::::")`


    `print(mapping["1000268201_693b08cb0e"])`


* Save and run the code to check the output.


