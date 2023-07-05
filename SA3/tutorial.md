Tokenize and Predict the Caption
==============================


In this activity, you will learn to tokenize the caption set and predict the caption forthe video.


<img src= "https://s3.amazonaws.com/media-p.slid.es/uploads/1525749/images/10596282/ezgif.com-optimize.gif" width = "480" height = "220">



Follow the given steps to complete this activity:


1. Tokenize the caption set.


* Open the file main.py.


* Create an empty list named `all_captions` for storing the captions.


    `all_captions = []`


* Use a nested for loop to iterate though mapping and each key in mapping and append each caption to all_captions.
   
    `for key in mapping:`

         `for caption in mapping[key]:`
         
              `all_captions.append(caption)`


   
* Create a tokenizer and store it in the `tokenizer` variable.


    `tokenizer = Tokenizer()`


* Update the internal vocabulary of tokenizer based on a list of texts.


    `tokenizer.fit_on_texts(all_captions)`


* Save and run the code to check the output.







