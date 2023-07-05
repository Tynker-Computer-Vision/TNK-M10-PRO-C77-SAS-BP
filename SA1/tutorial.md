Prepare the Caption Dataset
==============================


In this activity, you will learn to feed sample images and corresponding captions to train a model.


<img src= "https://media.slid.es/uploads/1525749/images/10570797/captions.gif" width = "480" height = "220">



Follow the given steps to complete this activity:


1. Prepare the dataset


* Open the file main.py.


* Make sure that  the features file features.pkl, captions file caption.txt and the trained model best_model.h5  are  downloaded and imported from https://drive.google.com/drive/folders/1DUxlrr7qZM5aQYQWeziUopecIJ68JrKo?usp=sharing
.


* Open a file `features.pkl` in read and write mode as `f` and load it using `pickle.load()`.


    `with open('../features.pkl', 'rb') as f:`


        `features = pickle.load(f)`


* Open the file `caption.txt` in read mode into a file `f` and store the file after reading in `captions_doc`.


    `with open('../captions.txt', 'r') as f:`
        `next(f)`
        `captions_doc = f.read()`

* Load the `best_model.h5` model using `load_model()` function.

    `model = keras.models.load_model('../best_model.h5')`

* Declare a variable `mapping` and store an empty dictonary to map the images to captions.


    `mapping = {}`


* Loop through every caption using a for loop.


    `for line in captions_doc.split('\n'):`


* Split the line at `comma(,) to separate the csption from file name `


    `tokens = line.split(',')`


* Move to the next iteration if length of line is less than `2` characters.


    `if len(line) < 2:`
        `continue`


* Assign `image_id` and caption from `tokens[0]`, `tokens[1]` respectively.


    `image_id, caption = tokens[0], tokens[1:]`


* Split the image name at `period(.)` to remove the extension from the image ID.


    `image_id = image_id.split('.')[0]`


* Join the caption with empty string using join to convert caption list to string.


    `caption = " ".join(caption)`


* Create an empty list for an `image_id` if its not present in `mapping` dictonary.


    `if image_id not in mapping:`


        `mapping[image_id] = []`


* Append the `caption` at the `image_id` in the `mapping` dictonary.


    `mapping[image_id].append(caption)`


* Print an element from the `mapping` dictionary.


    `print(mapping["1000268201_693b08cb0e"])`


* Save and run the code to check the output.
