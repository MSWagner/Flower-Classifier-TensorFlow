
import argparse

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import json

from PIL import Image

def get_input_args():
    """
    Command Line Arguments:
      1. Image path
      2. Path of the saved model
      3. Number of the top k most likely classes as --top_k with default value 5
      4. JSON file to map categories to real names as --category_names with default value label_map.json
    This function returns these arguments as an ArgumentParser object.
    
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', type = str, help = 'Path of the prediction image')    
    parser.add_argument('model_path', type = str, help = 'Path of the saved model') 
    parser.add_argument('-k', '--top_k', type = int, default = 1, help = 'Number of the top k most likely classes') 
    parser.add_argument('-json', '--category_names', type = str, default = "label_map.json", help = 'JSON file to map categories to real names')

    in_args = parser.parse_args()
    
    if in_args is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        print("Command Line Arguments:\n image_path =", in_args.image_path, "\n model_path =", in_args.model_path, "\n top_k =", in_args.top_k, "\n category_names =", in_args.category_names)

    return in_args

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image,
        returns an Numpy array
    '''
    image = np.asarray(image)
    
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    
    return np.expand_dims(image.numpy(), axis=0)

def predict(image, model, topk):    
    prediction = model.predict(image)
    topk_predictions = tf.math.top_k(prediction, k=topk, sorted=True, name=None)
    
    probs = topk_predictions.values[0].numpy()
    classes = topk_predictions.indices[0].numpy()
    
    return probs, classes

def main():
    input_args = get_input_args()
        
    # Load model
    model = tf.keras.models.load_model(input_args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    # Process image
    image = Image.open(input_args.image_path)
    processed_image = process_image(image)
      
    # Predict image
    probs, classes = predict(processed_image, model, input_args.top_k)
    
    # Show result
    with open(input_args.category_names, 'r') as f:
        class_names = json.load(f)
        
    categories = [class_names[str(category_index+1)] for category_index in classes]
        
    for i in range(len(probs)):
        print("\nTopK {}, Probability: {}, Category: {}".format(i+1, probs[i], categories[i]))

if __name__ == '__main__':
    main()