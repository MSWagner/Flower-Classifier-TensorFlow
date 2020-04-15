# Flower image classifier with TensorFlow

Project code for Udacity's 'Intro to Machine Learning with TensorFlow' Nanodegree program.

<img src="https://github.com/MSWagner/Flower-Classifier-TensorFlow/blob/master/assets/prediction.png" width="400">

## Jupyter Notebook Files

- [Jupyter Notebook](https://github.com/MSWagner/Flower-Classifier-TensorFlow/blob/master/Flower_Image_Classifier.ipynb)
- [HTML](https://github.com/MSWagner/Flower-Classifier-TensorFlow/blob/master/Flower_Image_Classifier.html)

## Command line application

#### Predict image:

```
python predict.py <image_path> <model_path>

Example:
python predict.py flowers/test/1/image_06764.jpg model_path.5h
``` 

| Argument      | Short         | Default | Description  |
| ------------- |:-------------:| -------------:| -----:|
| image_path      | | | Image path for the prediction |
| model_path | | | Path of the saved model |
| --top_k | -k    | 1 | Number of the top k most likely classes |
| --category_names | -c | label_map.json | JSON file path to map categories to real names |

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
