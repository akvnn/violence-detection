# Violence Detection

## Description

This project aims to detect violence in video sequences using deep learning. We use a modified Densenet121 model, with added ConvLSTM2D layers, to process video frames and predict whether violence is present. Additionally, optical flow was used for temporal feature extraction.

## Technologies Used

This project is implemented using Python and the following libraries:

- [OpenCV](https://opencv.org/): Used for real-time computer vision to read and manipulate images and videos.
- [NumPy](https://numpy.org/): Used for numerical computations in Python.
- [PyTorch](https://pytorch.org/): An open-source machine learning library used to create and train the neural network.
- [Pandas](https://pandas.pydata.org/): Used for data manipulation and analysis.
- [Torchvision](https://pytorch.org/vision/stable/index.html): Used to load the pre-trained Densenet121 model and perform image transformations.
- [ConvLSTM](https://github.com/ndrplz/ConvLSTM_pytorch): Used to add ConvLSTM layers to the model.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/akvnn/violence-detection
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To use the violence detection model, run the cells in the `app.ipynb` Jupyter Notebook File

## Credits

This project was developed for Computer Vision Course Project by Ahmed, Mohammed, Saeed, Yousef

## License

This project is under the [MIT License](./LICENSE).
