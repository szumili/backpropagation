# MNIST Handwritten Digit Recognition with Backpropagation

This project implements a neural network with backpropagation to recognize handwritten digits from the **MNIST dataset**. The solution includes Fourier transform features as additional inputs to improve recognition accuracy.

Tested with `Python 3.10.8`

## Features
- Interactive interface for drawing digits
- Pre-trained weights loading/saving functionality (*weights1.txt, weights2.txt, weights3.txt, weights4.txt*) 
- Visualization of recognition confidence scores
- Various image manipulation tools (noise, negation, shifting)

## Usage

1. Drawing Digits:
    - Left-click to draw digits on the 28x28 grid
    <p align="center">
        <img src="./images/1.1_digit_draw.png" width="400">
    </p>
    - Use `Show the matrix` button to show the matrix
    <p align="center">
        <img src="./images/1.2_show_matrix.png" width="400">
    </p>
    - Use `Clear` button to clear the canvas
    <p align="center">
        <img src="./images/1.3_clear.png" width="400">
    </p>

2. Predefined Digits:
    - Buttons `Draw 0`-`Draw 9` will draw a random MNIST sample for each digit
    <table>
        <tr>
            <td><img src="images/2.0.png" width="200"></td>
            <td><img src="images/2.1.png" width="200"></td>
            <td><img src="images/2.2.png" width="200"></td>
            <td><img src="images/2.3.png" width="200"></td>
            <td><img src="images/2.4.png" width="200"></td>
        </tr>
    </table>
    <table>
        <tr>
            <td><img src="images/2.5.png" width="200"></td>
            <td><img src="images/2.6.png" width="200"></td>
            <td><img src="images/2.7.png" width="200"></td>
            <td><img src="images/2.8.png" width="200"></td>
            <td><img src="images/2.9.png" width="200"></td>
        </tr>
    </table>

3. Image Manipulation:
    - `Add noise`: Adds random noise to the image
    <table>
        <tr>
            <td><img src="images/3.1_noise.png" width="200"></td>
            <td><img src="images/3.2_noise.png" width="400"></td>
        </tr>
    </table>
    - `Negative`: Inverts the image colors
    <p align="center">
        <img src="./images/3.3_neg.png" width="400">
    </p>
    - Arrow buttons (`↑`, `↓`, `←`, `→`): Shift the image in different directions
    <table>
        <tr>
            <td><img src="images/3.4_move.png" width="400"></td>
            <td><img src="images/3.5_move.png" width="400"></td>
        </tr>
    </table>

4. Recognition:
    - Click `What digit is this?` to classify the drawn digit
    - The system will display the most probable digit and bar chart of per-class confidence scores
    <p align="center">
        <img src="./images/4_predict.png" width="600">
    </p>

5. Training:
    - `Train` button trains the network on MNIST data
    - Training uses both pixel values and Fourier transform features
    - After training, the `Performance` window is displayed automatically 

6. Performance:
    - The `Performance` button evaluates the currently loaded model on the training and test sets
    <p align="center">
        <img src="./images/6.1_pred.png" width="500">
    </p>
    - After prediction, the window shows accuracy on the training and test sets
    <p align="center">
        <img src="./images/6.2_acc.png" width="400">
    </p>
    - Detailed metrics are available for each class, including precision, recall, F1 score, and support
    <p align="center">
        <img src="./images/6.3_cr.png" width="400">
    </p>

7. Predefined Weights:
    - The provided weight files (*weights1.txt, weights2.txt, weights3.txt, weights4.txt*) were generated during training using a batch size of 64 and 200 epochs
    - The model using these weights achieves approximately **96.61% accuracy on the training set** and **93.89% accuracy on the test set**
    - Predefined weights are stored in the `weights/` directory
