# MNIST Handwritten Digit Recognition with Backpropagation

This project implements a neural network with backpropagation to recognize handwritten digits from the MNIST dataset. The solution includes Fourier transform features as additional inputs to improve recognition accuracy.

## Features
- Interactive interface for drawing digits
- Pre-trained weights loading/saving functionality (*weights1.txt, weights2.txt, weights3.txt, weights4.txt*) 
- Visualization of recognition confidence scores
- Various image manipulation tools (noise, negation, shifting)

## Usage

1. Drawing Digits:
    - Left-click to draw digits on the 28x28 grid
    - Use the "Show the matrix" button to show the matrix
    - Use the "Clear" button to clear the canvas

2. Predefined Digits:
    - Buttons "Draw 0"-"Draw 9" will draw a random MNIST sample for each digit

3. Image Manipulation:
    - "Add noise": Adds random noise to the image
    - "Negative": Inverts the image colors
    - Arrow buttons: Shift the image in different directions

4. Recognition:
    - Click "What digit is this?" to classify the drawn digit
    - The system will display the most probable digit and a confidence histogram

5. Training:
    - "Train" button trains the network on MNIST data
    - Training uses both pixel values and Fourier transform features



