# MNIST Handwritten Digit Recognition with Backpropagation

This project implements a neural network with backpropagation to recognize handwritten digits from the MNIST dataset. The solution includes Fourier transform features as additional inputs to improve recognition accuracy.

## Features
- Interactive interface for drawing digits
- Pre-trained weights (*wagi1.txt, wagi2.txt, wagi3.txt, wagi4.txt*) loading/saving functionality
- Visualization of recognition confidence scores
- Various image manipulation tools (noise, negation, shifting)

## Usage

1. Drawing Digits:
    - Left-click to draw digits on the 28x28 grid
    - Use the "Wyświetl macierz" button to show the matrix
    - Use the "Wyczyść" button to clear the canvas

2. Predefined Digits:
    - Buttons "Narysuj 0"-"Narysuj 9" will generate random MNIST samples of each digit

3. Image Manipulation:
    - "Zaszumianie": Adds random noise to the image
    - "Negatyw": Inverts the image colors
    - Arrow buttons: Shift the image in different directions

4. Recognition:
    - Click "Co to za cyfra?" to classify the drawn digit
    - The system will display the most probable digit and a confidence histogram

5. Training:
    - "Uczenie" button trains the network on MNIST data
    - Training uses both pixel values and Fourier transform features



