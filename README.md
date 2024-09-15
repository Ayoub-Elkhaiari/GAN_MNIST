# GAN Project (PyTorch Version)
the TensorFlow version is similar to that the only difference is in the display method 

## Overview
This project implements a Generative Adversarial Network (GAN) using PyTorch. The GAN is composed of two main components:
- **Generator**: Creates fake images from random noise.
- **Discriminator**: Distinguishes between real images (from the dataset) and fake images generated by the Generator.

The project uses the MNIST dataset to train the GAN and generate new handwritten digits similar to the ones in the dataset.

## Project Structure
- `Discriminator.py`: Contains the definition of the Discriminator network and its associated loss function.
- `Generator.py`: Contains the definition of the Generator network and its associated loss function.
- `app.py`: Script to train the GAN using the MNIST dataset, visualize the results, and monitor losses during training.

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.x
- PyTorch
- torchvision
- matplotlib
- tqdm

To install the necessary packages, run:
```bash
pip install torch torchvision matplotlib tqdm
```

## How to Run

1. Clone the repository
First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/Ayoub-Elkhaiari/GAN_MNIST.git
cd gan-project
```

2. Install dependencies
Install the necessary dependencies for the project. The required Python packages are listed below:
* `torch`
* `torchvision`
* `matplotlib`
* `tqdm`

You can install them using pip:

```bash
pip install torch torchvision matplotlib tqdm
```

3. Train the GAN
To start the training process, run the `app.py` script. It will train the GAN on the MNIST dataset and periodically display generated images alongside the real images.

```bash
python app.py
```

During training, the losses for both the generator and the discriminator will be displayed after every few steps, and generated images will be visualized for comparison.

4. View Results
Generated images will be periodically shown using `matplotlib`. At each display step, you'll see real images from the MNIST dataset alongside fake images generated by the GAN.

Example output:

```plaintext
Step 500: Generator loss: 1.234, discriminator loss: 0.567
Step 1000: Generator loss: 0.987, discriminator loss: 0.654
```

5. Modify Hyperparameters
You can modify the hyperparameters like the number of epochs, learning rate, batch size, and others in the `app.py` file:

```python
# Hyperparameters in app.py
n_epochs = 200
z_dim = 64
batch_size = 128
lr = 0.00001
```

Feel free to adjust them according to your computational resources and desired output quality.

6. Visualize Generated Images
To visualize the generated images at various stages of the training process, the `show_tensor_images` function is used in `app.py`. This will create a grid of images for easy comparison between the real and generated images.

```python
show_tensor_images(fake)
show_tensor_images(real)
```

## Key Components of the Code

### Discriminator
The Discriminator network takes in an image and predicts whether it is real or fake, consisting of several linear layers followed by LeakyReLU activation.

```python
class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        # Neural network structure for the discriminator
        ...
```

### Generator
The Generator network takes in random noise as input and produces a fake image, using a series of linear layers with Batch Normalization and ReLU activation.

```python
class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        # Neural network structure for the generator
        ...
```

### Loss Functions
* **Discriminator Loss**: Binary Cross-Entropy between the discriminator's predictions for real and fake images.

```python
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    # Compute the loss for the discriminator
    ...
```

* **Generator Loss**: Binary Cross-Entropy between the discriminator's predictions for generated images and a ground truth of all real images.

```python
def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    # Compute the loss for the generator
    ...
```

## Results
As the training progresses, the generator learns to create more realistic images, while the discriminator learns to better distinguish between real and fake images. Below are some sample results after training for some epochs and still training:

![Screenshot 2024-09-15 160155](https://github.com/user-attachments/assets/8c6412bd-68ec-47e4-b68c-f7a5753bb406)
