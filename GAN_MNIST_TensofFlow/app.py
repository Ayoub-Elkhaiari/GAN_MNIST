# app.py
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from generator import Generator, get_gen_loss, get_noise
from discriminator import Discriminator, get_disc_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.datasets import mnist

# Parameters
z_dim = 64
batch_size = 128
lr = 1e-5
n_epochs = 200

# Load MNIST dataset
(train_images, _), (_, _) = mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 784).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(batch_size)

# Initialize models and optimizer
generator = Generator(z_dim)
discriminator = Discriminator()
generator_optimizer = tf.keras.optimizers.Adam(lr)
discriminator_optimizer = tf.keras.optimizers.Adam(lr)

# Loss function
criterion = BinaryCrossentropy(from_logits=True)

# Visualization function
def show_tensor_images(image_tensor, num_images=25, size=(28, 28)):
    image_grid = (image_tensor[:num_images].numpy().reshape(num_images, *size) + 1) / 2.0
    fig, axs = plt.subplots(5, 5, figsize=(5, 5))
    idx = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(image_grid[idx], cmap='gray')
            axs[i, j].axis('off')
            idx += 1
    plt.show()

# Training loop
cur_step = 0
for epoch in range(n_epochs):
    for real in tqdm(train_dataset):
        # Update discriminator
        with tf.GradientTape() as disc_tape:
            disc_loss = get_disc_loss(generator, discriminator, criterion, real, batch_size, z_dim)
        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

        # Update generator
        with tf.GradientTape() as gen_tape:
            gen_loss = get_gen_loss(generator, discriminator, criterion, batch_size, z_dim)
        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

        if cur_step % 500 == 0:
            print(f"Step {cur_step}: Generator loss: {gen_loss}, Discriminator loss: {disc_loss}")
            fake_noise = get_noise(batch_size, z_dim)
            fake = generator(fake_noise)
            show_tensor_images(fake)
        cur_step += 1
