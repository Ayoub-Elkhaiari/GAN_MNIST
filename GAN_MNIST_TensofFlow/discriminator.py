from generator import get_noise
import tensorflow as tf

def get_discriminator_block(input_dim, output_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(output_dim),
        tf.keras.layers.LeakyReLU(0.2)
    ])

class Discriminator(tf.keras.Model):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = tf.keras.Sequential([
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            tf.keras.layers.Dense(1)
        ])

    def call(self, image):
        return self.disc(image)

def get_disc_loss(generator, discriminator, criterion, real, num_images, z_dim):
    noise = get_noise(num_images, z_dim)
    fake = generator(noise)
    disc_fake_pred = discriminator(fake)
    disc_real_pred = discriminator(real)
    
    disc_fake_loss = criterion(tf.zeros_like(disc_fake_pred), disc_fake_pred)
    disc_real_loss = criterion(tf.ones_like(disc_real_pred), disc_real_pred)
    
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss
