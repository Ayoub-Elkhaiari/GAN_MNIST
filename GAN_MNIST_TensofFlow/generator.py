
import tensorflow as tf

def get_generator_block(input_dim, output_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(output_dim),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()
    ])

class Generator(tf.keras.Model):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = tf.keras.Sequential([
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            tf.keras.layers.Dense(im_dim, activation='sigmoid')
        ])

    def call(self, noise):
        return self.gen(noise)

def get_noise(n_samples, z_dim):
    return tf.random.normal([n_samples, z_dim])

def get_gen_loss(generator, discriminator, criterion, num_images, z_dim):
    noise = get_noise(num_images, z_dim)
    fake = generator(noise)
    disc_fake_pred = discriminator(fake)
    gen_loss = criterion(tf.ones_like(disc_fake_pred), disc_fake_pred)
    return gen_loss
