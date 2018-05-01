import os
import time
import numpy as np
import tensorflow as tf
import load_vgg
import utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

def setup():
    utils.safe_mkdir('checkpoints')
    utils.safe_mkdir('output')


# try removing the object argument
class StyleTransfer(object):
    def __init__(self, content_img, style_img, img_width, img_height):
        '''
        img_width and img_height are the dimensions we expect from the generated image.
        We will resize input content image and input style image to match this dimension.
        '''
        self.img_width = img_width
        self.img_height = img_height
        self.content_img = utils.get_resized_image(content_img, img_width, img_height)
        self.style_img = utils.get_resized_image(style_img, img_width, img_height)
        self.initial_img = utils.generate_noise_image(self.content_img, img_width, img_height)

        self.content_layer = 'conv4_2'
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.content_w = 0.01
        # self.content_w = 1

        self.style_w = 1

        self.style_layer_w = [0.5, 1.0, 1.5, 3.0, 4.0]
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.lr = 2.0
        ###############################

    def create_input(self):
        '''
        We will use one input_img as a placeholder for the content image,
        style image, and generated image, because:
            1. they have the same dimension
            2. we have to extract the same set of features from them
        We use a variable instead of a placeholder because we're, at the same time,
        training the generated image to get the desirable result.

        Note: image height corresponds to number of rows, not columns.
        '''
        with tf.variable_scope('input'):
            self.input_img = tf.get_variable('in_img',
                                             shape=([1, self.img_height, self.img_width, 3]),
                                             dtype=tf.float32,
                                             initializer=tf.zeros_initializer())

    def load_vgg(self):
        '''
        Load the saved model parameters of VGG-19, using the input_img
        as the input to compute the output at each layer of vgg.

        During training, VGG-19 mean-centered all images and found the mean pixels
        to be [123.68, 116.779, 103.939] along RGB dimensions. We have to subtract
        this mean from our images.

        '''

        self.vgg = load_vgg.VGG(self.input_img)
        self.vgg.load()
        self.content_img -= self.vgg.mean_pixels
        self.style_img -= self.vgg.mean_pixels

    def _content_loss(self, p, f):
        """ Calculate the loss between the feature representation of the
        content image and the generated image.

        Inputs:
            p: content representation of the content image
            f: content representation of the generated image

            Note: Don't use the coefficient 0.5 as defined in the paper.
            Use the coefficient defined in the assignment handout.
        """

        self.content_loss = tf.reduce_sum((f - p) ** 2) / (4.0 * p.size)

    def _gram_matrix(self, F, N, M):
        """ Create and return the gram matrix for tensor F
        """

        F = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(F), F)

    def _single_style_loss(self, a, g):
        """ Calculate the style loss at a certain layer
        Inputs:
            a is the feature representation of the style image at that layer
            g is the feature representation of the generated image at that layer
        Output:
            the style loss at a certain layer (which is E_l in the paper)
        """

        N = a.shape[3]  # number of filters
        M = a.shape[1] * a.shape[2]  # height times width of the feature map
        A = self._gram_matrix(a, N, M)
        G = self._gram_matrix(g, N, M)
        return tf.reduce_sum((G - A) ** 2 / ((2 * N * M) ** 2))

    def _style_loss(self, A):
        """ Calculate the total style loss as a weighted sum
        of style losses at all style layers
        """
        n_layers = len(A)
        E = [self._single_style_loss(A[i], getattr(self.vgg, self.style_layers[i])) for i in range(n_layers)]

        self.style_loss = sum([self.style_layer_w[i] * E[i] for i in range(n_layers)])

    def losses(self):
        with tf.variable_scope('losses'):
            with tf.Session() as sess:
                sess.run(self.input_img.assign(self.content_img))
                gen_img_content = getattr(self.vgg, self.content_layer)
                content_img_content = sess.run(gen_img_content)
            self._content_loss(content_img_content, gen_img_content)

            with tf.Session() as sess:
                sess.run(self.input_img.assign(self.style_img))
                style_layers = sess.run([getattr(self.vgg, layer) for layer in self.style_layers])
            self._style_loss(style_layers)

            self.total_loss = self.content_w * self.content_loss + self.style_w * self.style_loss

    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss, global_step=self.gstep)

    def create_summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('content loss', self.content_loss)
            tf.summary.scalar('style loss', self.style_loss)
            tf.summary.scalar('total loss', self.total_loss)
            self.summary_op = tf.summary.merge_all()

    def build(self):
        self.create_input()
        self.load_vgg()
        self.losses()
        self.optimize()
        self.create_summary()

    def train(self, n_iters, name_graphs):
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('graphs/'+name_graphs, sess.graph)
            sess.run(self.input_img.assign(self.initial_img))

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/style_transfer/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            initial_step = self.gstep.eval()

            start_time = time.time()
            for index in range(initial_step, n_iters):
                sess.run(self.opt)
                gen_image, total_loss, summary = sess.run([self.input_img,
                                                           self.total_loss,
                                                           self.summary_op])

                gen_image = gen_image + self.vgg.mean_pixels
                writer.add_summary(summary, global_step=index)
                print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                print('   Loss: {:5.1f}'.format(total_loss))
                print('   Took: {} seconds'.format(time.time() - start_time))
                start_time = time.time()

                filename = 'output/%d.png' % (index)
                utils.save_image(filename, gen_image)

                if (index + 1) % 20 == 0:
                    saver.save(sess, 'checkpoints/style_stranfer/style_transfer', index)


if __name__ == '__main__':
    setup()
    # machine = StyleTransfer('content/hidelberg.jpg', 'styles/starry_night.jpg', 700, 500)
    machine = StyleTransfer('content/projects/style_transfer_project/styles/starry_night.jpg',
                            'content/projects/style_transfer_project/styles/starry_night.jpg', 700, 500)
    machine.build()
    machine.train(600, 'hidelberg_starry')
