import tensorflow as tf

import regular_grid_interpolation
from tensorboard.plugins.mesh import summary_v2 as mesh_summary
from tensorflow_graphics.nn.loss import chamfer_distance

from tensorflow_graphics.nn.layer import pointnet
from tensorflow.keras.layers import multiply
import numpy as np
import open3d as o3d


class Conv1DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid'):
        super().__init__()
        self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(
            filters, (kernel_size, 1), (strides, 1), padding
        )

    def call(self, x):
        x = tf.expand_dims(x, axis=2)
        x = self.conv2dtranspose(x)
        x = tf.squeeze(x, axis=2)
        return x


class PointGenConTF(tf.keras.layers.Layer):
    def __init__(self, bottleneck_size=16):
        super(PointGenConTF, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.conv1 = tf.keras.layers.Conv1D(self.bottleneck_size, 1, padding='SAME', data_format='channels_first')
        self.conv2 = tf.keras.layers.Conv1D(self.bottleneck_size // 2, 1, padding='SAME', data_format='channels_first')
        self.conv3 = tf.keras.layers.Conv1D(self.bottleneck_size // 4, 1, padding='SAME', data_format='channels_first')
        self.conv4 = tf.keras.layers.Conv1D(3, 1, padding='SAME', data_format='channels_first')

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = tf.keras.activations.tanh(self.conv4(x))
        return x


def _dense_to_sparse(data):
    """Convert a numpy array to a tf.SparseTensor."""
    indices = np.where(data)
    return tf.SparseTensor(
        np.stack(indices, axis=-1), data[indices], dense_shape=data.shape)


def _dummy_data(batch_size, num_vertices, num_channels):
    """Create inputs for feature_steered_convolution."""
    if batch_size > 0:
        data = np.zeros(
            shape=(batch_size, num_vertices, num_channels), dtype=np.float32)
        neighbors = _dense_to_sparse(
            np.tile(np.eye(num_vertices, dtype=np.float32), (batch_size, 1, 1)))
    else:
        data = np.zeros(shape=(num_vertices, num_channels), dtype=np.float32)
        neighbors = _dense_to_sparse(np.eye(num_vertices, dtype=np.float32))
    return data, neighbors


class AttentionBlockGate(tf.keras.layers.Layer):
    def __init__(self, n_filters, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

        self.attconv1 = tf.keras.layers.Conv1D(n_filters, kernel_size=1)
        self.attbn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.attconv2 = tf.keras.layers.Conv1D(n_filters, kernel_size=1)

        self.attbn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.attaddf = tf.keras.layers.Add()
        # self.actRelu = tf.keras.layers.Activation('relu')
        self.attconv3 = tf.keras.layers.Conv1D(1, kernel_size=1)
        self.attbn3 = tf.keras.layers.BatchNormalization(axis=-1)
        # self.actSig = tf.keras.layers.Activation('sigmoid')

    def call(self, x, shortcut, training=True):
        g1 = self.attconv1(shortcut)
        g1 = self.attbn1(g1, training)
        x1 = self.attconv2(x)
        x1 = self.attbn2(x1, training)

        g1_x1 = self.attaddf([g1, x1])
        psi = tf.nn.relu(g1_x1)
        psi = self.attconv3(psi)
        psi = self.attbn3(psi, training)
        psi = tf.nn.sigmoid(psi)
        x = multiply([x, psi])
        return x


class ResnetBlock1D(tf.keras.layers.Layer):
    def __init__(self, n_filters, out_channels, kernel_size=1, final_relu=False, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.activation = tf.keras.layers.Activation('relu')
        self.conv1d = tf.keras.layers.Conv1D(n_filters, kernel_size=kernel_size, strides=1,
                                             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.,
                                                                                                   stddev=0.02))
        self.conv1d2 = tf.keras.layers.Conv1D(n_filters, kernel_size=3, strides=1, padding='same',
                                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.,
                                                                                                    stddev=0.02))
        self.conv1d3 = tf.keras.layers.Conv1D(out_channels, kernel_size=1, strides=1,
                                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.,
                                                                                                    stddev=0.02))
        self.lambdaf = tf.keras.layers.Lambda(lambda x: x * 0.3)
        self.addf = tf.keras.layers.Add()
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.shortcut = tf.keras.layers.Conv1D(out_channels, kernel_size=1, strides=1)
        self.final_relu = final_relu

    def call(self, inputs, training=True):
        identity = inputs
        output = inputs

        output = self.conv1d(output)
        output = self.bn1(output, training)
        output = tf.nn.relu(output)

        output = self.conv1d2(output)
        output = self.bn2(output, training)
        output = tf.nn.relu(output)

        output = self.conv1d3(output)
        output = self.bn3(output, training)
        output = tf.nn.relu(output)

        output += self.shortcut(identity)

        if self.final_relu:
            output = tf.nn.relu(output)

        return output


import math


class PointnetGenerator(tf.keras.layers.Layer):
    def __init__(self, bottleneck_size=1024, momentum=0.5, in_grid_res=32,
                 out_grid_res=16,
                 num_filters=16,
                 max_filters=512 * 8,
                 out_features=32, *args, **kwargs):
        super(PointnetGenerator, self).__init__(*args, **kwargs)
        self.in_grid_res = in_grid_res
        self.out_grid_res = out_grid_res
        self.num_filters = num_filters
        self.max_filters = max_filters
        self.out_features = out_features
        self.num_in_level = math.log(self.in_grid_res, 2)
        self.num_out_level = math.log(self.out_grid_res, 2)
        self.num_in_level = int(self.num_in_level)  # number of input levels
        self.num_out_level = int(self.num_out_level)
        num_filter_down = [
            self.num_filters * (2 ** (i + 1)) for i in range(self.num_in_level)
        ]
        # num. features in downward path
        num_filter_down = [
            n if n <= self.max_filters else self.max_filters
            for n in num_filter_down
        ]
        num_filter_up = num_filter_down[::-1][:self.num_out_level]
        self.num_filter_down = num_filter_down
        self.num_filter_up = num_filter_up
        self.momentum = momentum
        self.bottleneck_size = bottleneck_size
        self.encoder = pointnet.VanillaEncoder(self.momentum)
        self.down_modules = [ResnetBlock1D(int(n / 2), n) for n in num_filter_down]
        self.up_modules = [ResnetBlock1D(n, n) for n in num_filter_down]
        self.attentions = [AttentionBlockGate(n) for n in num_filter_down]
        self.upsamp = tf.keras.layers.UpSampling1D(2)
        self.conv_in = ResnetBlock1D(self.num_filters, self.num_filters)
        self.conv_conc = ResnetBlock1D(self.out_features, self.out_features)
        self.conv = tf.keras.layers.Conv1D(3, 1, 1, 'SAME')
        self.conv2 = pointnet.PointNetConv2Layer(self.bottleneck_size * 4, 0.5)
        self.conv3 = pointnet.PointNetConv2Layer(self.bottleneck_size * 8, 0.5)
        self.dense1 = pointnet.PointNetDenseLayer(self.bottleneck_size * 16, 0.5)
        self.dense2 = pointnet.PointNetDenseLayer(self.bottleneck_size * 8, 0.5)
        self.dense3 = pointnet.PointNetDenseLayer(1, 0.5)
        self.denss = [self.dense1, self.dense2, self.dense3]

        self.bn = tf.keras.layers.BatchNormalization(momentum=self.momentum, axis=-1)
        self.conv_out = ResnetBlock1D(
            self.out_features, self.out_features, final_relu=False)
        self.addf = tf.keras.layers.Add()
        self.final = tf.keras.layers.Conv1D(3, 1)
        self.dnpool = tf.keras.layers.MaxPool1D(2)
        self.upsamp = tf.keras.layers.UpSampling1D(2)
        self.class_head = pointnet.ClassificationHead(2)
        self.min_grid_value = (0, 0, 0)
        self.max_grid_value = (1, 1, 1)
        self.x_location_max = 1

    def _interp(self, grid, pts):
        """Interpolation function to get local latent code, weights & relative loc.
        Args:
          grid: `[batch_size, *self.size, in_features]` tensor, input feature grid.
          pts: `[batch_size, num_points, dim]` tensor, coordinates of points that
          are within the range (0, 1).
        Returns:
          lat: `[batch_size, num_points, 2**dim, in_features]` tensor, neighbor
          latent codes for each input point.
          weights: `[batch_size, num_points, 2**dim]` tensor, bi/tri-linear
          interpolation weights for each neighbor.
          xloc: `[batch_size, num_points, 2**dim, dim]`tensor, relative coordinates.
        """
        lat, weights, xloc = regular_grid_interpolation.get_interp_coefficients(
            grid,
            pts,
            min_grid_value=self.min_grid_value,
            max_grid_value=self.max_grid_value)
        xloc *= self.x_location_max

        return lat, weights, xloc

    def call(self, ligand, protein, concat=False, training=False):

        out = protein

        downP = self.conv_in(out)
        downsP = [downP]
        downL = self.conv_in(ligand)
        downsL = [downL]
        for mod in self.down_modules:
            x_ = self.dnpool(mod(downsP[-1], training=training))
            downsP.append(x_)
            x_ = self.dnpool(mod(downsL[-1], training=training))
            downsL.append(x_)

        ups = [tf.concat([downsP.pop(-1), downsL.pop(-1)], axis=1)]

        for mod, attention in zip(self.up_modules, self.attentions):
            downat = attention(self.upsamp(ups[-1]), tf.concat([downsP.pop(-1), downsL.pop(-1)], axis=1))
            x_ = tf.concat([self.upsamp(ups[-1]), downat], axis=-1)
            x_ = mod(x_, training=training)
            ups.append(x_)
        out = self.upsamp(ups[-1])[:, :-1, :]

        out = self.conv_out(out)
        out_ = out

        for dens in self.denss:
            out_ = tf.nn.leaky_relu(dens(out_))
            out_ = tf.concat([out_, out], axis=-1)

        out = self.conv(out_)

        if concat:
            out = tf.concat([out, ligand], axis=1)

        return out

    @staticmethod
    def loss(labels, logits):
        """The classification model training loss.
        Note:
          see tf.nn.sparse_softmax_cross_entropy_with_logits
        Args:
          labels: a tensor with shape `[B,]`
          logits: a tensor with shape `[B,num_classes]`
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits
        residual = cross_entropy(labels, logits)
        return tf.reduce_mean(input_tensor=residual)


def reduce_sample(path, reduce_bound=1024, sample_points=1024):
    mesh = o3d.io.read_triangle_mesh(path, print_progress=False)

    voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / reduce_bound
    mesh_smp = mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)

    pcd = mesh_smp.sample_points_uniformly(number_of_points=sample_points)

    return pcd


def sample_from_points(points, sample_points=1024):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    if avg_dist == 0:
        avg_dist = 1.0
    radius = 1.5 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius * 2]))
    sample = mesh.sample_points_uniformly(number_of_points=sample_points)
    print('sampled')
    return sample


def get_points_mesh(path, reduce_bound=1024, sample_points=1024):
    pcd = reduce_sample(path, reduce_bound, sample_points)
    return np.asarray(pcd.points, dtype=np.float32)


def get_bounding_box_points(points):
    pcdfake = o3d.geometry.PointCloud()
    pcdfake.points = o3d.utility.Vector3dVector(points)
    pcdfake.estimate_normals()
    obb = pcdfake.get_oriented_bounding_box()
    return obb


def get_points_full(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float32)


def prepare_data(pathLigand, pathProtein, reduce_bound=1024, sample_pointsLignads=1024, sample_pointsProteins=1024):
    plyLingand = reduce_sample(pathLigand, reduce_bound, sample_pointsLignads)

    plyProtein = reduce_sample(pathProtein, reduce_bound, sample_pointsProteins)

    ligand = np.array(plyLingand.points, dtype=np.float32)
    protein = np.array(plyProtein.points, dtype=np.float32)

    return ligand, protein


with tf.device('/device:GPU:1'):
    num_classes = 2
    lr_d = 1e-4
    lr_g = 1e-4
    discriminator_optimizer = tf.keras.optimizers.Adam(lr_d, beta_1=0.5, beta_2=0.9)
    generator_optimizer = tf.keras.optimizers.Adam(lr_d, beta_1=0.5, beta_2=0.9)
    generator_optimizer2 = tf.keras.optimizers.Adam(lr_d, beta_1=0.5, beta_2=0.9)

    discriminator = pointnet.PointNetVanillaClassifier(num_classes, momentum=.5)
    generator = PointnetGenerator()
    generator2 = PointnetGenerator()


    @tf.function
    def train_generator_content(ligand, protein, both, concat=False):
        with tf.GradientTape() as t:
            fake = generator(ligand, protein, training=True, concat=concat)

            cont_loss = tf.nn.l2_loss(chamfer_distance.evaluate(fake, both))

        Content_grad = t.gradient(cont_loss, generator.trainable_variables)

        generator_optimizer.apply_gradients(zip(Content_grad, generator.trainable_variables))
        return fake, cont_loss


    @tf.function
    def train_generator_content2(ligand, protein, both, concat=False):
        with tf.GradientTape() as t:
            fake = generator2(ligand, protein, training=True, concat=concat)

            cont_loss = tf.nn.l2_loss(chamfer_distance.evaluate(fake, both))

        Content_grad = t.gradient(cont_loss, generator2.trainable_variables)

        generator_optimizer2.apply_gradients(zip(Content_grad, generator2.trainable_variables))
        return fake, cont_loss


    writer = tf.summary.create_file_writer("logs3d")

    iters_per_checkpoint = 5

    config_dict = {
        'camera': {'cls': 'PerspectiveCamera', 'fov': 75},
        'lights': [
            {
                'cls': 'AmbientLight',
                'color': '#ffffff',
                'intensity': 0.75,
            }, {
                'cls': 'DirectionalLight',
                'color': '#ffffff',
                'intensity': 0.75,
                'position': [0, -1, 2],
            }],
        'material': {
            'cls': 'MeshStandardMaterial',
            'roughness': 1,
            'metalness': 0
        }
    }
    colors = np.random.randint(0, 255, (2, 3))
    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    ckpt = tf.train.Checkpoint(G=generator, G2=generator2, D=discriminator, G2_optimizer=generator_optimizer2,
                               G_optimizer=generator_optimizer,
                               D_optimizer=discriminator_optimizer,
                               step=ep_cnt)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts_v2', max_to_keep=1)
    iters_per_checkpoint = 5

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    iterator = iter([('3d/converted/6VXX_ligandA.wrl.ply', '3d/converted/6VXX_proteinA.wrl.ply',
                      '3d/converted/6VXX_ligand_bondsA.wrl.ply'),
                     ('3d/converted/6VXX_ligandB.wrl.ply', '3d/converted/6VXX_proteinB.wrl.ply',
                      '3d/converted/6VXX_ligand_bondsB.wrl.ply'),

                     ('3d/converted/6WPT_ligandA.wrl.ply', '3d/converted/6WPT_proteinA.wrl.ply',
                      '3d/converted/6WPT_ligand_bondsA.wrl.ply'),
                     ('3d/converted/6WPT_ligandB.wrl.ply', '3d/converted/6WPT_proteinB.wrl.ply',
                      '3d/converted/6WPT_ligand_bondsB.wrl.ply'),
                     ('3d/converted/6WPT_ligandC.wrl.ply', '3d/converted/6WPT_proteinC.wrl.ply',
                      '3d/converted/6WPT_ligand_bondsC.wrl.ply'),
                     ('3d/converted/6WPT_ligandE.wrl.ply', '3d/converted/6WPT_proteinE.wrl.ply',
                      '3d/converted/6WPT_ligand_bondsE.wrl.ply'),

                     ('3d/converted/6VW1_ligandA.wrl.ply', '3d/converted/6VW1_proteinA.wrl.ply',
                      '3d/converted/6VW1_ligand_bondsA.wrl.ply'),
                     ('3d/converted/6VW1_ligandB.wrl.ply', '3d/converted/6VW1_proteinB.wrl.ply',
                      '3d/converted/6VW1_ligand_bondsB.wrl.ply'),

                     ('3d/converted/6VYB_ligandA.wrl.ply', '3d/converted/6VYB_proteinA.wrl.ply',
                      '3d/converted/6VYB_ligand_bondsA.wrl.ply')
                        ,
                     ('3d/converted/6VYB_ligandB.wrl.ply', '3d/converted/6VYB_proteinB.wrl.ply',
                      '3d/converted/6VYB_ligand_bondsB.wrl.ply'),

                     ])
    progressive_samples = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    for n in range(50000):
        iterator = iter([('3d/converted/6VXX_ligandA.wrl.ply', '3d/converted/6VXX_proteinA.wrl.ply',
                          '3d/converted/6VXX_ligand_bondsA.wrl.ply'),
                         ('3d/converted/6VXX_ligandB.wrl.ply', '3d/converted/6VXX_proteinB.wrl.ply',
                          '3d/converted/6VXX_ligand_bondsB.wrl.ply'),

                         ('3d/converted/6WPT_ligandA.wrl.ply', '3d/converted/6WPT_proteinA.wrl.ply',
                          '3d/converted/6WPT_ligand_bondsA.wrl.ply'),
                         ('3d/converted/6WPT_ligandB.wrl.ply', '3d/converted/6WPT_proteinB.wrl.ply',
                          '3d/converted/6WPT_ligand_bondsB.wrl.ply'),
                         ('3d/converted/6WPT_ligandC.wrl.ply', '3d/converted/6WPT_proteinC.wrl.ply',
                          '3d/converted/6WPT_ligand_bondsC.wrl.ply'),
                         ('3d/converted/6WPT_ligandE.wrl.ply', '3d/converted/6WPT_proteinE.wrl.ply',
                          '3d/converted/6WPT_ligand_bondsE.wrl.ply'),

                         ('3d/converted/6VW1_ligandA.wrl.ply', '3d/converted/6VW1_proteinA.wrl.ply',
                          '3d/converted/6VW1_ligand_bondsA.wrl.ply'),
                         ('3d/converted/6VW1_ligandB.wrl.ply', '3d/converted/6VW1_proteinB.wrl.ply',
                          '3d/converted/6VW1_ligand_bondsB.wrl.ply'),

                         ('3d/converted/6VYB_ligandA.wrl.ply', '3d/converted/6VYB_proteinA.wrl.ply',
                          '3d/converted/6VYB_ligand_bondsA.wrl.ply')
                            ,
                         ('3d/converted/6VYB_ligandB.wrl.ply', '3d/converted/6VYB_proteinB.wrl.ply',
                          '3d/converted/6VYB_ligand_bondsB.wrl.ply'),

                         ])
        for it in iterator:
            ligandPath, proteinPath, ligand_bondsPath = it
            print(it)

            fakes = []
            for i in range(2):
                lg, pp = prepare_data(ligandPath, proteinPath, sample_pointsLignads=int(1024 / progressive_samples[i]))

                ligand_bonds = get_points_mesh(ligand_bondsPath)

                concat = True
                fake, cont_loss = train_generator_content(np.array([lg]), np.array([pp]), ligand_bonds, concat=concat)
                fakes.append(tf.squeeze(fake))
                print('G_loss1', cont_loss)
            fake = tf.concat(fakes, axis=0)
            obb = get_bounding_box_points(fake)
            pcdProteinpoints = get_points_full(proteinPath)
            ligandPoints = get_points_full(ligand_bondsPath)

            fakes2 = []
            for i in range(1):
                try:
                    lg = ligandPoints[
                        obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(ligandPoints))]
                    pp = pcdProteinpoints[
                        obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(pcdProteinpoints))]
                    print('calculated bounding boxes')
                    lg = lg[
                        np.random.choice(np.arange(len(lg)), size=int(1024 / progressive_samples[i]), replace=False)]
                    pp = pp[np.random.choice(np.arange(len(pp)), size=1024, replace=False)]

                    concat = True
                    fake2, cont_loss2 = train_generator_content(np.array([lg]), np.array([pp]), ligand_bonds,
                                                                concat=concat)
                    fakes2.append(tf.squeeze(fake2))
                    try:
                        pcdfake = o3d.geometry.PointCloud()
                        pcdfake.points = o3d.utility.Vector3dVector(fake[0])
                        pcdfake.estimate_normals()

                        name = proteinPath.split('_')[0].replace('3d/converted/', '')
                        chain = proteinPath.split('_')[-1].replace('protein', '').replace('.wrl.ply', '')
                        print('saving',
                              "generated/{}_chain{}_{}.ply".format(name, chain, str(tf.keras.backend.get_value(
                                  ckpt.step))))
                        o3d.io.write_point_cloud(
                            "generated/{}_chain{}_{}.ply".format(name, chain, str(tf.keras.backend.get_value(
                                ckpt.step))), pcdfake)



                    except Exception as exc:
                        print('save error', exc)

                    print('G_loss2', cont_loss2)
                except Exception as exc:
                    print(exc)
            if len(fakes2) > 0:
                fake2 = tf.concat(fakes2, axis=0)
                fake = tf.concat([fake, fake2], axis=0)
            ckpt.step.assign_add(1)
            print(cont_loss)
            if int(ckpt.step) % iters_per_checkpoint == 0:
                print('G_loss', cont_loss)
                fake = np.array([fake])
                with writer.as_default():
                    out_mes_vertices = []
                    out_mesh_faceses = []
                    tf.summary.scalar("G_loss", cont_loss, step=ckpt.step)
                    tf.summary.scalar("G_loss_stacked", cont_loss2, step=ckpt.step)

                    ligandcolor = colors[np.full(len(get_points_full(ligandPath)), 0)]
                    c = []
                    c.extend(ligandcolor)
                    mesh_summary.mesh('ground_ligand', np.array([get_points_full(ligandPath)]), step=ckpt.step,
                                      colors=[c])

                    ligandcolor = colors[np.full(len(np.array(fake[0])), 0)]
                    c = []
                    c.extend(ligandcolor)
                    try:
                        pcdfake = o3d.geometry.PointCloud()
                        pcdfake.points = o3d.utility.Vector3dVector(fake[0])
                        pcdfake.estimate_normals()

                        name = proteinPath.split('_')[0].replace('3d/converted/', '')
                        chain = proteinPath.split('_')[-1].replace('protein', '').replace('.wrl.ply', '')
                        print('saving',
                              "generated/{}_chain{}_{}.ply".format(name, chain, str(tf.keras.backend.get_value(
                                  ckpt.step))))
                        o3d.io.write_point_cloud(
                            "generated/{}_chain{}_{}.ply".format(name, chain, str(tf.keras.backend.get_value(
                                ckpt.step))), pcdfake)



                    except Exception as exc:
                        print('save error', exc)

                    mesh_summary.mesh('pred_ligand', fake, step=ckpt.step, colors=[c])

                    ligandcolor = colors[np.full(len(get_points_full(ligand_bondsPath)), 0)]
                    proteincolor = colors[np.full(len(get_points_full(proteinPath)), 1)]
                    c = []
                    c.extend(ligandcolor)
                    c.extend(proteincolor)

                    mesh_summary.mesh('ground_both', tf.concat(
                        [np.array([get_points_full(proteinPath)]), np.array([get_points_full(ligand_bondsPath)])],
                        axis=1),
                                      step=ckpt.step, colors=[c])

                    ligandcolor = colors[np.full(len(np.array(fake[0])), 0)]
                    proteincolor = colors[np.full(len(get_points_full(proteinPath)), 1)]
                    c = []
                    c.extend(ligandcolor)
                    c.extend(proteincolor)

                    mesh_summary.mesh('pred_both', tf.concat([fake, np.array([get_points_full(proteinPath)])], axis=1),
                                      step=ckpt.step, colors=[c])
                    mesh_summary.mesh('input', np.array([tf.concat([lg, pp], axis=0)]), step=ckpt.step)

                    writer.flush()
                try:
                    save_path = manager.save()

                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

                except Exception as excx:
                    print(excx)
