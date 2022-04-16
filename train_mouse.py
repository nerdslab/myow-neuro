from absl import flags, app
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from myow import MLP, MYOW, MLP3
from myow.data import RodentNeuralDataset
from myow.transforms import get_neural_transform
from myow.samplers import RelativePositioningDataLoader
from myow.utils import seed_everything, collect_params
from myow.tasks.train_sleep_classifier import linear_evaluate

# General settings,
flags.DEFINE_integer('seed', None, 'Random seed.')
flags.DEFINE_integer('num_workers', 4, 'Number of CPU workers for training.')
flags.DEFINE_string('root', './data/rodent', 'Where the dataset reside.')
flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')
flags.DEFINE_enum('rodent', 'mouse', ['mouse', 'rat'], 'Primate name.')

# Settings for backbone
flags.DEFINE_list('encoder_hidden_layers', [64, 64, 64], 'Sizes of hidden layers in encoder.')
flags.DEFINE_integer('representation_size', 32, 'Representation size.')

# Settings for logging and checkpointing
flags.DEFINE_integer('checkpoint_epochs', 1, 'Save checkpoint at at every checkpoint_epochs.')
flags.DEFINE_integer('log_scalar_steps', 100, 'Log scalars at every log_scalar_steps.')
flags.DEFINE_integer('linear_eval_epochs', 800, 'Perform linear eval at every linear_eval_epochs.')
flags.DEFINE_float('weight_decay', 2e-5, 'The value of the weight decay for training.')

# Settings for training.
flags.DEFINE_integer('num_epochs', 800, 'The number of training epochs.')
flags.DEFINE_integer('batch_size', 512, 'The number of images in each batch during training.')

# Mined views term
flags.DEFINE_integer('projector_hidden_size', 256, 'Hidden size of second projector.')
flags.DEFINE_integer('projector_output_size', 16, 'Output size of second projector.')
flags.DEFINE_float('mined_weight', 0.5, 'The base loss weight for myow term.')  # todo between 0 and 1
flags.DEFINE_integer('mined_weight_warmup_epochs', 10, 'Warmup period.')

# Settings for view mining
flags.DEFINE_integer('knn_nneighs', 5, 'Number of nearest neighbors.')

# Settings for lr, momentum and loss weight.
flags.DEFINE_float('lr', 0.002, 'The base learning rate for model training.')
flags.DEFINE_float('mm', 0.98, 'The base momentum for moving average.')

# Settings for schedulers.
flags.DEFINE_integer('lr_warmup_epochs', 100, 'Warmup period for learning rate.')

# Transforms
flags.DEFINE_integer('pos_k', 3, 'Max lookahead.')
flags.DEFINE_integer('neg_k', 150, 'Max lookahead.')
flags.DEFINE_float('noise_sigma', 0.2, 'Noise sigma.', lower_bound=0.)
flags.DEFINE_float('noise_apply_p', 0., 'Probability of applying noise.', lower_bound=0., upper_bound=1.)
flags.DEFINE_float('dropout_p', 0.8, 'Dropout probability.', lower_bound=0., upper_bound=1.)
flags.DEFINE_float('dropout_apply_p', 0.9, 'Probability of applying dropout.', lower_bound=0., upper_bound=1.)
flags.DEFINE_float('pepper_p', 0.0, 'Pepper probability.', lower_bound=0., upper_bound=1.)
flags.DEFINE_float('pepper_sigma', 0.3, 'Pepper sigma.', lower_bound=0.)
flags.DEFINE_float('pepper_apply_p', 0.0, 'Probability of applying pepper.', lower_bound=0., upper_bound=1.)

FLAGS = flags.FLAGS

def main(argv):
    # set random seed
    seed_everything(seed=FLAGS.seed)

    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device}.')
    ###################
    # Prepare dataset #
    ###################
    # load dataset for ssl (pre-train)
    dataset = RodentNeuralDataset(FLAGS.root, FLAGS.rodent, 'trainval')

    # prepare datasets for linear eval
    train_dataset = RodentNeuralDataset(FLAGS.root, FLAGS.rodent, 'train')
    val_dataset = RodentNeuralDataset(FLAGS.root, FLAGS.rodent, 'val')
    test_dataset = RodentNeuralDataset(FLAGS.root, FLAGS.rodent, 'test')

    # prepare dataset for visualization
    full_dataset = RodentNeuralDataset(FLAGS.root, FLAGS.rodent, None)

    # get transforms
    fr_mean, fr_std = dataset.get_mean_std('firing_rates')

    transform = get_neural_transform(
        randomized_dropout=dict(p=FLAGS.dropout_p, apply_p=FLAGS.dropout_apply_p,
                                same_on_trial=False, same_on_batch=False),
        pepper=dict(p=FLAGS.pepper_p, c=FLAGS.pepper_sigma, apply_p=FLAGS.pepper_apply_p,
                    same_on_trial=False, same_on_batch=False),
        noise=dict(std=FLAGS.noise_sigma, apply_p=FLAGS.noise_apply_p,
                   same_on_trial=False, same_on_batch=False),
        normalize=dict(mean=fr_mean, std=fr_std),
        noise_after_norm=False
    )

    normalize = get_neural_transform(normalize=dict(mean=fr_mean, std=fr_std))

    # get dataloader
    dataloader = RelativePositioningDataLoader(dataset, batch_size=FLAGS.batch_size, drop_last=True, shuffle=True,
                                               pos_kmin=0, pos_kmax=FLAGS.pos_k, neg_k=FLAGS.neg_k,
                                               num_workers=FLAGS.num_workers, transform=transform,
                                               precompute_neg=True, persistent_workers=FLAGS.num_workers !=0)

    #################
    # Prepare model #
    #################
    # Build encoder network
    input_size = (fr_std != 0).sum()
    representation_size = FLAGS.representation_size
    hidden_layers = map(int, FLAGS.encoder_hidden_layers)
    encoder = MLP([input_size, *hidden_layers, FLAGS.representation_size], batchnorm=True)

    # Create projectors
    projector = nn.Identity()
    projector_m = MLP3(representation_size, FLAGS.projector_output_size, hidden_size=FLAGS.projector_hidden_size)
    # Create predictors
    predictor = MLP3(representation_size, representation_size, hidden_size=FLAGS.projector_hidden_size)  # used to predict across augmented views
    predictor_m = MLP3(FLAGS.projector_output_size, FLAGS.projector_output_size, hidden_size=FLAGS.projector_hidden_size)  # used to predict across mined views

    # This is used to provide an evaluation of the representation quality during training.
    model = MYOW(encoder, projector, projector_m, predictor, predictor_m, layout='cascaded')
    model.to(device)

    # define optimizer
    params = collect_params(model.trainable_modules, exclude_bias_and_bn=True)
    optimizer = torch.optim.SGD(params, lr=FLAGS.lr, momentum=FLAGS.mm, weight_decay=FLAGS.weight_decay)

    # define tensorboard writer
    writer = SummaryWriter(FLAGS.logdir)

    ##############################
    # Define scheduler functions #
    ##############################
    step = 0

    # compute total number of steps
    num_steps_per_epoch = dataloader.num_examples // FLAGS.batch_size
    total_steps = num_steps_per_epoch * FLAGS.num_epochs
    lr_warmup_steps = num_steps_per_epoch * FLAGS.lr_warmup_epochs
    mined_weight_warmup_steps = num_steps_per_epoch * FLAGS.mined_weight_warmup_epochs

    # define schedulers
    def update_learning_rate(step, max_val=FLAGS.lr, total_steps=total_steps, warmup_steps=lr_warmup_steps):
        if 0 <= step <= warmup_steps:
            return max_val * step / warmup_steps + 1e-9
        else:
            return max_val * (1 + np.cos((step - warmup_steps) * np.pi / (total_steps - warmup_steps))) / 2

    def update_momentum(step, max_val=FLAGS.mm, total_steps=total_steps):
        return 1 - max_val * (1 + np.cos(step * np.pi / total_steps)) / 2

    def update_weight(step, max_val=FLAGS.mined_weight, warmup_steps=mined_weight_warmup_steps):
        if 0 <= step <= warmup_steps:
            return max_val * step / warmup_steps + 1e-9
        else:
            return max_val

    #################
    # Training loop #
    #################
    def train(step):
        for inputs in tqdm(dataloader, leave=False):
            optimizer.zero_grad()

            # update params
            lr = update_learning_rate(step)
            mm = update_momentum(step)
            weight = update_weight(step)

            view_2 = inputs['view_2'].to(device)
            out_2 = model(online_view=view_2, target_view=view_2)

            data = inputs['data'].to(device)
            out = model(online_view=data, target_view=data)

            # Augmented Views
            view_1_index = inputs['view_1_index'].to(device)
            online_q, target_z = out.online.q[view_1_index], out_2.target.z

            # Augmented Views (Symmetric)
            online_q_s, target_z_s = out_2.online.q, out.target.z[view_1_index]

            # Mining
            online_y, target_candidate_y = out.online.y, out.target.y
            online_y = online_y[view_1_index]

            # Compute cosine distance
            online_y = F.normalize(online_y, dim=-1, p=2)
            target_candidate_y = F.normalize(target_candidate_y, dim=-1, p=2)
            dist = - torch.einsum('nc,kc->nk', [online_y, target_candidate_y])

            # remove ineligible candidates
            row, col = inputs['ccand_edge_index'].to(device)
            n_mask = torch.unique(row)
            n_idx = torch.zeros(target_candidate_y.size(0), dtype=torch.long)
            n_idx[n_mask] = torch.arange(n_mask.size(0))
            dist[n_idx[row], col] = torch.finfo(dist.dtype).max

            # get k nearest neighbors
            _, topk_index = torch.topk(dist, k=FLAGS.knn_nneighs, largest=False)

            # randomly select mined view out the k nearest neighbors
            mined_view_id = topk_index[torch.arange(topk_index.size(0), dtype=torch.long, device=dist.device),
                                       torch.randint(FLAGS.knn_nneighs, size=(topk_index.size(0),))]

            # Mined views
            online_q_m = out.online.q_m[view_1_index]
            target_v = out.target.v[mined_view_id]

            # loss
            aug_loss = 1 - 0.5 * cosine_similarity(online_q, target_z.detach(), dim=-1).mean() \
                       - 0.5 * cosine_similarity(online_q_s, target_z_s.detach(), dim=-1).mean()
            mined_loss = 1 - cosine_similarity(online_q_m, target_v.detach(), dim=-1).mean()

            loss = (1 - weight) * aug_loss + weight * mined_loss

            loss.backward()
            # update online network
            optimizer.step()
            # update target network
            model.update_target_network(mm)

            # log scalars
            if step % FLAGS.log_scalar_steps == 0:
                writer.add_scalar('params/lr', lr, step)
                writer.add_scalar('params/mm', mm, step)
                writer.add_scalar('train/loss', loss, step)
                writer.add_scalar('train/aug_loss', aug_loss, step)
                writer.add_scalar('train/mined_loss', mined_loss, step)

            step += 1
        return step

    ###################
    # Evaluation loop #
    ###################
    def test(step):
        encoder = copy.deepcopy(model.online_encoder.eval())
        linear_evaluate(encoder, train_dataset, val_dataset, test_dataset, normalize, writer, device, epoch)

    ######################################
    # Representation space visualization #
    ######################################
    def visualize(step):
        encoder = copy.deepcopy(model.online_encoder.eval())
        # prepare data
        x = normalize(full_dataset.x).to(device)

        # compute representations
        x = x.to(device)
        with torch.inference_mode():
            representations = encoder(x).to('cpu')

        # get metadata
        sleep_state = full_dataset.y.numpy()
        timestep = full_dataset.timestep.numpy()

        # get __seq_next__ to display trajectory
        seq_next = np.zeros(full_dataset.num_samples, dtype='U8')
        seq_next[full_dataset.edge_index[0]] = full_dataset.edge_index[1].numpy().astype('U8')

        # combine metadata
        metadata = np.column_stack([sleep_state, timestep, seq_next]).tolist()
        metadata_header = ['sleep_state', 'timestep', '__seq_next__']

        # log to tensorboard
        writer.add_embedding(representations, metadata=metadata, metadata_header=metadata_header, global_step=step)


    # start training
    for epoch in tqdm(range(1, FLAGS.num_epochs+1), leave=False):
        step = train(step)
        if epoch % FLAGS.linear_eval_epochs == 0:
            test(step)

    visualize(step)


if __name__ == "__main__":
    print(" /\_/\ \n( o.o )\n > ^ <")
    app.run(main)
