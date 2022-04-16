import torch
import numpy as np

from myow import MetricLogger
from . import compute_representations, batch_iter


def prepare_data(dataset, transform):
    fr, angles = dataset.x, dataset.angle
    cos_sin = torch.column_stack([torch.cos(angles), torch.sin(angles)])
    fr = transform(fr)
    return [fr, angles, cos_sin]


def train_classifer(classifier, optimizer, train_data, val_data, test_data, device, batch_size=512):
    def train(classifier, train_data, optimizer):
        classifier.train()

        for step in range(100):
            for x, target_angle, target_cos_sin in batch_iter(*train_data, batch_size=batch_size):
                x, target_cos_sin = x.to(device), target_cos_sin.to(device)

                # forward
                optimizer.zero_grad()
                pred_logits = classifier(x)

                # loss and backprop
                loss = criterion(pred_logits, target_cos_sin)
                loss.backward()
                optimizer.step()

    def test(classifier, data):
        classifier.eval()
        x, target_angle, target_cos_sin = data
        target_angle = target_angle.squeeze()
        # feed to network and classifier
        pred_logits = classifier(x.to(device)).detach().cpu()
        # compute acc
        pred_angles = torch.atan2(pred_logits[:, 1], pred_logits[:, 0])
        pred_angles[pred_angles < 0] = pred_angles[pred_angles < 0] + 2 * np.pi
        diff_angles = torch.abs(pred_angles - target_angle)
        diff_angles[diff_angles > np.pi] = torch.abs(diff_angles[diff_angles > np.pi] - 2 * np.pi)
        acc = (diff_angles < (np.pi / 8)).sum() / diff_angles.size(0)
        delta_acc = (diff_angles < (3 * np.pi / 16)).sum() / diff_angles.size(0)
        return acc, delta_acc

    # train
    criterion = torch.nn.MSELoss()
    train(classifier, train_data, optimizer)

    # eval
    train_acc, train_delta_acc = test(classifier, train_data)
    val_acc, val_delta_acc = test(classifier, val_data)
    test_acc, test_delta_acc = test(classifier, test_data)

    return (train_acc, val_acc, test_acc), (train_delta_acc, val_delta_acc, test_delta_acc)


def linear_evaluate(encoder, train_dataset, val_dataset, test_dataset, transform, writer, device, epoch=None):
    # prepare data
    train_data = prepare_data(train_dataset, transform)
    val_data = prepare_data(val_dataset, transform)
    test_data = prepare_data(test_dataset, transform)

    # compute representations
    train_data[0] = compute_representations(encoder, train_data[0], device)
    val_data[0] = compute_representations(encoder, val_data[0], device)
    test_data[0] = compute_representations(encoder, test_data[0], device)
    num_feats = train_data[0].size(1)

    # set meters
    acc = MetricLogger(early_stopping=True, max=True)
    delta_acc = MetricLogger(early_stopping=True, max=True)

    # performe HP search for weight decay
    for weight_decay in 2.0 ** np.arange(-10, 10):
        classifier = torch.nn.Linear(num_feats, 2).to(device)
        optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=0.01, weight_decay=weight_decay)

        (train_acc, val_acc, test_acc), (train_delta_acc, val_delta_acc, test_delta_acc) = \
            train_classifer(classifier, optimizer, train_data, val_data, test_data,
                            device=device, batch_size=512)

        acc.update(train=train_acc, val=val_acc, test=test_acc, step=weight_decay)
        delta_acc.update(train=train_delta_acc, val=val_delta_acc, test=test_delta_acc, step=weight_decay)

    # get accuracies corresponding to best val_acc
    train_acc, val_acc, test_acc = acc.train_max, acc.val_max, acc.test_max
    train_delta_acc, val_delta_acc, test_delta_acc = delta_acc.hist(acc.step_minmax)

    # log to writer
    writer.add_scalar('angle_pred/acc_train', train_acc, epoch)
    writer.add_scalar('angle_pred/delta_acc_train', train_delta_acc, epoch)
    writer.add_scalar('angle_pred/acc_val', val_acc, epoch)
    writer.add_scalar('angle_pred/delta_acc_val', val_delta_acc, epoch)
    writer.add_scalar('angle_pred/acc_test', test_acc, epoch)
    writer.add_scalar('angle_pred/delta_acc_test', test_delta_acc, epoch)

    return test_acc, test_delta_acc
