import torch
import numpy as np
from sklearn import metrics
from torch.utils.data import TensorDataset

from myow import MetricLogger
from . import compute_representations, batch_iter



def prepare_data(dataset, transform):
    fr, sleep_state = dataset.x, dataset.y
    fr = transform(fr)
    return [fr, sleep_state]


def train_classifer(classifier, optimizer, train_data, val_data, test_data, device, batch_size=512):
    def train(classifier, train_data, optimizer):
        classifier.train()

        # C) BALANCED
        class_sample_count = torch.unique(train_data[1], return_counts=True)[1]
        class_weights = 1 / class_sample_count
        weights = class_weights[train_data[1]]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights, num_samples=5000)
        train_loader = torch.utils.data.DataLoader(TensorDataset(*train_data), batch_size=batch_size, drop_last=True,
                                                   sampler=sampler, num_workers=0)

        for step in range(100):
            #for x, y in batch_iter(*train_data, batch_size=batch_size):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                # forward
                optimizer.zero_grad()
                pred_logits = classifier(x)

                # loss and backprop
                loss = criterion(pred_logits, y)
                loss.backward()
                optimizer.step()

    def test(classifier, data):
        classifier.eval()
        x, y = data
        # feed to network and classifier
        pred_logits = classifier(x.to(device)).detach().cpu()
        # compute acc
        _, pred_class = torch.max(pred_logits, 1)
        acc = pred_class.eq(y.view_as(pred_class)).sum().item() / y.size(0)
        # compute f1-score
        f1 = metrics.f1_score(y.numpy(), pred_class.numpy(), average='weighted') if pred_class.sum() > 0 else 0
        return acc, f1

    # train
    criterion = torch.nn.CrossEntropyLoss()
    train(classifier, train_data, optimizer)

    # eval
    train_acc, train_f1 = test(classifier, train_data)
    val_acc, val_f1 = test(classifier, val_data)
    test_acc, test_f1 = test(classifier, test_data)

    return (train_acc, val_acc, test_acc), (train_f1, val_f1, test_f1)


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
    f1 = MetricLogger(early_stopping=True, max=True)

    # performe HP search for weight decay
    for weight_decay in 2.0 ** np.arange(-10, 10):
        classifier = torch.nn.Linear(num_feats, 3).to(device)
        optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=0.01, weight_decay=weight_decay)

        (train_acc, val_acc, test_acc), (train_f1, val_f1, test_f1) = \
            train_classifer(classifier, optimizer, train_data, val_data, test_data,
                                   device=device, batch_size=512)

        acc.update(train=train_acc, val=val_acc, test=test_acc, step=weight_decay)
        f1.update(train=train_f1, val=val_f1, test=test_f1, step=weight_decay)

    # get accuracies corresponding to best val_acc
    train_acc, val_acc, test_acc = acc.train_max, acc.val_max, acc.test_max
    train_f1, val_f1, test_f1 = f1.hist(acc.step_minmax)

    print(test_f1)

    # log to writer
    writer.add_scalar('arousal_state/acc_train', train_acc, epoch)
    writer.add_scalar('arousal_state/f1_train', train_f1, epoch)
    writer.add_scalar('arousal_state/acc_val', val_acc, epoch)
    writer.add_scalar('arousal_state/f1_val', val_f1, epoch)
    writer.add_scalar('arousal_state/acc_test', test_acc, epoch)
    writer.add_scalar('arousal_state/f1_test', test_f1, epoch)
