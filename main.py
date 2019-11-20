import time
import traceback
import os
import gin
import numpy as np
import copy

import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torchvision.models as models

from trainer import KDMultiTaskTrainer
from models import MultiTaskModel
from dataset import CustomDatasetFromImages

from torch.optim.lr_scheduler import ReduceLROnPlateau


# Hacks for Reproducibility
seed = 3
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

@gin.configurable
def run(batch_size, epochs, val_split, num_workers, print_every,
        trainval_csv_path, test_csv_path, model_type, tasks, lr, weight_decay, 
        momentum, dataset_dir, distill_temp):

    train_dataset = CustomDatasetFromImages(trainval_csv_path, data_dir = dataset_dir)
    val_from_images = CustomDatasetFromImages(test_csv_path, data_dir = dataset_dir)

    dset_len = len(val_from_images)
    val_size = int(val_split * dset_len)
    test_size = int(0.15 * dset_len)
    train_size = dset_len - val_size - test_size


    train_dataset_small, val_dataset, test_dataset = torch.utils.data.random_split(val_from_images,
                                                               [train_size,
                                                                val_size,
                                                                test_size])
    # Load opth labelled data 
    train_loader_small = torch.utils.data.DataLoader(dataset=train_dataset_small,
                                               batch_size=batch_size,
                                               pin_memory=False,
                                               drop_last=True,
                                               shuffle=True,
                                               num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             pin_memory=False,
                                             drop_last=True,
                                             shuffle=True, 
                                             num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              pin_memory=False,
                                              drop_last=True,
                                              shuffle=True,
                                              num_workers=num_workers)

    # Load unlabelled data
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=2 * batch_size,
                                               pin_memory=False,
                                               drop_last=True,
                                               shuffle=True,
                                               num_workers=num_workers)
    lang = train_dataset.get_lang()

    if model_type == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_type == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_type == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif model_type == 'vgg19':
        model = models.vgg19(pretrained=True)

    model = MultiTaskModel(model, vocab_size = lang.n_words, model_type=model_type)
    model = nn.DataParallel(model)

    kd_model = models.resnet50(pretrained=True)
    kd_model = MultiTaskModel(kd_model, vocab_size = lang.n_words, model_type = model_type)
    kd_model = nn.DataParallel(kd_model)

    model.load_state_dict(torch.load('/data2/sachelar/models/pretrained_patient_level_filtered/best_model.pt'))

    print(kd_model)
    print(model)
    print(test_csv_path)

    kd_model = kd_model.to('cuda')
    model = model.to('cuda')

    criterion = nn.CrossEntropyLoss()

    # =============================== PRE-TRAIN KD MODEL on small labelled data ========================
    optimizer = torch.optim.SGD(model.parameters(),
                                weight_decay=weight_decay,
                                momentum=momentum,
                                lr=lr,
                                nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer,
                                  factor=0.5,
                                  patience=3,
                                  min_lr=1e-7,
                                  verbose=True)
    trainset_percent = (1 - val_split - 0.15)
    trainer = KDMultiTaskTrainer(model, kd_model, optimizer, scheduler, criterion,
            tasks, epochs, lang, print_every =  print_every, trainset_split = trainset_percent, distill_temp = distill_temp, kd_type = 'nll_only')
    trainer.train(train_loader_small, val_loader)

    # Load best KD model into model
    kd_model.load_state_dict(torch.load(os.path.join(trainer.save_location_dir,'best_model.pt')))

    val_loss, total_d_acc, total_acc, bleu, total_f1,total_recall, total_precision, sent_gt, sent_pred, total_topk,per_disease_topk, per_disease_bleu, total_cm = trainer.validate(test_loader)

    with open(trainer.output_log, 'a+') as out:
        print('Test Loss:{:.8f}\tAcc:{:.8f}\tDAcc:{:.8f}\tBLEU:{:.8f}'.format(val_loss, total_acc, total_d_acc, bleu), file=out)
        print('total_topk',total_topk, file=out)
        print('per_disease_topk', per_disease_topk, file=out)
        print('per_disease_bleu', per_disease_bleu, file=out)
        print(total_cm, file=out)
        for k in np.random.choice(list(range(len(sent_gt))), size=10, replace=False):
            print(sent_gt[k], file=out)
            print(sent_pred[k], file=out)
            print('---------------------', file=out)

    # =============================== TRAIN MODEL WITH KD on small labelled data ========================
    optimizer = torch.optim.SGD(model.parameters(),
                                weight_decay=weight_decay,
                                momentum=momentum,
                                lr=lr,
                                nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer,
                                  factor=0.5,
                                  patience=3,
                                  min_lr=1e-7,
                                  verbose=True)
    trainset_percent = (1 - val_split - 0.15)

    trainer = KDMultiTaskTrainer(model, kd_model, optimizer, scheduler, criterion,
            tasks, epochs, lang, print_every =  print_every, trainset_split = trainset_percent, distill_temp = distill_temp, kd_type = 'full')
    trainer.train(train_loader_small, val_loader)

    # Load best model into KD model
    kd_model.load_state_dict(torch.load(os.path.join(trainer.save_location_dir,'best_model.pt')))

    val_loss, total_d_acc, total_acc, bleu, total_f1,total_recall, total_precision, sent_gt, sent_pred, total_topk,per_disease_topk, per_disease_bleu, total_cm = trainer.validate(test_loader)

    with open(trainer.output_log, 'a+') as out:
        print('Test Loss:{:.8f}\tAcc:{:.8f}\tDAcc:{:.8f}\tBLEU:{:.8f}'.format(val_loss, total_acc, total_d_acc, bleu), file=out)
        print('total_topk',total_topk, file=out)
        print('per_disease_topk', per_disease_topk, file=out)
        print('per_disease_bleu', per_disease_bleu, file=out)
        print(total_cm, file=out)
        for k in np.random.choice(list(range(len(sent_gt))), size=10, replace=False):
            print(sent_gt[k], file=out)
            print(sent_pred[k], file=out)
            print('---------------------', file=out)

    # =============================== TRAIN WITH KD MODEL on unlabelled ========================
    optimizer = torch.optim.SGD(model.parameters(),
                                weight_decay=weight_decay,
                                momentum=momentum,
                                lr = lr,
                                nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer,
                                  factor=0.5,
                                  patience=3,
                                  min_lr=1e-7,
                                  verbose=True)
    epochs = 20
    print_every = 100

    # Init model with best KD model weights

    trainer = KDMultiTaskTrainer(model, kd_model, optimizer, scheduler, criterion, tasks, epochs, lang, print_every =  print_every, trainset_split = trainset_percent, distill_temp = distill_temp, kd_type = 'kd_only')
    trainer.train(train_loader, val_loader)

    val_loss, total_d_acc, total_acc, bleu, total_f1,total_recall, total_precision, sent_gt, sent_pred, total_topk,per_disease_topk, per_disease_bleu, total_cm = trainer.validate(test_loader)

    with open(trainer.output_log, 'a+') as out:
        print('Test Loss:{:.8f}\tAcc:{:.8f}\tDAcc:{:.8f}\tBLEU:{:.8f}'.format(val_loss, total_acc, total_d_acc, bleu), file=out)
        print('total_topk',total_topk, file=out)
        print('per_disease_topk', per_disease_topk, file=out)
        print('per_disease_bleu', per_disease_bleu, file=out)
        print(total_cm, file=out)
        for k in np.random.choice(list(range(len(sent_gt))), size=10, replace=False):
            print(sent_gt[k], file=out)
            print(sent_pred[k], file=out)
            print('---------------------', file=out)

    trainer.test(test_loader)

if __name__ == "__main__":
    temp_configs = [0.5, 1.0, 5.0, 10.0, 100]
    val_configs = [0.15, 0.25, 0.4, 0.55, 0.7]
    # val_configs = [0.15, 0.20.7]
    # task_configs = [[0], [1],[2],[0,1],[1,2],[0,2], [0, 1, 2]]
    task_configs = [[0], [1],[2]]

    # task_configs = [[1]]
    # task_configs = [[1],[2],[0,1],[1,2],[0,2], [0, 1,2]]
    # task_configs = [[0,1,2]]
    for d in temp_configs:
        for v in val_configs:
            for t in task_configs:
                try:
                    print("Running", (1 - v - 0.15), t, d)
                    gin.parse_config_file('config_small.gin')
                    gin.bind_parameter('run.val_split', v)
                    gin.bind_parameter('run.tasks', t)
                    gin.bind_parameter('run.distill_temp',d)
                    run()
                    gin.clear_config()
                except Exception as ex:
                    with open('error.log', 'a') as err_out:
                        print("Error occured", (1-v-0.15),t,d, file=err_out)
                        print(''.join(traceback.format_exception(etype=type(ex),
                            value=ex, tb=ex.__traceback__)), file=err_out)
                    time.sleep(5)

