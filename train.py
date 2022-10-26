# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ast
import argparse
import numpy as np
import time

from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common import Tensor
from mindspore.common import set_seed
from mindspore.communication.management import init
from mindspore.nn.optim import Adam
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import config
from src.data.dataset import create_dataset_coco
from src.data.dataset import flip_pairs
from src.network_with_loss import JointsMSELoss, OmniPoseWithLoss
from src.utils.callback import LossMonitorV2
from src.utils.coco import evaluate
from src.utils.transforms import flip_back
from src.utils.inference import get_final_preds
from src.OmniPose import create_model

if config["MODELARTS"]["IS_MODEL_ARTS"]:
    import moxing as mox

set_seed(config["GENERAL"]["TRAIN_SEED"])


def get_lr(begin_epoch,
           total_epochs,
           steps_per_epoch,
           lr_init=0.001,
           factor=0.1,
           epoch_number_to_drop=(170, 200)
           ):
    """
    get_lr
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    step_number_to_drop = [steps_per_epoch * x for x in epoch_number_to_drop]
    for i in range(int(total_steps)):
        if i in step_number_to_drop:
            lr_init = lr_init * factor
        lr_each_step.append(lr_init)
    current_step = steps_per_epoch * begin_epoch
    lr_each_step = np.array(lr_each_step, dtype=np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser(description="omnipose training")
    parser.add_argument('--data_url', required=False,
                        default=None, help='data_url to download dataset.')
    parser.add_argument('--train_url', required=False,
                        default=None, help='train_url to upload results.')
    parser.add_argument('--device_id', required=False, default=0,
                        type=int, help='Device id to be used.')
    parser.add_argument('--device_num', required=False, default=1,
                        type=int, help='Number of devices for training.')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--run_distribute', type=ast.literal_eval,
                        default=False, help='Whether or not run distribute, defalut false.')
    parser.add_argument('--is_model_arts', type=ast.literal_eval,
                        default=False, help='Whether or not run on model_arts, defalut false.')
    parser.add_argument('--auto_dataset', type=ast.literal_eval,
                        default=False, help='Whether or not auto download dataset, default false.')
    args = parser.parse_args()
    
    return args


def train(args):
    device_id = args.device_id
    config["TRAIN"]["DEVICE_NUM"] = args.device_num
    config["TRAIN"]["DEVICE_TARGET"] = args.device_target
    config["GENERAL"]["RUN_DISTRIBUTE"] = args.run_distribute
    config["GENERAL"]["AUTO_DATASET"] = args.auto_dataset
    config["MODELARTS"]["IS_MODEL_ARTS"] = args.is_model_arts

    if config["GENERAL"]["RUN_DISTRIBUTE"] or config["MODELARTS"]["IS_MODEL_ARTS"]:
        device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config["TRAIN"]["DEVICE_TARGET"],
                        save_graphs=False,
                        device_id=device_id)
    
    if config["GENERAL"]["RUN_DISTRIBUTE"]:
        init()
        rank = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        rank = 0
        device_num = 1

    if config["MODELARTS"]["IS_MODEL_ARTS"]:
        pip_lock_file = "/cache/pip_lock.txt"
        if rank == 0:
            tag = True
            try:
                # pip_cmd_0 = "pip install --upgrade pip --user"
                # pip_cmd_1 = "pip install pycocotools --user"
                # os.system(pip_cmd_0)
                # os.system(pip_cmd_1)
                pip_cmd = "pip install pycocotools"
                os.system(pip_cmd)
            except Exception as e:
                tag = True
                print("install pycocotools error!\n{}".format(e), flush=True)
            if tag:
                with open(pip_lock_file, "w") as fp:
                    fp.write("1\n")
        else:
            while True:
                if os.path.exists(pip_lock_file):
                    break
                time.sleep(10)

    if config["MODELARTS"]["IS_MODEL_ARTS"] and not config["GENERAL"]["AUTO_DATASET"]:
        data_lock_file = "/cache/data_lock.txt"
        if rank == 0:
            tag = True
            data_dir = config["MODELARTS"]["CACHE_INPUT"]
            model_dir = config["MODELARTS"]["CACHE_OUTPUT"]
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            try:
                mox.file.copy_parallel(src_url=args.data_url, dst_url=data_dir)
            except Exception as e:
                print("moxing download {} to {} failed, error:\n{}".format(
                    args.data_url, data_dir, e), flush=True)
                tag = False
            if tag:
                with open(data_lock_file, "w") as fp:
                    fp.write("1\n")
        else:
            while True:
                if os.path.exists(data_lock_file):
                    break
                time.sleep(10)

    dataset = create_dataset_coco(
        rank=rank, group_size=device_num,
        train_mode=True, num_parallel_workers=config["TRAIN"]["NUM_PARALLEL_WORKERS"])

    m = create_model()
    loss = JointsMSELoss(config["LOSS"]["USE_TARGET_WEIGHT"])
    net_with_loss = OmniPoseWithLoss(m, loss)
    dataset_size = dataset.get_dataset_size()
    lr = Tensor(get_lr(config["TRAIN"]["BEGIN_EPOCH"],
                       config["TRAIN"]["END_EPOCH"],
                       dataset_size,
                       lr_init=config["TRAIN"]["LR"],
                       factor=config["TRAIN"]["LR_FACTOR"],
                       epoch_number_to_drop=config["TRAIN"]["LR_STEP"]))
    optim = Adam(m.trainable_params(), learning_rate=lr, loss_scale=1024.0)
    # optim = Adam(m.trainable_params(), learning_rate=lr)
    time_cb = TimeMonitor(data_size=dataset_size)
    # loss_cb = LossMonitor()
    loss_cb = LossMonitorV2()
    cb = [time_cb, loss_cb]
    if config["TRAIN"]["SAVE_CKPT"]:
        config_ck = CheckpointConfig(
            save_checkpoint_steps=dataset_size, keep_checkpoint_max=50)

        prefix = "device_{}_omnipose_".format(rank)

        if config["MODELARTS"]["IS_MODEL_ARTS"]:
            ckpt_dir = os.path.join(config["MODELARTS"]["CACHE_OUTPUT"], "ckpt_device_{}".format(rank))
        else:
            if not os.path.exists(config["TRAIN"]["CKPT_PATH"]):
                os.mkdir(config["TRAIN"]["CKPT_PATH"])
            ckpt_dir = os.path.join(config["TRAIN"]["CKPT_PATH"], "ckpt_device_{}".format(rank))

        ckpoint_cb = ModelCheckpoint(
            prefix=prefix, directory=ckpt_dir, config=config_ck)
        cb.append(ckpoint_cb)
    model = Model(net_with_loss, optimizer=optim, amp_level="O2")
    epoch_size = config["TRAIN"]["END_EPOCH"] - config["TRAIN"]["BEGIN_EPOCH"]
    print("************ Start training now ************", flush=True)
    print('start training, epoch size: {}'.format(epoch_size), flush=True)
    model.train(epoch_size, dataset, callbacks=cb)

    if config["MODELARTS"]["IS_MODEL_ARTS"] and config["TRAIN"]["SAVE_CKPT"]:
        if not config["GENERAL"]["AUTO_DATASET"]:
            # mox.file.copy_parallel(
            #     src_url=config["MODELARTS"]["CACHE_OUTPUT"], dst_url=args.train_url)
            ckpt_dir = os.path.join(config["MODELARTS"]["CACHE_OUTPUT"], "ckpt_device_{}".format(rank))
            obs_train_dir = os.path.join(args.train_url, "ckpt_device_{}".format(rank))
            try:
                mox.file.copy_parallel(src_url=ckpt_dir, dst_url=obs_train_dir)
                print("Successfully Upload {} to {}".format(ckpt_dir, obs_train_dir), flush=True)
            except Exception as e:
                print("moxing upload {} to {} failed, error: \n{}".format(
                    ckpt_dir, obs_train_dir, e), flush=True)
        else:
            output_dir = "/cache/output/"
            try:
                os.system("cd /cache/script_for_grampus/ && ./uploader_for_npu " + "/cache/output/")
                print("Successfully Upload to {}".format(output_dir), flush=True)
            except Exception as e:
                print("Failed Upload to {}".format(output_dir), flush=True)
    
    return rank


def validate(cfg, val_dataset, model, output_dir, ann_path):
    """
    validate
    """
    model.set_train(False)
    num_samples = val_dataset.get_dataset_size() * cfg["TEST"]["BATCH_SIZE"]
    all_preds = np.zeros((num_samples, cfg["MODEL"]["NUM_JOINTS"], 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 2))
    image_id = []
    idx = 0

    start = time.time()
    for item in val_dataset.create_dict_iterator():
        inputs = item['image'].asnumpy()
        output = model(Tensor(inputs, mstype.float32)).asnumpy()
        if cfg["TEST"]["FLIP_TEST"]:
            inputs_flipped = Tensor(inputs[:, :, :, ::-1], mstype.float32)
            output_flipped = model(inputs_flipped)
            output_flipped = flip_back(output_flipped.asnumpy(), flip_pairs)

            if cfg["TEST"]["SHIFT_HEATMAP"]:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.copy()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        c = item['center'].asnumpy()
        s = item['scale'].asnumpy()
        score = item['score'].asnumpy()
        file_id = list(item['id'].asnumpy())

        preds, maxvals = get_final_preds(output.copy(), c, s)
        num_images, _ = preds.shape[:2]
        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        all_boxes[idx:idx + num_images, 0] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 1] = score
        image_id.extend(file_id)
        idx += num_images
        if idx % 1024 == 0:
            print('{} samples validated in {} seconds'.format(idx, time.time() - start), flush=True)
            start = time.time()

    print(all_preds[:idx].shape, all_boxes[:idx].shape, len(image_id), flush=True)
    _, perf_indicator = evaluate(cfg, all_preds[:idx], output_dir,
                                 all_boxes[:idx], image_id, ann_path)
    # print("{} AP: {}".format(log_info, perf_indicator), flush=True)
    return perf_indicator


def eval(train_url, rank, ckpt_names, ckpt_paths):
    # context.set_context(mode=context.GRAPH_MODE,
    #                     device_target=config["TEST"]["DEVICE_TARGET"],
    #                     # device_id=config["TEST"]["DEVICE_ID"])
    #                     device_id=rank)

    model = create_model()
    valid_dataset = create_dataset_coco(
        train_mode=False,
        num_parallel_workers=config["TEST"]["NUM_PARALLEL_WORKERS"],
        )

    max_ap = 0.0
    max_ckpt_name = "none"

    for ckpt_name, ckpt_path in zip(ckpt_names, ckpt_paths):
        try:
            param_dict = load_checkpoint(ckpt_path)
            load_param_into_net(model, param_dict)
            print("load checkpoint from: {}.".format(ckpt_path, flush=True))

            if config["MODELARTS"]["IS_MODEL_ARTS"]:
                output_dir = os.path.join(config["MODELARTS"]["CACHE_OUTPUT"], "device_{}_result".format(rank))
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                ann_path = config["MODELARTS"]["CACHE_INPUT"]
            else:
                output_dir = config["TEST"]["OUTPUT_DIR"]
                ann_path = config["DATASET"]["ROOT"]
            output_dir = os.path.join(output_dir, ckpt_name)
            ann_path = os.path.join(ann_path, config["DATASET"]["TEST_JSON"])
            AP = validate(config, valid_dataset, model, output_dir, ann_path)
            print("device {}, {}, AP: {}".format(rank, ckpt_name, AP), flush=True)
            
            if AP > max_ap:
                max_ap = AP
                max_ckpt_name = ckpt_name

            if config["MODELARTS"]["IS_MODEL_ARTS"]:
                if not config["GENERAL"]["AUTO_DATASET"]:
                    obs_result_url = os.path.join(train_url, "device_{}_result".format(rank), ckpt_name)
                    try:
                        mox.file.copy_parallel(src_url=output_dir, dst_url=obs_result_url)
                        print("Successfully Upload {} to {}".format(output_dir, obs_result_url), flush=True)
                    except Exception as e:
                        print('moxing upload {} to {} failed, error: \n{}'.format(
                            output_dir, obs_result_url, e), flush=True)                    
        
        except Exception as e:
            print("eval {} {} {} error!\n {}".format(rank, ckpt_name, ckpt_path, e), flush=True)
    
    print("golbal max AP: {}, ckpt_name: {}".format(max_ap, max_ckpt_name), flush=True)


def main():
    print("loading parse...", flush=True)
    args = parse_args()

    rank = train(args)
    
    if config["MODELARTS"]["IS_MODEL_ARTS"]:
        ckpt_dir = os.path.join(config["MODELARTS"]["CACHE_OUTPUT"], "ckpt_device_{}".format(rank))
    else:
        ckpt_dir = os.path.join(config["TRAIN"]["CKPT_PATH"], "ckpt_device_{}".format(rank))
    
    list_files = os.listdir(ckpt_dir)
    ckpt_files = []
    for src_file in list_files:
        if src_file.endswith(".ckpt"):
            ckpt_files.append(src_file)

    ckpt_files = sorted(ckpt_files, key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)

    ckpt_paths = []
    ckpt_names = []
    for ckpt_file in ckpt_files:
        ckpt_paths.append(os.path.join(ckpt_dir, ckpt_file))
        ckpt_names.append(ckpt_file.replace(".ckpt", ""))
    
    eval(args.train_url, rank, ckpt_names, ckpt_paths)

    if config["GENERAL"]["AUTO_DATASET"]:
        output_dir = "/cache/output/"
        try:
            os.system("cd /cache/script_for_grampus/ && ./uploader_for_npu " + "/cache/output/")
            print("Successfully Upload to {}".format(output_dir), flush=True)
        except Exception as e:
            print("Failed Upload to {}".format(output_dir), flush=True)


if __name__ == '__main__':
    main()
