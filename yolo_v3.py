from __future__ import division
import torch
import torch.nn as nn
import datetime
import time
import logging
from utilis import create_module, box_iou, _save_checkpoint
from evaluate import eval_score, get_map
from tensorboardX import SummaryWriter
from torch import optim
import numpy as np
import os


class yolo_v3(nn.Module):
    def __init__(self, params, blocks,
                 weight_file="../4TrainingWeights/yolov3.weights",
                 detection_layers=[81, 93, 105]):
        super().__init__()
        self.blocks = blocks
        self.params = params
        for layer in detection_layers:
            self.blocks[layer]['filters'] = (self.params['num_classes'] +
                                             5) * self.params['num_anchors']
        self.layer_type_dic, self.module_list = create_module(
                                                    self.params, blocks)
        print(self.module_list)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
        if self.params['pretrain_snapshot']:
            state_dic = torch.load(self.params['pretrain_snapshot'])
            self.load_state_dict(state_dic)
        else:
            self.load_weights(weight_file, cust_train_zero=True)
        if self.params['cuda']:
            self = self.cuda()

    def fit(self, train_loader, test_loader, loop_conf=False):
        # inituate a dictionry to store all the logs for tensorboard
        ts_writer = {}

        # create working directory if necessary
        if not os.path.exists(self.params["working_dir"]):
            os.makedirs(self.params["working_dir"])

        date_time_now = str(
            datetime.datetime.now()).replace(" ", "_").replace(":", "_")

        # Create sub_working_dir
        sub_working_dir = os.path.join(self.params["working_dir"] +
                                       self.params['sub_name'] +
                                       date_time_now)

        if not os.path.exists(sub_working_dir):
            os.makedirs(sub_working_dir)
        self.params["sub_working_dir"] = sub_working_dir
        logging.info("sub working dir: %s" % sub_working_dir)

        # Creat tf_summary writer
        ts_writer["tensorboard_writer"] = SummaryWriter(sub_working_dir)
        logging.info("Please using 'python -m tensorboard.main --logdir={} \
                     '".format(sub_working_dir))

        # optimizer
        optimizer_dic = {'sgd': torch.optim.SGD(
                            self.module_list.parameters(),
                            lr=self.params["learning_rate"],
                            momentum=self.params["momentum"],
                            weight_decay=self.params["decay"]),
                         'adam': torch.optim.Adam(
                            self.module_list.parameters(),
                            lr=self.params["learning_rate"],
                            weight_decay=self.params["decay"])}
        optimizer = optimizer_dic[self.params['optimizer'].lower()]

        # initiate global step
        self.params["global_step"] = 0

        # initiate learning rate scheduler
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.params["steps"],
            gamma=self.params["scales"])

        self.train()
        map_results_names = ["best_map", "best_ap", "best_conf",
                             "specific_conf_map", "specific_conf_ap"]
        # Start the training loop
        logging.info("Start training.")
        for epoch in range(self.params["epochs"]):
            save = 1
            eva = 1
            if self.params['loop_epoch'] and epoch > self.params['loop_epoch']:
                loop_conf = True
            for step, samples in enumerate(train_loader):
                if self.params['cuda']:
                    images, labels = (samples["image"].to('cuda'),
                                      samples["label"])
                else:
                    images, labels = samples["image"], samples["label"]

                start_time = time.time()
                self.params["global_step"] += 1

                # Forward and backward
                optimizer.zero_grad()
                batch_size = images.size(0)
                losses = self(images, is_training=True, labels=labels)
                loss = losses[0]
                loss.backward()
                optimizer.step()

                if step > 0 and step % self.params['loss_step'] == 0:
                    _loss = loss.item()
                    duration = float(time.time() - start_time)
                    example_per_second = batch_size / duration
                    lr = optimizer.param_groups[0]['lr']
                    logging.info(
                        "epoch [%.3d] iter = \
                        %d loss = %.2f example/sec = %.3f lr = %.5f " %
                        (epoch, step, _loss, example_per_second, lr)
                    )
                    ts_writer["tensorboard_writer"].add_scalar(
                            "lr", lr, self.params["global_step"])
                    ts_writer["tensorboard_writer"].add_scalar(
                            "example/sec", example_per_second,
                            self.params["global_step"])
                    for i, name in enumerate(self.losses_name):
                        value = _loss if i == 0 else losses[i]
                        ts_writer["tensorboard_writer"].add_scalar(
                                name, value, self.params["global_step"])

                if eva and (epoch+1) % self.params['eva_epoch'] == 0:
                    self.train(False)
                    logging.info(f"test epoch number {epoch+1}")
                    # results consist best_map, best_ap, best_conf,
                    # specific_conf_map, specific_conf_ap
                    map_results = get_map(self, test_loader, train=True,
                                          loop_conf=loop_conf)
                    self.params['best_map'] = map_results[0]
                    self.params['confidence'] = map_results[2]
                    for index, mr_name in enumerate(map_results_names):
                        try:
                            ts_writer["tensorboard_writer"].add_scalar(
                                    mr_name, map_results[index],
                                    self.params["global_step"])
                        except AttributeError:
                            continue

                    evaluate_running_loss = eval_score(self, test_loader)
                    for i, name in enumerate(self.losses_name):
                        ts_writer["tensorboard_writer"].add_scalar(
                                "evel_" + name, evaluate_running_loss[i],
                                self.params["global_step"])
                    self.train(True)
                    eva = 0
                if save and (epoch+1) % self.params['save_epoch'] == 0:
                    _save_checkpoint(self)
                    save = 0
            lr_scheduler.step()

        best_map, best_ap, best_conf, specific_conf_map, specific_conf_ap, \
            map_frame = get_map(self, test_loader, train=False, loop_conf=True)
        self.params['best_map'] = best_map
        self.params['confidence'] = best_conf
        _save_checkpoint(self)
        for index, mr_name in enumerate(map_results_names):
            try:
                ts_writer["tensorboard_writer"].add_scalar(
                        mr_name, map_results[index],
                        self.params["global_step"])
            except AttributeError:
                continue
        # model.train(True)
        logging.info("Bye~")
        if self.params['return_csv']:
            map_frame.to_csv(
                    f"{self.params['sub_working_dir']}/final_performance.csv",
                    index=True)
        return best_map, best_ap, best_conf, specific_conf_map,\
            specific_conf_ap, map_frame

    def predict(self, x, layer, batch_size):
        anchors = layer[0].anchors

        # x shape is in bs, num_anchors * (5 + c), in_w ,in_h (4d tensor)
        yolo_size = x.size(2)
        stride = self.params['height'] // yolo_size
        depth = 5 + self.params['num_classes']
        wh = yolo_size**2
        # transform x shape into bs, num_b * (5 + c), in_w * in_h (3d tensor)
        x = x.view(batch_size, depth*self.params['num_anchors'], wh)
        x = x.transpose(1, 2).contiguous().view(
                batch_size, wh*self.params['num_anchors'], depth)
        # x y centre point must be within 0 to 1, same to the object confidence
        x[:, :, 0] = torch.sigmoid(x[:, :, 0])  # centre x
        x[:, :, 1] = torch.sigmoid(x[:, :, 1])  # centre y
        x[:, :, 4] = torch.sigmoid(x[:, :, 4])  # object confidence

        # offset the centre coordinates according to their grid
# =============================================================================
#         offset = torch.FloatTensor(np.arange(yolo_size)).unsqueeze(1)
#         offset = offset.repeat(yolo_size, 2*self.num_anchors).view(1, -1, 2)
#         if cuda:
#             offset = offset.cuda()
#         x[:, :, :2] += offset
# =============================================================================
        grid = np.arange(yolo_size)
        a, b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(-1, 1)
        y_offset = torch.FloatTensor(b).view(-1, 1)

        if self.params['cuda']:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        x_y_offset = torch.cat((
                            x_offset, y_offset), 1
                            ).repeat(
                                    1, self.params['num_anchors']
                                    ).view(-1, 2).unsqueeze(0)

        x[:, :, :2] += x_y_offset

        anchors = [anchor/stride for anchor in anchors]

        anchors = torch.FloatTensor(anchors)   # in order to use repeat
        if self.params['cuda']:
            anchors = anchors.cuda()
        anchors = anchors.repeat(wh, 1).unsqueeze(0)

        x[:, :, 2:4] = torch.exp(x[:, :, 2:4])*anchors

        # sigmoid class confidence
        x[:, :, 5:] = torch.sigmoid(x[:, :, 5:])

        # standadize to the imput size
        x[:, :, :4] *= stride
        return x

    def yolo_loss(self, x, labels, layer, batch_size):

        # x shape is in bs, num_b * (5 + c), in_w ,in_h (4d tensor)
        yolo_size = x.size(2)
        stride = self.params['height'] // yolo_size
        depth = 5 + self.params['num_classes']
        scaled_anchors = [anchor/stride for anchor in layer[0].anchors]

        # transform x shape into bs, num_b * (5 + c), in_w * in_h (3d tensor)
        prediction = x.view(batch_size,
                            self.params['num_anchors'], depth,
                            yolo_size, yolo_size)
        # rearrange to: batch_size, num_anchors, yolo_size, yolo_size, depth
        # contiguous() is required after transposing the axis
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()

        # x y centre point must be within 0 to 1, same to the object confidence
        # centre x
        tx_pred = torch.sigmoid(prediction[:, :, :, :, 0])
        # centre y
        ty_pred = torch.sigmoid(prediction[:, :, :, :, 1])

        # w
        tw_pred = prediction[:, :, :, :, 2]

        # h
        th_pred = prediction[:, :, :, :, 3]

        # object confidence
        con_pred = torch.sigmoid(prediction[:, :, :, :, 4])
        # object confidence
        cls_pred = torch.sigmoid(prediction[:, :, :, :, 5:])

        # get label
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls, tw_pred, th_pred = \
            self.convert_label(labels, tw_pred, th_pred, scaled_anchors,
                               batch_size, yolo_size)

        if self.params['cuda']:
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
            self.mse_loss = self.mse_loss.cuda()
            self.bce_loss = self.bce_loss.cuda()

        # cordinates loss
        loss_x = self.mse_loss(tx_pred * mask, tx * mask)
        loss_y = self.mse_loss(ty_pred * mask, ty * mask)

        loss_w = self.mse_loss(((torch.abs(tw_pred) + 1e-16) ** (1/2)) * mask,
                               ((tw + 1e-16) ** (1/2)) * mask)
        loss_h = self.mse_loss(((torch.abs(th_pred) + 1e-16) ** (1/2)) * mask,
                               ((th + 1e-16) ** (1/2)) * mask)

#        loss_w = self.mse_loss(tw_pred  * mask, tw  * mask)
#        loss_h = self.mse_loss(th_pred * mask, th * mask)

        # confidence loss
        # noobj mask * 0.0 is to get noobj_mask size tensor filled with zeros
        loss_conf = self.bce_loss(con_pred * mask, mask) + \
            self.params['lambda_noobj'] * self.bce_loss(con_pred * noobj_mask,
                                                        noobj_mask * 0.0)

        # class loss
        loss_cls = self.bce_loss(cls_pred[mask == 1], tcls[mask == 1])

        # final loss
        loss = self.params['lambda_coord'] *\
            (loss_x + loss_y + loss_w + loss_h) + \
            self.params['conf_lambda'] * loss_conf +\
            self.params['cls_lambda'] * loss_cls

        return loss, loss_x.item(), loss_y.item(), loss_w.item(), \
            loss_h.item(), loss_conf.item(), loss_cls.item()

    def convert_label(self, labels, tw_pred, th_pred, scaled_anchors,
                      batch_size, yolo_size):

        # mask(i) is 1 when there is an object in cell i and 0 elsewhere
        # we set them all zero first and later change the one with highest conf
        # within the 3 bouding box (anchor box) to 1
        mask = torch.zeros(batch_size,
                           self.params['num_anchors'], yolo_size, yolo_size,
                           requires_grad=False)

        # noobj_mask(i) is 1 when there is no object in the cell i, else 0.
        noobj_mask = torch.ones(batch_size,
                                self.params['num_anchors'], yolo_size,
                                yolo_size, requires_grad=False)
        # initiate zero mask for 5 + c
        tx = torch.zeros(batch_size,
                         self.params['num_anchors'], yolo_size, yolo_size,
                         requires_grad=False)
        ty = torch.zeros(batch_size,
                         self.params['num_anchors'], yolo_size, yolo_size,
                         requires_grad=False)
        tw = torch.zeros(batch_size,
                         self.params['num_anchors'], yolo_size, yolo_size,
                         requires_grad=False)
        th = torch.zeros(batch_size,
                         self.params['num_anchors'], yolo_size, yolo_size,
                         requires_grad=False)
        tconf = torch.zeros(batch_size,
                            self.params['num_anchors'], yolo_size, yolo_size,
                            requires_grad=False)
        tcls = torch.zeros(batch_size,
                           self.params['num_anchors'], yolo_size, yolo_size,
                           self.params['num_classes'], requires_grad=False)

        for b in range(batch_size):
            for label in range(labels[b].shape[0]):
                if (labels[b][label] == 0).all():
                    continue
                gx = labels[b][label, 1] * yolo_size
                gy = labels[b][label, 2] * yolo_size
                gw = labels[b][label, 3] * yolo_size
                gh = labels[b][label, 4] * yolo_size

                # Get grid box indices
                gi = int(gx)
                gj = int(gy)

                # transform ground true box to torch tensor
                # returns a 1 * 4 tensor : xmin = 0 (origin), ymin = 0 , x
                # x_max, y_max
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh]
                                                    )).unsqueeze(0)

                # transform anchor box to torch tensor
                # returns a num_anchors * 4 tensor
                print(f"this is scaled_anchors {scaled_anchors}")
                anchor_box = torch.cat((torch.zeros(self.params['num_anchors'],
                                                    2),
                                        torch.FloatTensor(
                                                scaled_anchors)), 1)

                # iou between gt and anchor box, which is a 3 * 1 tensor
                ious = box_iou(gt_box, anchor_box)

                # According to the paper: YOLOv3 predicts an objectness score
                # for each bounding box using logistic regression. This should
                # be 1 if the bounding box prior overlaps a ground truth object
                # by more than any other bounding box prior. If the bounding
                # box prior is not the best but does overlap a ground truth
                # object by more than some threshold we ignore the prediction

                # Where the overlap is larger than threshold set noobj_mask to
                # zero, so that no loss will be calculated for the noobj part
                # in yolo v3 paper, threshold was set to 0.5
                noobj_mask[b, ious > self.params['ignore_threshold'],
                           gj, gi] = 0
                # Find the best matching anchor box\
                # np.argmax returns the indices of the maximum values
                best_n = np.argmax(ious)

                # only the highest conf bbox will be marked as 1
                mask[b, best_n, gj, gi] = 1

                # transform gx gy into the same format as our prediction
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj

                # Width and height
                tw[b, best_n, gj, gi] = gw
                th[b, best_n, gj, gi] = gh
                tw_pred[b, best_n, gj, gi] = torch.exp(
                        tw_pred[b, best_n, gj, gi]
                        ) * scaled_anchors[best_n][0]
                th_pred[b, best_n, gj, gi] = torch.exp(
                        th_pred[b, best_n, gj, gi]
                        ) * scaled_anchors[best_n][1]
#                print(tw_pred[b, best_n, gj, gi])
                # only the highest conf bbox will be marked as 1
                tconf[b, best_n, gj, gi] = 1

                # mark corresponding target to 1
                tcls[b, best_n, gj, gi, int(labels[b][label, 0])] = 1

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls, tw_pred, th_pred

    def forward(self, x, is_training=False, labels=None):
        cache = {}
        batch_size = x.size(0)
        for index, layer in enumerate(self.module_list):
            if index in self.layer_type_dic["conv"] or \
               index in self.layer_type_dic["upsampling"]:
                    x = self.module_list[index](x)

            elif index in self.layer_type_dic["route_1"]:
                referred_layer = self.layer_type_dic[
                                                    "referred_relationship"
                                                    ][index]
                x = cache[referred_layer]
            elif index in self.layer_type_dic["route_2"]:
                referred_layer = self.layer_type_dic[
                                                     "referred_relationship"
                                                     ][index]
                x = torch.cat((cache[referred_layer[0]],
                               cache[referred_layer[1]]), 1)                    

            elif index in self.layer_type_dic["shortcut"]:
                referred_layer = self.layer_type_dic[
                                                    "referred_relationship"
                                                    ][index]
                x += cache[referred_layer]

            elif index in self.layer_type_dic["yolo"]:
                if is_training:
                    ls = self.yolo_loss(x, labels, layer, batch_size)
                    if index == self.layer_type_dic["yolo"][0]:
                        losses = ls
                    elif index == self.layer_type_dic["yolo"][1]:
                        losses = [sum(x) for x in zip(losses, ls)]
                    else:
                        losses = [sum(x) for x in zip(losses, ls)]
                        return losses

                else:
                    x = self.predict(x, layer, batch_size)
                    if index == self.layer_type_dic["yolo"][0]:
                        detections = x
                    elif index == self.layer_type_dic["yolo"][1]:
                        detections = torch.cat((detections, x), 1)
                    else:
                        detections = torch.cat((detections, x), 1)
                        return detections
            if index in self.layer_type_dic["referred"]:
                cache[index] = x

    def load_weights(self, weightfile, cust_train_zero=False):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        if (cust_train_zero):
            before_yolo_layer = [x - 1 for x in self.layer_type_dic['yolo']]
        else:
            before_yolo_layer = []
        for i in range(len(self.module_list)):
            module_type = self.blocks[i]["type"]

            # If module_type is convolutional load weights
            # Otherwise ignore.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr +
                                                         num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr +
                                                          num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr +
                                                               num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr +
                                                              num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    if i in before_yolo_layer:
                        # Number of biases
                        num_biases = conv.bias.numel()

                        # Load the weights
                        conv_biases = torch.zeros(num_biases,
                                                  dtype=torch.float,
                                                  requires_grad=True)
                        ptr = ptr + 255

                        # reshape the loaded weights according to the dims of
                        # the model weights
                        conv_biases = conv_biases.view_as(conv.bias.data)

                        # Finally copy the data
                        conv.bias.data.copy_(conv_biases)

                    else:
                        num_biases = conv.bias.numel()

                        # Load the weights
                        conv_biases = torch.from_numpy(
                                weights[ptr: ptr + num_biases])
                        ptr = ptr + num_biases

                        # reshape the loaded weights according to the dims of
                        # the model weights
                        conv_biases = conv_biases.view_as(conv.bias.data)

                        # Finally copy the data
                        conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                if i in before_yolo_layer:
                    # Do the same as above for weights
                    conv_weights = torch.zeros(num_weights, dtype=torch.float,
                                               requires_grad=True)
                    ptr = ptr + 255

                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)

                else:
                    # Do the same as above for weights
                    conv_weights = torch.from_numpy(
                            weights[ptr:ptr+num_weights])
                    ptr = ptr + num_weights

                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)
