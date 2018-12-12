from __future__ import division

import torch
import torch.nn as nn
from utilis import create_module, box_iou
import numpy as np


class yolo_v3(nn.Module):
    def __init__(self, blocks, lambda_coord=3,
                 lambda_noobj=0.5, ignore_threshold=0.7, conf_lambda=3,
                 cls_lambda=2):
        super().__init__()
        self.blocks = blocks[1:]
        self.net = blocks[0]
        self.layer_type_dic, self.module_list = create_module(blocks)
        self.num_anchors = self.layer_type_dic["net_info"]["num_anchors"]
        self.num_classes = self.layer_type_dic["net_info"]["num_classes"]

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.ignore_threshold = ignore_threshold
        self.conf_lambda = conf_lambda
        self.cls_lambda = cls_lambda

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]

    def predict(self, x, layer, batch_size, input_size, cuda):
        anchors = layer[0].anchors
        
        # x shape is in bs, num_anchors * (5 + c), in_w ,in_h (4d tensor)
        yolo_size = x.size(2)
        stride = input_size // yolo_size
        depth = 5 + self.num_classes
        wh = yolo_size**2
        # transform x shape into bs, num_b * (5 + c), in_w * in_h (3d tensor)
        x = x.view(batch_size, depth*self.num_anchors, wh)
        x = x.transpose(1, 2).contiguous().view(
                                              batch_size,
                                              wh*self.num_anchors,
                                              depth
                                              )
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

        if cuda:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()

        x_y_offset = torch.cat((
                            x_offset, y_offset), 1
                    ).repeat(1, self.num_anchors).view(-1, 2).unsqueeze(0)

        x[:, :, :2] += x_y_offset


        anchors = [anchor/stride for anchor in anchors]

        anchors = torch.FloatTensor(anchors)   # in order to use repeat
        if cuda:
            anchors = anchors.cuda()
        anchors = anchors.repeat(wh, 1).unsqueeze(0)
        
        x[:, :, 2:4] = torch.exp(x[:, :, 2:4])*anchors


        # sigmoid class confidence
        x[:, :, 5:] = torch.sigmoid(x[:, :, 5:])

        # standadize to the imput size
        x[:, :, :4] *= stride
        return x

    def yolo_loss(self, x, labels, layer, batch_size, input_size, cuda = True):

        # x shape is in bs, num_b * (5 + c), in_w ,in_h (4d tensor)
        yolo_size = x.size(2)
        stride = input_size // yolo_size
        depth = 5 + self.num_classes
        scaled_anchors = [anchor/stride for anchor in layer[0].anchors]

        # transform x shape into bs, num_b * (5 + c), in_w * in_h (3d tensor)
        prediction = x.view(batch_size, self.num_anchors, depth,
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
#        print(tw_pred)

        # object confidence
        con_pred = torch.sigmoid(prediction[:, :, :, :, 4])
        # object confidence
        cls_pred = torch.sigmoid(prediction[:, :, :, :, 5:])

        # get label
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls, tw_pred, th_pred = \
            self.convert_label(labels, batch_size, tw_pred, th_pred,
                               scaled_anchors, yolo_size,
                               self.ignore_threshold)

        if cuda:
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
            self.mse_loss = self.mse_loss.cuda()
            self.bce_loss = self.bce_loss.cuda()

        # cordinates loss
        loss_x = self.mse_loss(tx_pred * mask, tx * mask)
        loss_y = self.mse_loss(ty_pred * mask, ty * mask)

        loss_w = self.mse_loss(((torch.abs(tw_pred) + 1e-16) ** (1/2)) * mask, ((tw + 1e-16) ** (1/2)) * mask)
        loss_h = self.mse_loss(((torch.abs(th_pred) + 1e-16) ** (1/2)) * mask, ((th + 1e-16) ** (1/2)) * mask)

#        loss_w = self.mse_loss(tw_pred  * mask, tw  * mask)
#        loss_h = self.mse_loss(th_pred * mask, th * mask)

        # confidence loss
        # noobj mask * 0.0 is to get noobj_mask size tensor filled with zeros
        loss_conf = self.bce_loss(con_pred * mask, mask) + \
            self.lambda_noobj * self.bce_loss(con_pred * noobj_mask,
                                              noobj_mask * 0.0)

        # class loss
        loss_cls = self.bce_loss(cls_pred[mask == 1], tcls[mask == 1])

        # final loss
        loss = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h) + \
            self.conf_lambda * loss_conf + self.cls_lambda * loss_cls

        return loss, loss_x.item(), loss_y.item(), loss_w.item(), \
            loss_h.item(), loss_conf.item(), loss_cls.item()

    def convert_label(self, labels, batch_size, tw_pred, th_pred,
                      anchors, yolo_size, ignore_threshold):

        # mask(i) is 1 when there is an object in cell i and 0 elsewhere
        # we set them all zero first and later change the one with highest conf
        # within the 3 bouding box (anchor box) to 1
        mask = torch.zeros(batch_size, self.num_anchors, yolo_size, yolo_size,
                           requires_grad=False)

        # noobj_mask(i) is 1 when there is no object in the cell i, else 0.
        noobj_mask = torch.ones(batch_size, self.num_anchors, yolo_size,
                                yolo_size, requires_grad=False)
        # initiate zero mask for 5 + c
        tx = torch.zeros(batch_size, self.num_anchors, yolo_size, yolo_size,
                         requires_grad=False)
        ty = torch.zeros(batch_size, self.num_anchors, yolo_size, yolo_size,
                         requires_grad=False)
        tw = torch.zeros(batch_size, self.num_anchors, yolo_size, yolo_size,
                         requires_grad=False)
        th = torch.zeros(batch_size, self.num_anchors, yolo_size, yolo_size,
                         requires_grad=False)
        tconf = torch.zeros(batch_size, self.num_anchors, yolo_size, yolo_size,
                            requires_grad=False)
        tcls = torch.zeros(batch_size, self.num_anchors, yolo_size, yolo_size,
                           self.num_classes, requires_grad=False)

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
                # returns a 3 * 4 tensor
                anchor_box = torch.cat((torch.zeros(self.num_anchors, 2),
                                        torch.FloatTensor(anchors)), 1)

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
                noobj_mask[b, ious > ignore_threshold, gj, gi] = 0
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
                        tw_pred[b, best_n, gj, gi]) * anchors[best_n][0]
                th_pred[b, best_n, gj, gi] = torch.exp(
                        th_pred[b, best_n, gj, gi]) * anchors[best_n][1]
#                print(tw_pred[b, best_n, gj, gi])
                # only the highest conf bbox will be marked as 1
                tconf[b, best_n, gj, gi] = 1

                # mark corresponding target to 1
                tcls[b, best_n, gj, gi, int(labels[b][label, 0])] = 1

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls, tw_pred, th_pred

    def forward(self, x, cuda, is_training=False, labels=None):
        cache = {}
        input_size = self.net["height"]
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
                    ls = self.yolo_loss(x, labels, layer, batch_size,
                                        input_size, cuda)
                    if index == self.layer_type_dic["yolo"][0]:
                        losses = ls
                    elif index == self.layer_type_dic["yolo"][1]:
                        losses = [sum(x) for x in zip(losses, ls)]
                    else:
                        losses = [sum(x) for x in zip(losses, ls)]
                        return losses

                else:
                    x = self.predict(x, layer, batch_size, input_size, cuda)
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
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]

              
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    if i in before_yolo_layer:
                        #Number of biases
                        num_biases = conv.bias.numel()
                    
                        #Load the weights
                        conv_biases = torch.zeros(num_biases, dtype=torch.float,
                                                 requires_grad=True)
                        ptr = ptr + 255
                    
                        #reshape the loaded weights according to the dims of the model weights
                        conv_biases = conv_biases.view_as(conv.bias.data)
                    
                        #Finally copy the data
                        conv.bias.data.copy_(conv_biases)
                
                    
                    else:
                        num_biases = conv.bias.numel()
                    
                        #Load the weights
                        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                        ptr = ptr + num_biases
                    
                        #reshape the loaded weights according to the dims of the model weights
                        conv_biases = conv_biases.view_as(conv.bias.data)

                        #Finally copy the data
                        conv.bias.data.copy_(conv_biases)

                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                if i in before_yolo_layer:
                    #Do the same as above for weights
                    conv_weights = torch.zeros(num_weights, dtype=torch.float,
                                              requires_grad=True)
                    ptr = ptr + 255

                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)

                else:
                    #Do the same as above for weights
                    conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                    ptr = ptr + num_weights
                    
                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)

