# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

class EmbodiedMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_score: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_score = cost_score
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, outputs, targets):
        self.reset()
        indices = []
        for bid in range(len(targets)):
            # load gt
            scores =  targets[bid]["scores"] # [num_gt]
            masks = targets[bid]["masks"] # [num_gt, N]
            query_gt_mask = targets[bid]["query_gt_mask"] # [num_queries, num_gt]
            # load pred
            pred_masks = outputs[bid]["pred_masks"] # [num_queries,N]
            pred_scores = outputs[bid]["pred_scores"].softmax(-1) # [num_queries, 2]
            # compute cost score
            cost_score = -pred_scores[:, scores] * self.cost_score # [num_queries, num_gt]
            # compute cost mask
            cost_mask = batch_sigmoid_ce_loss_jit(pred_masks, masks.float()) * self.cost_mask   # [num_queries, num_gt] 
            # compute cost dice
            cost_dice = batch_dice_loss_jit(pred_masks, masks.float()) * self.cost_dice # [num_queries, num_gt]
            # compute final cost
            C = cost_score + cost_mask + cost_dice # [num_queries, num_gt]
            C = torch.where(query_gt_mask.bool(), C, 1e8)
            if C.shape[0] == 1:
                topk_C = torch.topk(C, 1, dim=0, sorted=True, largest=False).values[-1:, :]
            else:
                topk_C = torch.topk(C, 2, dim=0, sorted=True, largest=False).values[-1:, :] # [1, num_gt]
            ids = torch.argwhere(C < topk_C)
            indices.append([ids[:, 0], ids[:, 1]])
        
        return indices
    def reset(self):
        self.cost_score = 1
        self.cost_mask = 1
        self.cost_dice = 1

class EmbodiedRecurrentMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_score: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_score = cost_score
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, outputs, targets):
        indices = []
        prev_indices = []
        new_indices = []
        for bid in range(len(targets)):
            # load gt
            scores =  targets[bid]["scores"] # [num_gt]
            masks = targets[bid]["masks"] # [num_gt, N]
            query_gt_mask = targets[bid]["query_gt_mask"] # [num_queries, num_gt]
            instance_ids = targets[bid]["instance_ids"] # [num_gt]
            # load pred
            pred_masks = outputs[bid]["pred_masks"] # [N, num_queries]
            pred_scores = outputs[bid]["pred_scores"].softmax(-1) # [num_queries, 2]
            query_instance_ids = outputs[bid]["query_instance_ids"] # [num_queries]
            # compute cost score
            cost_score = -pred_scores[:, scores] * self.cost_score # [num_queries, num_gt]
            # compute cost mask
            cost_mask = batch_sigmoid_ce_loss_jit(pred_masks.T, masks.float()) * self.cost_mask   # [num_queries, num_gt] 
            # compute cost dice
            cost_dice = batch_dice_loss_jit(pred_masks.T, masks.float()) * self.cost_dice # [num_queries, num_gt]
            # compute final cost
            C = cost_score + cost_mask + cost_dice # [num_queries, num_gt]
            C = torch.where(query_gt_mask.bool(), C, 1e8)
            # removeprevious matched query
            C[query_instance_ids != -1, :] = 1e8
            # match
            topk_C = torch.topk(C, 2, dim=0, sorted=True, largest=False).values[-1:, :] # [1, num_gt]
            ids = torch.argwhere(C < topk_C)
            # add previous matched query
            new_ids = ids.clone()
            prev_ids = []
            for i, prev_id in enumerate(query_instance_ids):
                if prev_id != -1 and prev_id in instance_ids:
                    prev_ids.append([i, (instance_ids == prev_id).nonzero(as_tuple=True)[0].item()])
            prev_ids = torch.tensor(prev_ids, device=ids.device, dtype=torch.long).reshape(-1, 2)
            ids = torch.cat([ids, prev_ids], dim=0)        
            indices.append([ids[:, 0], ids[:, 1]])
            prev_indices.append([prev_ids[:, 0], prev_ids[:, 1]])
            new_indices.append([new_ids[:, 0], new_ids[:, 1]])
        return indices, prev_indices, new_indices


class HungarianMatcherRecurrent(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
        ignore_label: int = -100
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.ignore_label = ignore_label

        assert (
            cost_class != 0 or cost_mask != 0 or cost_dice != 0
        ), "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, mask_type):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(
                -1
            )  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"].clone()
            if len(tgt_ids) == 0:
                indices.append(([], []))
                continue

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            filter_ignore = tgt_ids == self.ignore_label
            tgt_ids[filter_ignore] = 0
            cost_class = -out_prob[:, tgt_ids]
            cost_class[
                :, filter_ignore
            ] = (
                -1.0
            )  # for ignore classes pretend perfect match ;) TODO better worst class match?

            out_mask = outputs["pred_masks"][
                b
            ].T  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b][mask_type].to(out_mask)

            if self.num_points != -1:
                point_idx = torch.randperm(
                    tgt_mask.shape[1], device=tgt_mask.device
                )[: int(self.num_points * tgt_mask.shape[1])]
                # point_idx = torch.randint(0, tgt_mask.shape[1], size=(self.num_points,), device=tgt_mask.device)
            else:
                # sample all points
                point_idx = torch.arange(
                    tgt_mask.shape[1], device=tgt_mask.device
                )

            # out_mask = out_mask[:, None]
            # tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            # point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            # tgt_mask = point_sample(
            #     tgt_mask,
            #     point_coords.repeat(tgt_mask.shape[0], 1, 1),
            #     align_corners=False,
            # ).squeeze(1)

            # out_mask = point_sample(
            #     out_mask,
            #     point_coords.repeat(out_mask.shape[0], 1, 1),
            #     align_corners=False,
            # ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(
                    out_mask[:, point_idx], tgt_mask[:, point_idx]
                )

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(
                    out_mask[:, point_idx], tgt_mask[:, point_idx]
                )

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            # modify cost matrix by previous matched result
            prev_intances_ids = outputs['prev_instance_ids'][b]
            instance_ids = targets[b]['instance_ids']
            for i, prev_id in enumerate(prev_intances_ids):
                if prev_id != -1:
                    C[i, instance_ids == prev_id] = -1e6

            indices.append(linear_sum_assignment(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets, mask_type):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets, mask_type)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
