from ..detr.loss_detr import DetrLoss, _set_aux_loss


def GroundingDinoForObjectDetectionLoss(
    logits,
    labels,
    pred_boxes,
    config,
    device,
    outputs,
    outputs_class,
    outputs_coord,
):
    # First: create the criterion
    criterion = DetrLoss(config)
    criterion.to(device)
    # Second: compute the losses, based on outputs and labels
    outputs_loss = {}
    auxiliary_outputs = {}
    outputs_loss = {}
    outputs_loss["logits"] = logits
    outputs_loss["pred_boxes"] = pred_boxes
    if config.auxiliary_loss:
        auxiliary_outputs = _set_aux_loss(outputs_class, outputs_coord)
        outputs_loss["auxiliary_outputs"] = auxiliary_outputs
    if config.two_stage:
        enc_outputs_coord = outputs[-1].sigmoid()
        outputs_loss["enc_outputs"] = {"logits": outputs[-2], "pred_boxes": enc_outputs_coord}

    loss_dict = criterion(outputs_loss, labels)
    # Fourth: compute total loss, as a weighted sum of the various losses
    weight_dict = {"loss_ce": 1, "loss_bbox": config.bbox_loss_coefficient}
    weight_dict["loss_giou"] = config.giou_loss_coefficient
    if config.auxiliary_loss:
        aux_weight_dict = {}
        for i in range(config.decoder_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

    return loss, loss_dict, auxiliary_outputs
