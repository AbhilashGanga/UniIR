import torch
from torch.cuda.amp import autocast

from models.uniir_align import utils
import transformers


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Train Epoch: [{}]".format(epoch)
    print_freq = config.trainer_config.print_freq

    accumulation_steps = config.trainer_config.gradient_accumulation_steps
    accumulation_counter = 0
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        #print(type(batch))
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(gpu_id, non_blocking=True)  # Batch is a dictionary of tensors
            elif isinstance(batch[key], transformers.tokenization_utils_base.BatchEncoding):
                for k in batch[key]:
                    batch[key][k] = batch[key][k].to(gpu_id, non_blocking=True)
            elif isinstance(batch[key], list):
                image_processor = model_without_ddp.get_img_preprocess_fn()
                batch[key] = image_processor(images = batch[key], return_tensors="pt")
                for k in batch[key]:
                    batch[key][k] = batch[key][k].to(gpu_id, non_blocking=True)

        # autocast for mixed precision
        #with autocast():
        outputs = model(batch)
        loss = outputs["loss"]
        inbatch_accuracy = outputs["accuracy"]

        # Scale the loss by the number of accumulation steps since backward averages the gradients.
        loss = loss / accumulation_steps

        # Use scaler for backward
        scaler.scale(loss).backward()

        accumulation_counter += 1
        if accumulation_counter == accumulation_steps:
            global_step += 1

            # optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()

            model.zero_grad()
            scheduler.step()
            accumulation_counter = 0

        metric_logger.update(loss=loss.item() * accumulation_steps)  # We scale back the loss for logging.
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  # TODO: might need to loop through all param groups
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_engine(model, model_without_ddp, data_loader, gpu_id, config):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("inbatch_accuracy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Test:"
    print_freq = config.evaluator.print_freq

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(gpu_id, non_blocking=True)  # Batch is a dictionary of tensors
            elif isinstance(batch[key], transformers.tokenization_utils_base.BatchEncoding):
                for k in batch[key]:
                    batch[key][k] = batch[key][k].to(gpu_id, non_blocking=True)
            elif isinstance(batch[key], list):
                image_processor = model_without_ddp.get_img_preprocess_fn()
                batch[key] = image_processor(images = batch[key], return_tensors="pt")
                for k in batch[key]:
                    batch[key][k] = batch[key][k].to(gpu_id, non_blocking=True)

        # autocast for mixed precision
        #with autocast():
        outputs = model(batch)
        loss = outputs["loss"]
        inbatch_accuracy = outputs["accuracy"]

        metric_logger.update(loss=loss.item())
        metric_logger.update(inbatch_accuracy=inbatch_accuracy.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
