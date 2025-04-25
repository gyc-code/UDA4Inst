# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import concurrent.futures
import logging
import numpy as np
import time
import datetime
import copy
import os
import random
import weakref
from typing import List, Mapping, Optional
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn.modules.dropout import _DropoutNd
from timm.models.layers import DropPath
import pathlib
import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import _log_api_usage
import cv2

from .uda_instance_utils import source_instance_paste_to_target_mix, target_instance_paste_to_source_mix, remove_ego_car_logo, break_source_target_match, visulize_color_instances

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer", "AMPTrainer"]
TOTAL_LOSS = True

VISUL = False
ITERATION_TO_START_UDA = 10000
MINI_BATCH_LOSS = True

class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()

    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    """

    trainer: "TrainerBase" = None
    """
    A weak reference to the trainer object. Set by the trainer when the hook is registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_backward(self):
        """
        Called after the backward pass of each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        """
        return {}


class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self) -> None:
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # Maintain the invariant that storage.iter == trainer.iter
        # for the entire execution of each step
        self.storage.iter = self.iter

        for h in self._hooks:
            h.before_step()

    def after_backward(self):
        for h in self._hooks:
            h.after_backward()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError

    def state_dict(self):
        ret = {"iteration": self.iter}
        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    # TODO handle repetitive stateful hooks
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)
        self.iter = state_dict["iteration"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        async_write_metrics=False,
    ):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
            gather_metric_period: an int. Every gather_metric_period iterations
                the metrics are gathered from all the ranks to rank 0 and logged.
            zero_grad_before_forward: whether to zero the gradients before the forward.
            async_write_metrics: bool. If True, then write metrics asynchronously to improve
                training speed
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        # to access the data loader iterator, call `self._data_loader_iter`
        self._data_loader_iter_obj = None
        self.optimizer = optimizer
        self.gather_metric_period = gather_metric_period
        self.zero_grad_before_forward = zero_grad_before_forward
        self.async_write_metrics = async_write_metrics
        # create a thread pool that can execute non critical logic in run_step asynchronically
        # use only 1 worker so tasks will be executred in order of submitting.
        self.concurrent_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        if not self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()
        losses.backward()

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self.data_loader)
        return self._data_loader_iter_obj

    def reset_data_loader(self, data_loader_builder):
        """
        Delete and replace the current data loader with a new one, which will be created
        by calling `data_loader_builder` (without argument).
        """
        del self.data_loader
        data_loader = data_loader_builder()
        self.data_loader = data_loader
        self._data_loader_iter_obj = None

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
        iter: Optional[int] = None,
    ) -> None:
        logger = logging.getLogger(__name__)

        iter = self.iter if iter is None else iter
        if (iter + 1) % self.gather_metric_period == 0:
            try:
                SimpleTrainer.write_metrics(loss_dict, data_time, iter, prefix)
            except Exception:
                logger.exception("Exception in writing metrics: ")
                raise

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        cur_iter: int,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        storage = get_event_storage()
        # Keep track of data time per rank
        storage.put_scalar("rank_data_time", data_time, cur_iter=cur_iter)

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time, cur_iter=cur_iter)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={cur_iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar(
                "{}total_loss".format(prefix), total_losses_reduced, cur_iter=cur_iter
            )
            if len(metrics_dict) > 1:
                storage.put_scalars(cur_iter=cur_iter, **metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def after_train(self):
        super().after_train()
        self.concurrent_executor.shutdown(wait=True)


class AMPTrainer(SimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        grad_scaler=None,
        precision: torch.dtype = torch.float16,
        log_grad_scaler: bool = False,
        async_write_metrics=False,
    ):
        """
        Args:
            model, data_loader, optimizer, gather_metric_period, zero_grad_before_forward,
                async_write_metrics: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
            precision: torch.dtype as the target precision to cast to in computations
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(
            model, data_loader, optimizer, gather_metric_period, zero_grad_before_forward
        )

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        self.precision = precision
        self.log_grad_scaler = log_grad_scaler
        ''' author add ema'''
        self.source_rare_class_samples = []
        self.local_iter = 0
        self.alpha = 0.999
        timestamp = time.time()
        human_readable_time = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d-%H:%M:%S')
        self.folder_name = './debug_in_img_' + human_readable_time+ '/'
        dir = pathlib.Path(self.folder_name)
        dir.mkdir(parents=True, exist_ok=True)

    def _init_ema_weights(self):
        self.ema_model = copy.deepcopy(self.model).eval() # init , no weight load

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(alpha_teacher).add_(param.data, alpha=1 - alpha_teacher)
                    
    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        self.local_iter += 1

        if 'source' in data[0] and self.local_iter > ITERATION_TO_START_UDA:# author add 
            batch_size = len(data)
            if self.local_iter == ITERATION_TO_START_UDA + 1:
                self._init_ema_weights()
            self._update_ema(self.local_iter)

            # source training
            if not MINI_BATCH_LOSS:
                self.optimizer.zero_grad()
                with autocast(dtype=self.precision):
                    source_loss_dict = self.model(data)
                    if isinstance(source_loss_dict, torch.Tensor):
                        source_losses = source_loss_dict
                        source_loss_dict = {"total_source_loss": source_loss_dict}
                    else:
                        source_losses = sum(source_loss_dict.values())
            
            ''' author : generate pseudo label for target and do mix '''
            with torch.no_grad():
                ''' author: forward with taget data in EMA '''
                self.ema_model.training = False
                # Generate pseudo-label
                pseudo_labels = self.ema_model(data, target=True) 
                pseudo_instances_num_list = []
                for i in range(len(pseudo_labels)): # filter pseudo instances which score are low
                    template_img = data[i]['target']['template_img']
                    pseudo_instances = pseudo_labels[i]['instances']
                    pseudo_labels[i]['instances'] = pseudo_instances[pseudo_instances.scores.cpu() > 0.9]
                    update_pseudo_label = remove_ego_car_logo(pseudo_labels[i]['instances'], template_img)
                    if update_pseudo_label is None:
                        pseudo_instances_num_list.append(0)
                        continue
                    else:
                        pseudo_labels[i]['instances'] = update_pseudo_label
                        pseudo_instances_num_list.append(len(pseudo_labels[i]['instances']._fields))
                    data[i]['target']['instances'] = pseudo_labels[i]['instances']
                del pseudo_labels
            # pseudo instance use to mix
            any_greater_than_zero = any(x > 0 for x in pseudo_instances_num_list)
            if any_greater_than_zero:
                data_ori = copy.deepcopy(data[0])
                data_copy = copy.deepcopy(data)
                if VISUL:
                    self.model.training = False
                    data[0]['source']['height'] = 1024
                    data[0]['source']['width'] = 1024

                    source = self.model(data)
                    source_instances = source[0]['instances']
                    source_instances = source_instances[source_instances.scores.cpu() > 0.8]

                    source_instances_img = visulize_color_instances(source_instances)
                    file_id = data[0]['target']['image_id'].split('.')[0]
                    cv2.imwrite(self.folder_name + file_id + '_' + str(self.local_iter)+ '_source_inference_color_instance.jpg', source_instances_img)
                    del source, source_instances, source_instances_img
                   
                for i in range(batch_size):
                    if pseudo_instances_num_list[i] > 0:
                        data[i], self.source_rare_class_samples = source_instance_paste_to_target_mix(data[i], self.local_iter, self.folder_name, self.source_rare_class_samples)
                        data_copy[i] = target_instance_paste_to_source_mix(data_copy[i], self.local_iter, self.folder_name)

                ''' author: train with source2target mix data '''
                if VISUL:
                    self.model.training = False
                    data[0]['source']['height'] = 1024
                    data[0]['source']['width'] = 1024
                    data_copy[0]['source']['height'] = 1024
                    data_copy[0]['source']['width'] = 1024
                    source_to_target_mix = self.model(data)
                    target_to_source_mix = self.model(data_copy)
                    source_to_target_mix_instances = source_to_target_mix[0]['instances']
                    target_to_source_mix_instances = target_to_source_mix[0]['instances']

                    source_to_target_mix_instances = source_to_target_mix_instances[source_to_target_mix_instances.scores.cpu() > 0.8]
                    target_to_source_mix_instances = target_to_source_mix_instances[target_to_source_mix_instances.scores.cpu() > 0.8]
                    
                    source_to_target_mix_instances_img = visulize_color_instances(source_to_target_mix_instances)
                    target_to_source_mix_instances_img = visulize_color_instances(target_to_source_mix_instances)
                    file_id = data[0]['target']['image_id'].split('.')[0]
                    cv2.imwrite(self.folder_name + file_id + '_' + str(self.local_iter)+ '_s2t_inference_color_instance.jpg', source_to_target_mix_instances_img)
                    cv2.imwrite(self.folder_name + file_id + '_' + str(self.local_iter)+ '_t2s_inference_color_instance.jpg', target_to_source_mix_instances_img)
                    del source_to_target_mix, source_to_target_mix_instances
                    del target_to_source_mix, target_to_source_mix_instances

                if not MINI_BATCH_LOSS:
                    self.model.training = True
                    mix_loss_dict = self.model(data)
                    if isinstance(mix_loss_dict, torch.Tensor):
                        s2t_mix_losses = mix_loss_dict
                        loss_dict = {"total_mix_loss": mix_loss_dict}
                    else:
                        s2t_mix_losses = sum(mix_loss_dict.values())

                    ''' author: train with target2source mix data '''
                    t2s_mix_loss_dict = self.model(data_copy)
                    if isinstance(t2s_mix_loss_dict, torch.Tensor):
                        t2s_mix_losses = t2s_mix_loss_dict
                        loss_dict = {"total_mix_loss": t2s_mix_loss_dict}
                    else:
                        t2s_mix_losses = sum(t2s_mix_loss_dict.values())
                    unite_loss = 0.5 * source_losses + 0.25 * t2s_mix_losses + 0.25 * s2t_mix_losses
                    unite_loss_dict = t2s_mix_loss_dict
                else:
                    ''' use mini batch loss ,one batch data=source+s2t+t2s ''' 
                    self.optimizer.zero_grad()
                    self.model.training = True
                    assert batch_size % 3 == 0, f"Batch size must be a multiple of 3, but got {batch_size}"
                    if batch_size == 3:
                        data[0] = data_ori
                        data[2] = data_copy[2]
                    with autocast(dtype=self.precision):
                        unite_loss_dict = self.model(data)
                        if isinstance(unite_loss_dict, torch.Tensor):
                            unite_loss = unite_loss_dict
                            unite_loss_dict = {"total_source_loss": unite_loss_dict}
                        else:
                            unite_loss = sum(unite_loss_dict.values())

                self.grad_scaler.scale(unite_loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                if self.async_write_metrics:
                    # write metrics asynchronically
                    self.concurrent_executor.submit(self._write_metrics, unite_loss_dict, data_time, iter=self.iter)
                else:
                    self._write_metrics(unite_loss_dict, data_time)

            if self.log_grad_scaler:
                storage = get_event_storage()
                storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())
            self.after_backward()


        else:
            if 'source' in data[0]: # when local_iter < ITERATION_TO_START_UDA, also use only source training
                data = [x['source'] for x in data]

            if self.zero_grad_before_forward:
                self.optimizer.zero_grad()
            with autocast(dtype=self.precision):
                loss_dict = self.model(data)
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())

            if not self.zero_grad_before_forward:
                self.optimizer.zero_grad()

            self.grad_scaler.scale(losses).backward()

            if self.log_grad_scaler:
                storage = get_event_storage()
                storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

            self.after_backward()

            if self.async_write_metrics:
                # write metrics asynchronically
                self.concurrent_executor.submit(
                    self._write_metrics, loss_dict, data_time, iter=self.iter
                )
            else:
                self._write_metrics(loss_dict, data_time)

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
