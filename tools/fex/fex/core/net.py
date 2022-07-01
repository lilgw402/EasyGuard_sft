"""
模型的定义。
基本参考 https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/core/lightning.py """

from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union
import collections
import inspect

import torch
from torch.nn import Module

from fex.utils import MetricType
from fex.utils.sync_tensor import sync_tensor
from fex.utils.distributed import rank_zero_warn, rank_zero_info, log


class Net(
    ABC,
    # DeviceDtypeModuleMixin,
    # GradInformation,
    # ModelIO,
    # ModelHooks,
    # DataHooks,
    # CheckpointHooks,
    Module,
):
    """一个模型的基类"""
    # Below is for property support of JIT in PyTorch 1.7
    # since none of them is important when using JIT, we are going to ignore them.
    # __jit_unused_properties__ = [
    #     "datamodule",
    #     "example_input_array",
    #     "hparams",
    #     "hparams_initial",
    #     "on_gpu",
    #     "current_epoch",
    #     "global_step",
    # ] + DeviceDtypeModuleMixin.__jit_unused_properties__

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: 这个是什么作用
        # see (https://github.com/pytorch/pytorch/blob/3e6bb5233f9ca2c5aa55d9cda22a7ee85439aa6e/
        # torch/nn/modules/module.py#L227)
        # torch._C._log_api_usage_once(f"fex.module.{self.__class__.__name__}")
        config.MOUDEL_FILE = inspect.getfile(self.__class__)
        self.exp_save_path = None

        self.loaded_optimizer_states_dict = {}
        self.config = config

        # TODO: 和trainer有一些交织，不是特别喜欢这个设计，但Net里可能有一些地方会用到step等的信息
        #: Pointer to the trainer object
        self.trainer = None

        #: Pointer to the logger object
        self.logger = log

        #: True if using amp
        self.use_amp = False

        #: The precision used
        self.precision = 32

        # optionally can be set by user
        self._example_input_array = None
        self._datamodule = None
        # self._results: Optional[Result] = None
        self._results = None
        # self._current_fx_name = ''
        # self._current_hook_fx_name = None
        # self._current_dataloader_idx = None
        self._total_step = 0
        self._step_per_epoch = 0

        self._freeze_bn_running_stats = self.config.NETWORK.FREEZE_BN_RUNNING_STATS
        self._log_frequent = self.config.TRAINER.LOG_FREQUENT

    @property
    def log_frequent(self) -> int:
        """ 获取config中的log_frequent"""
        return self._log_frequent

    @property
    def example_input_array(self) -> Any:
        """ 用来展示示例输入数据 """
        return self._example_input_array

    @property
    def current_epoch(self) -> int:
        """The current epoch"""
        return self.trainer.current_epoch if self.trainer else 0

    @property
    def global_step(self) -> int:
        """Total training batches seen across all epochs"""
        return self.trainer.global_step if self.trainer else 0

    @property
    def current_step(self) -> int:
        """ The current step """
        return self.trainer.current_step if self.trainer else 0

    @property
    def total_step(self) -> int:
        return self._total_step

    @property
    def step_per_epoch(self) -> int:
        return self._step_per_epoch

    @property
    def model_device(self) -> int:
        return self.trainer.local_rank if self.trainer else 0

    def set_train_base(self, train_base: Any) -> None:
        """
        set train_base to trainer
        """
        self.trainer = train_base
        self._total_step = train_base.total_step
        self._step_per_epoch = train_base.step_per_epoch

    # @example_input_array.setter  # type: ignore
    # def example_input_array(self, example: Any) -> None:
    #     self._example_input_array = example

    @property
    def datamodule(self) -> Any:
        """ 用来展示datamodule """
        return self._datamodule

    @datamodule.setter
    def datamodule(self, datamodule: Any) -> None:
        self._datamodule = datamodule

    # @property
    # def on_gpu(self):
    #     """
    #     True if your model is currently running on GPUs.
    #     Useful to set flags around the Net for different CPU vs GPU behavior.
    #     """
    #     return self.cuda()

    def print(self, *args, **kwargs) -> None:
        r"""
        Prints only from process 0. Use this in any distributed mode to log only once.
        Args:
            *args: The thing to print. Will be passed to Python's built-in print function.
            **kwargs: Will be passed to Python's built-in print function.
        Example:
            .. code-block:: python
                def forward(self, x):
                    self.print(x, 'in forward')
        """
        if not self.logger:
            rank_zero_info(*args, **kwargs)

    @torch.no_grad()
    def log(
        self,
        name: str,
        value: Any,
        write_log: bool = False,
        metric_type: MetricType = MetricType.SCALAR,
        reduce_op: str = "avg",
        frequence: int = None,
        sycn_between_rank: bool = True
    ) -> None:
        """
        该方法用于记录训练中的metrics, forward的total_loss会在trainer中记录, 无需额外添加 .
        Log a key, value
        Example::
            self.log('train_acc', acc)
        Args:
            name(str): metrics的key .
            value(Any): metrics key对应的value .
            write_log(bool): 如果为True, 写入到logger中, default: False .
            metric_type(MetricType): metirc的类型, 支持scalar和histogram两种类型, default: MetricType.SCALAR.
            reduce_op(str): reduction function 的类型 , default: avg .
            frequence(int): metrics打点的频率, default: config.LOG_FREQUENT .
        """
        frequence = frequence if frequence else self._log_frequent
        if self.global_step % frequence == 0:
            metric_tag = "Train" if self.training else "Val"
            if self.trainer is None or self.trainer.writer is None:
                # log.warning("Writer is None in TrainBase, skip write metrics!")
                return
            if isinstance(value, torch.Tensor) and sycn_between_rank and metric_type != MetricType.IMAGE:  # 如果是image，sync 会重合
                value = sync_tensor(input_value=value, reduce_op=reduce_op)
            if metric_type == MetricType.SCALAR:
                self.trainer.add_scalar(
                    tag='%s/%s' % (metric_tag, name), scalar_value=value, global_step=self.global_step)
            elif metric_type == MetricType.HISTOGRAM:
                self.trainer.add_histogram(
                    tag='%s/%s' % (metric_tag, name), values=value, global_step=self.global_step)
            elif metric_type == MetricType.IMAGE:
                self.trainer.writer.add_image(
                    tag='%s/%s' % (metric_tag, name), img_tensor=value, global_step=self.global_step)
            elif metric_type == MetricType.IMAGES:
                self.trainer.writer.add_images(
                    tag='%s/%s' % (metric_tag, name), img_tensor=value, global_step=self.global_step)
            else:
                raise ValueError(
                    "Not support metric_type: {}".format(metric_type))

    def log_dict(
        self,
        dictionary: dict,
        write_log: bool = False,
        metric_type: MetricType = MetricType.SCALAR,
        reduce_op: str = "avg"
    ) -> None:
        """
        Log a dictonary of values at once, all values should be one metric_type.
        Example::
            values = {'acc': acc, ..., 'metric_n': metric_n}
            self.log_dict(values)
        Args:
            dictionary: key value pairs (str, tensors)
            write_log: if True logs to the logger.
            metric_type: support scalar and histogram type.
            reduce_op: reduction function over step values for end of epoch.
        """
        for k, v in dictionary.items():
            self.log(
                name=k,
                value=v,
                metric_type=metric_type,
                write_log=write_log,
                reduce_op=reduce_op)

    def __auto_choose_log_on_step(self, on_step):
        if on_step is None:
            if self._current_fx_name in {'training_step', 'training_step_end'}:
                on_step = True
            elif self._current_fx_name in {'evaluation_step', 'evaluation_step_end',
                                           'evaluation_epoch_end', 'training_epoch_end'}:
                on_step = False
            else:
                on_step = False

        return on_step

    def __auto_choose_log_on_epoch(self, on_epoch):
        if on_epoch is None:
            if self._current_fx_name in {'training_step', 'training_step_end'}:
                on_epoch = False
            elif self._current_fx_name in {'evaluation_step', 'evaluation_step_end',
                                           'evaluation_epoch_end', 'training_epoch_end'}:
                on_epoch = True
            else:
                on_epoch = True

        return on_epoch

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        """ batch norm 的 running mean 和 running var只有在eval()下才不更新，但是每次net.train()都会让模型重置到train状态，
            所以如果希望bn在训练的时候running mean 和 running var 不更新（比如finetune的时候），需要在train() 之后，让bn再次eval()
        """
        if self._freeze_bn_running_stats:
            self.bn_eval()

    def bn_eval(self):
        def set_bn_eval(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        self.apply(set_bn_eval)

    def forward(self, *args, **kwargs):
        r"""
        forward的整体思想是，只做最基本的encode的计算，一些业务相关的计算：
        比如triplet、包括head和loss等，可以放到`training_step` 里做。
        这样的好处是，可以分开定义 training 和 predicting 的操作，但又把重复的计算逻辑写到`forward`里，避免重复和bug。
        Same as :meth:`torch.nn.Module.forward()`, however in Lightning you want this to define
        the operations you want to use for prediction (i.e.: on a server or as a feature extractor).
        Normally you'd call ``self()`` from your :meth:`training_step` method.
        This makes it easy to write a complex system for training with the outputs
        you'd want in a prediction setting.
        Args:
            *args: Whatever you decide to pass into the forward method.
            **kwargs: Keyword arguments are also possible.
        Return:
            Predicted output
        Examples:
            .. code-block:: python
                # example if we were using this model as a feature extractor
                def forward(self, x):
                    feature_maps = self.convnet(x)
                    return feature_maps
                def training_step(self, batch, batch_idx):
                    x, y = batch
                    feature_maps = self(x)
                    logits = self.classifier(feature_maps)
                    # ...
                    return loss
                # splitting it this way allows model to be used a feature extractor
                model = MyModelAbove()
                inputs = server.get_request()
                results = model(inputs)
                server.write_results(results)
                # -------------
                # This is in stark contrast to torch.nn.Module where normally you would have this:
                def forward(self, batch):
                    x, y = batch
                    feature_maps = self.convnet(x)
                    logits = self.classifier(feature_maps)
                    return logits
        """
        return super().forward(*args, **kwargs)

    def training_step(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        r"""
        Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        Args:
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (int): Integer displaying index of this batch
            optimizer_idx (int): When using multiple optimizers, this argument will also be present.
            hiddens(:class:`~torch.Tensor`): Passed in if
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.truncated_bptt_steps` > 0.
        Return:
            Any of.
            - :class:`~torch.Tensor` - The loss tensor
            - `dict` - A dictionary. Can include any keys, but must include the key 'loss'
            - `None` - Training will skip to the next batch
        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.
        Example::
            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss(out, x)
                return loss
        If you define multiple optimizers, this step will be called with an additional
        ``optimizer_idx`` parameter.
        .. code-block:: python
            # Multiple optimizers (e.g.: GANs)
            def training_step(self, batch, batch_idx, optimizer_idx):
                if optimizer_idx == 0:
                    # do training_step with encoder
                if optimizer_idx == 1:
                    # do training_step with decoder
        If you add truncated back propagation through time you will also get an additional
        argument with the hidden states of the previous step.
        .. code-block:: python
            # Truncated back-propagation through time
            def training_step(self, batch, batch_idx, hiddens):
                # hiddens are the hidden states from the previous truncated backprop step
                ...
                out, hiddens = self.lstm(data, hiddens)
                ...
                return {'loss': loss, 'hiddens': hiddens}
        Note:
            The loss value shown in the progress bar is smoothed (averaged) over the last values,
            so it differs from the actual loss returned in train/validation step.
        """
        rank_zero_warn(
            "`training_step` must be implemented to be used with the Trainer"
        )

    def training_step_end(self, *args, **kwargs):
        """
        这个函数是在做完一个training_step之后做的，用处是做一些整batch级别的操作。
        因为如果是数据分布式的话，training_step只是定义了每个node单独的操作，
        如果还想在每个node单独计算完之后，再来个full-batch的操作（比如softmax NCE），
        可以在这里定义。
        Use this when training with dp because :meth:`training_step`
        will operate on only part of the batch. However, this is still optional
        and only needed for things like softmax or NCE loss.
        Note:
            If you later switch to ddp or some other mode, this will still be called
            so that you don't have to change your code
        .. code-block:: python
            # pseudocode
            sub_batches = split_batches_for_dp(batch)
            batch_parts_outputs = [training_step(sub_batch) for sub_batch in sub_batches]
            training_step_end(batch_parts_outputs)
        Args:
            batch_parts_outputs: What you return in `training_step` for each batch part.
        Return:
            Anything
        When using dp/ddp2 distributed backends, only a portion of the batch is inside the training_step:
        .. code-block:: python
            def training_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch
                out = self(x)
                # softmax uses only a portion of the batch in the denomintaor
                loss = self.softmax(out)
                loss = nce_loss(loss)
                return loss
        If you wish to do something with all the parts of the batch, then use this method to do it:
        .. code-block:: python
            def training_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch
                out = self.encoder(x)
                return {'pred': out}
            def training_step_end(self, training_step_outputs):
                gpu_0_pred = training_step_outputs[0]['pred']
                gpu_1_pred = training_step_outputs[1]['pred']
                gpu_n_pred = training_step_outputs[n]['pred']
                # this softmax now uses the full batch
                loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
                return loss
        See Also:
            See the :ref:`multi_gpu` guide for more details.
        """

    def training_epoch_end(self, outputs: List[Any]) -> None:
        """
        Called at the end of the training epoch with the outputs of all training steps.
        Use this in case you need to do something with all the outputs for every training_step.
        .. code-block:: python
            # the pseudocode for these calls
            train_outs = []
            for train_batch in train_data:
                out = training_step(train_batch)
                train_outs.append(out)
            training_epoch_end(train_outs)
        Args:
            outputs: List of outputs you defined in :meth:`training_step`, or if there are
                multiple dataloaders, a list containing a list of outputs for each dataloader.
        Return:
            None
        Note:
            If this method is not overridden, this won't be called.
        Example::
            def training_epoch_end(self, training_step_outputs):
                # do something with all training_step outputs
                return result
        With multiple dataloaders, ``outputs`` will be a list of lists. The outer list contains
        one entry per dataloader, while the inner list contains the individual outputs of
        each training step for that dataloader.
        .. code-block:: python
            def training_epoch_end(self, training_step_outputs):
                for out in training_step_outputs:
                    # do something here
        """

    def validation_step(self, *args, **kwargs):
        r"""
        Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.
        .. code-block:: python
            # the pseudocode for these calls
            val_outs = []
            for val_batch in val_data:
                out = validation_step(train_batch)
                val_outs.append(out)
                validation_epoch_end(val_outs)
        Args:
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (int): The index of this batch
            dataloader_idx (int): The index of the dataloader that produced this batch
                (only if multiple val datasets used)
        Return:
           Any of.
            - Any object or value
            - `None` - Validation will skip to the next batch
        .. code-block:: python
            # pseudocode of order
            out = validation_step()
            if defined('validation_step_end'):
                out = validation_step_end(out)
            out = validation_epoch_end(out)
        .. code-block:: python
            # if you have one val dataloader:
            def validation_step(self, batch, batch_idx)
            # if you have multiple val dataloaders:
            def validation_step(self, batch, batch_idx, dataloader_idx)
        Examples:
            .. code-block:: python
                # CASE 1: A single validation dataset
                def validation_step(self, batch, batch_idx):
                    x, y = batch
                    # implement your own
                    out = self(x)
                    loss = self.loss(out, y)
                    # log 6 example images
                    # or generated text... or whatever
                    sample_imgs = x[:6]
                    grid = torchvision.utils.make_grid(sample_imgs)
                    self.logger.experiment.add_image('example_images', grid, 0)
                    # calculate acc
                    labels_hat = torch.argmax(out, dim=1)
                    val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
                    # log the outputs!
                    self.log_dict({'val_loss': loss, 'val_acc': val_acc})
            If you pass in multiple val datasets, validation_step will have an additional argument.
            .. code-block:: python
                # CASE 2: multiple validation datasets
                def validation_step(self, batch, batch_idx, dataloader_idx):
                    # dataloader_idx tells you which dataset this is.
        Note:
            If you don't need to validate you don't need to implement this method.
        Note:
            When the :meth:`validation_step` is called, the model has been put in eval mode
            and PyTorch gradients have been disabled. At the end of validation,
            the model goes back to training mode and gradients are enabled.
        """

    def validation_step_end(self, *args, **kwargs):
        """
        Use this when validating with dp or ddp2 because :meth:`validation_step`
        will operate on only part of the batch. However, this is still optional
        and only needed for things like softmax or NCE loss.
        Note:
            If you later switch to ddp or some other mode, this will still be called
            so that you don't have to change your code.
        .. code-block:: python
            # pseudocode
            sub_batches = split_batches_for_dp(batch)
            batch_parts_outputs = [validation_step(sub_batch) for sub_batch in sub_batches]
            validation_step_end(batch_parts_outputs)
        Args:
            batch_parts_outputs: What you return in :meth:`validation_step`
                for each batch part.
        Return:
            None or anything
        .. code-block:: python
            # WITHOUT validation_step_end
            # if used in DP or DDP2, this batch is 1/num_gpus large
            def validation_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch
                out = self.encoder(x)
                loss = self.softmax(out)
                loss = nce_loss(loss)
                self.log('val_loss', loss)
            # --------------
            # with validation_step_end to do softmax over the full batch
            def validation_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch
                out = self(x)
                return out
            def validation_epoch_end(self, val_step_outputs):
                for out in val_step_outputs:
                    # do something with these
        See Also:
            See the :ref:`multi_gpu` guide for more details.
        """

    def validation_epoch_end(
        self, outputs: List[Any]
    ) -> Dict[str, Any]:
        """
        Called at the end of the validation epoch with the outputs of all validation steps.
        .. code-block:: python
            # the pseudocode for these calls
            val_outs = []
            for val_batch in val_data:
                out = validation_step(val_batch)
                val_outs.append(out)
            validation_epoch_end(val_outs)
        Args:
            outputs: List of outputs you defined in :meth:`validation_step`, or if there
                are multiple dataloaders, a list containing a list of outputs for each dataloader.
        Return:
            None
        Note:
            If you didn't define a :meth:`validation_step`, this won't be called.
        Examples:
            With a single dataloader:
            .. code-block:: python
                def validation_epoch_end(self, val_step_outputs):
                    for out in val_step_outputs:
                        # do something
            With multiple dataloaders, `outputs` will be a list of lists. The outer list contains
            one entry per dataloader, while the inner list contains the individual outputs of
            each validation step for that dataloader.
            .. code-block:: python
                def validation_epoch_end(self, outputs):
                    for dataloader_output_result in outputs:
                        dataloader_outs = dataloader_output_result.dataloader_i_outputs
                    self.log('final_metric', final_value)
        The outputs is List[Dict[str, torch.Tensor]] in default.
        """

        if len(outputs) <= 0:
            raise ValueError(
                "Ouputs should large than zero in validation_epoch_end, please check!")
        res_out: Dict[str, torch.Tensor] = {}
        output_dict = collections.defaultdict(list)
        for output in outputs:
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    output_dict[k].append(v.mean().item())

        for key, value in output_dict.items():
            res_out[key] = torch.tensor(value).mean()

        for key, value in res_out.items():
            self.log(key, value)

        return res_out

    def test_step(self, *args, **kwargs):
        r"""
        Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest
        such as accuracy.
        .. code-block:: python
            # the pseudocode for these calls
            test_outs = []
            for test_batch in test_data:
                out = test_step(test_batch)
                test_outs.append(out)
            test_epoch_end(test_outs)
        Args:
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (int): The index of this batch.
            dataloader_idx (int): The index of the dataloader that produced this batch
                (only if multiple test datasets used).
        Return:
           Any of.
            - Any object or value
            - `None` - Testing will skip to the next batch
        .. code-block:: python
            # if you have one test dataloader:
            def test_step(self, batch, batch_idx)
            # if you have multiple test dataloaders:
            def test_step(self, batch, batch_idx, dataloader_idx)
        Examples:
            .. code-block:: python
                # CASE 1: A single test dataset
                def test_step(self, batch, batch_idx):
                    x, y = batch
                    # implement your own
                    out = self(x)
                    loss = self.loss(out, y)
                    # log 6 example images
                    # or generated text... or whatever
                    sample_imgs = x[:6]
                    grid = torchvision.utils.make_grid(sample_imgs)
                    self.logger.experiment.add_image('example_images', grid, 0)
                    # calculate acc
                    labels_hat = torch.argmax(out, dim=1)
                    test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
                    # log the outputs!
                    self.log_dict({'test_loss': loss, 'test_acc': test_acc})
            If you pass in multiple validation datasets, :meth:`test_step` will have an additional
            argument.
            .. code-block:: python
                # CASE 2: multiple test datasets
                def test_step(self, batch, batch_idx, dataloader_idx):
                    # dataloader_idx tells you which dataset this is.
        Note:
            If you don't need to validate you don't need to implement this method.
        Note:
            When the :meth:`test_step` is called, the model has been put in eval mode and
            PyTorch gradients have been disabled. At the end of the test epoch, the model goes back
            to training mode and gradients are enabled.
        """

    def test_step_end(self, *args, **kwargs):
        """
        Use this when testing with dp or ddp2 because :meth:`test_step` will operate
        on only part of the batch. However, this is still optional
        and only needed for things like softmax or NCE loss.
        Note:
            If you later switch to ddp or some other mode, this will still be called
            so that you don't have to change your code.
        .. code-block:: python
            # pseudocode
            sub_batches = split_batches_for_dp(batch)
            batch_parts_outputs = [test_step(sub_batch) for sub_batch in sub_batches]
            test_step_end(batch_parts_outputs)
        Args:
            batch_parts_outputs: What you return in :meth:`test_step` for each batch part.
        Return:
            None or anything
        .. code-block:: python
            # WITHOUT test_step_end
            # if used in DP or DDP2, this batch is 1/num_gpus large
            def test_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch
                out = self(x)
                loss = self.softmax(out)
                self.log('test_loss', loss)
            # --------------
            # with test_step_end to do softmax over the full batch
            def test_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch
                out = self.encoder(x)
                return out
            def test_epoch_end(self, output_results):
                # this out is now the full size of the batch
                all_test_step_outs = output_results.out
                loss = nce_loss(all_test_step_outs)
                self.log('test_loss', loss)
        See Also:
            See the :ref:`multi_gpu` guide for more details.
        """

    def test_epoch_end(
        self, outputs: List[Any]
    ) -> None:
        """
        Called at the end of a test epoch with the output of all test steps.
        .. code-block:: python
            # the pseudocode for these calls
            test_outs = []
            for test_batch in test_data:
                out = test_step(test_batch)
                test_outs.append(out)
            test_epoch_end(test_outs)
        Args:
            outputs: List of outputs you defined in :meth:`test_step_end`, or if there
                are multiple dataloaders, a list containing a list of outputs for each dataloader
        Return:
            None
        Note:
            If you didn't define a :meth:`test_step`, this won't be called.
        Examples:
            With a single dataloader:
            .. code-block:: python
                def test_epoch_end(self, outputs):
                    # do something with the outputs of all test batches
                    all_test_preds = test_step_outputs.predictions
                    some_result = calc_all_results(all_test_preds)
                    self.log(some_result)
            With multiple dataloaders, `outputs` will be a list of lists. The outer list contains
            one entry per dataloader, while the inner list contains the individual outputs of
            each test step for that dataloader.
            .. code-block:: python
                def test_epoch_end(self, outputs):
                    final_value = 0
                    for dataloader_outputs in outputs:
                        for test_step_out in dataloader_outputs:
                            # do something
                            final_value += test_step_out
                    self.log('final_metric', final_value)
        """

    def configure_optimizers(
        self,
    ):
        r"""
        # TODO: 我们先只确保一个optimizer的情况没问题。
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler' key which value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        Note:
            The 'frequency' value is an int corresponding to the number of sequential batches
            optimized with the specific optimizer. It should be given to none or to all of the optimizers.
            There is a difference between passing multiple optimizers in a list,
            and passing multiple optimizers in dictionaries with a frequency of 1:
            In the former case, all optimizers will operate on the given batch in each optimization step.
            In the latter, only one optimizer will operate on the given batch at every step.
            The lr_dict is a dictionary which contains scheduler and its associated configuration.
            It has five keys. The default configuration is shown below.
            .. code-block:: python
                {
                    'scheduler': lr_scheduler, # The LR schduler
                    'interval': 'epoch', # The unit of the scheduler's step size
                    'frequency': 1, # The frequency of the scheduler
                    'reduce_on_plateau': False, # For ReduceLROnPlateau scheduler
                    'monitor': 'val_loss', # Metric for ReduceLROnPlateau to monitor
                    'strict': True # Whether to crash the training if `monitor` is not found
                }
            If user only provides LR schedulers, then their configuration will set to default as shown above.
        Examples:
            .. code-block:: python
                # most cases
                def configure_optimizers(self):
                    opt = Adam(self.parameters(), lr=1e-3)
                    return opt
                # multiple optimizer case (e.g.: GAN)
                def configure_optimizers(self):
                    generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
                    disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
                    return generator_opt, disriminator_opt
                # example with learning rate schedulers
                def configure_optimizers(self):
                    generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
                    disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
                    discriminator_sched = CosineAnnealing(discriminator_opt, T_max=10)
                    return [generator_opt, disriminator_opt], [discriminator_sched]
                # example with step-based learning rate schedulers
                def configure_optimizers(self):
                    gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
                    dis_opt = Adam(self.model_disc.parameters(), lr=0.02)
                    gen_sched = {'scheduler': ExponentialLR(gen_opt, 0.99),
                                 'interval': 'step'}  # called after each training step
                    dis_sched = CosineAnnealing(discriminator_opt, T_max=10) # called every epoch
                    return [gen_opt, dis_opt], [gen_sched, dis_sched]
                # example with optimizer frequencies
                # see training procedure in `Improved Training of Wasserstein GANs`, Algorithm 1
                # https://arxiv.org/abs/1704.00028
                def configure_optimizers(self):
                    gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
                    dis_opt = Adam(self.model_disc.parameters(), lr=0.02)
                    n_critic = 5
                    return (
                        {'optimizer': dis_opt, 'frequency': n_critic},
                        {'optimizer': gen_opt, 'frequency': 1}
                    )
        Note:
            Some things to know:
            - We calls ``.backward()`` and ``.step()`` on each optimizer
              and learning rate scheduler as needed.
            - If you use 16-bit precision (``precision=16``), We will automatically
              handle the optimizers for you.
            - If you use multiple optimizers, :meth:`training_step` will have an additional
              ``optimizer_idx`` parameter.
            - If you use LBFGS Lightning handles the closure function automatically for you.
            - If you use multiple optimizers, gradients will be calculated only
              for the parameters of current optimizer at each training step.
            - If you need to control how often those optimizers step or override the
              default ``.step()`` schedule, override the :meth:`optimizer_step` hook.
            - If you only want to call a learning rate scheduler every ``x`` step or epoch,
              or want to monitor a custom metric, you can specify these in a lr_dict:
              .. code-block:: python
                  {
                      'scheduler': lr_scheduler,
                      'interval': 'step',  # or 'epoch'
                      'monitor': 'val_f1',
                      'frequency': x,
                  }
        """
        rank_zero_warn(
            "`configure_optimizers` must be implemented to be used with the Lightning Trainer"
        )

    # def summarize(self, mode: str = ModelSummary.MODE_DEFAULT) -> ModelSummary:
    #     # TODO: 涉及到ModelSummary的类，后面看看
    #     model_summary = ModelSummary(self, mode=mode)
    #     log.info("\n" + str(model_summary))
    #     return model_summary

    def trace(self, *args, **kwargs) -> Any:
        """
        该方法主要是用来做model trace , 自定义model需要实现该方法才可以trace.
        Examples:
            .. code-block:: python
            def trace(self, text_input_ids,
                      text_token_type_ids, text_mask,
                      vision_embeds, vision_masks):
                if self.need_img_factorize:
                    obj_reps= self.img_emb_decoder(obj_reps)

                text_mask = text_mask.bool()
                text_tags = text_input_ids.new_zeros(text_input_ids.shape)

                text_visual_embeddings = self._collect_obj_reps(
                    text_tags, obj_reps)
                object_layer_out, pooled_out, att \
                    = self.vlbert(text_input_ids,
                                text_token_type_ids,
                                text_mask,
                                text_visual_embeddings,
                                obj_reps,
                                box_mask,
                                output_text_and_object_separately=False,
                                output_all_encoded_layers=False,
                                output_attention_probs=True,
                                output_mask=False,
                                is_text_visual=True)
                query_pooled_out_single = pooled_out[-1]
                query_text_visual_out = pooled_out[:-1]

                query_pooled_out = query_pooled_out_single.expand(query_text_visual_out.size(0), query_text_visual_out.size(1))
                relevance_logits = self.relevance_head(query_pooled_out, query_text_visual_out)
                return relevance_logits
        """

        raise ValueError("Trace method not implement in Net, please check! ")

    def freeze(self) -> None:
        r"""
        Freeze all params for inference.
        Example:
            .. code-block:: python
                model = MyLightningModule(...)
                model.freeze()
        """
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self) -> None:
        """
        Unfreeze all parameters for training.
        .. code-block:: python
            model = MyLightningModule(...)
            model.unfreeze()
        """
        for param in self.parameters():
            param.requires_grad = True

        self.train()

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        r"""
        # TODO: 其实不太喜欢progress bar，先跳过
        Implement this to override the default items displayed in the progress bar.
        By default it includes the average loss value, split index of BPTT (if used)
        and the version of the experiment when using a logger.
        .. code-block::
            Epoch 1:   4%|▎         | 40/1095 [00:03<01:37, 10.84it/s, loss=4.501, v_num=10]
        Here is an example how to override the defaults:
        .. code-block:: python
            def get_progress_bar_dict(self):
                # don't show the version number
                items = super().get_progress_bar_dict()
                items.pop("v_num", None)
                return items
        Return:
            Dictionary with the items to be displayed in the progress bar.
        """
        # call .item() only once but store elements without graphs
        running_train_loss = self.trainer.train_loop.running_loss.mean()
        avg_training_loss = (
            running_train_loss.cpu().item()
            if running_train_loss is not None
            else float("NaN")
        )
        tqdm_dict = {"loss": "{:.3f}".format(avg_training_loss)}

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict["split_idx"] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            version = self.trainer.logger.version
            # show last 4 places of long version strings
            version = version[-4:] if isinstance(version, str) else version
            tqdm_dict["v_num"] = version

        return tqdm_dict  # type: ignore

    def _verify_is_manual_optimization(self, fn_name):
        if self.trainer.train_loop.automatic_optimization:
            m = f'to use {fn_name}, please disable automatic optimization: Trainer(automatic_optimization=False)'
            raise Exception(m)

    def do_with_frequence(self, config_frequence):
        """
        根据设置的config_frequence返回True和False
        """
        if config_frequence != 0 and self.global_step % config_frequence == 0:
            return True
        return False

    def data_to_channels_last(self, data: torch.Tensor) -> torch.Tensor:
        """
        将data从 NCHW 转为 NHWC
        """
        if len(data.size()) != 4:
            return data
        data = data.to(memory_format=torch.channels_last)  # image to channels_last
        return data
