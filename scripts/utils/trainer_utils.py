import torch
from transformers import Trainer
from transformers.utils import (
    is_sagemaker_mp_enabled, 
    is_apex_available,
)

if is_apex_available():
    from apex import amp

class AWP:
    def __init__(
        self,
        model,
        optimizer,
        compute_loss_fn,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
    ):
        self.model = model
        self.optimizer = optimizer
        self.compute_loss_fn = compute_loss_fn
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def __call__(self, inputs):
        self._save()
        self._attack_step() 
        adv_loss = self.compute_loss_fn(self.model, inputs)   
        adv_loss.backward()  
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class Trainer_Awp(Trainer):
    def __init__(
        self, 
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        adv_param='weight', 
        adv_lr=0.001,
        adv_eps=0.001, 
        awp_start=1
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics
        )
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.awp_start = awp_start

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        self.awp = AWP(
            self.model, 
            self.optimizer, 
            self.compute_loss, 
            self.adv_param, 
            self.adv_lr, 
            self.adv_eps
        )

    def training_step(self, model, inputs):
        """
        AWPありのtraining_step
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            from transformers.trainer_pt_utils import smp_forward_backward
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        # AWP実行
        if self.state.epoch >= self.awp_start:
            try:
                if self.state.log_history[-1]["eval_f5"] >= 0.85:
                    self.awp(inputs)
            except:
                try:
                    if self.state.log_history[-2]["eval_f5"] >= 0.85:
                        self.awp(inputs)
                except:
                    print("AWP is not executed.")
                    print(self.state.log_history[-1].keys())
                    print(self.state.log_history[-2].keys())

        return loss.detach() / self.args.gradient_accumulation_steps
