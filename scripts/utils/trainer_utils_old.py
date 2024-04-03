import torch
from transformers import Trainer
from transformers.utils import (
    is_sagemaker_mp_enabled, 
    is_apex_available,
)

if is_apex_available():
    from apex import amp

class AWP:
    def __init__(self, model, optimizer, adv_param='weight',
                 adv_lr=0.0001, adv_eps=0.0001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def perturb(self):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        self._save()  # save model parameters
        self._attack_step()  # perturb weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param]['exp_avg']
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Set lower and upper limit in change
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # Perturb along gradient
                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

                    # Apply the limit to the change
                    param.data.clamp_(param_min, param_max)

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])


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

        # AWP実行
        if self.state.epoch >= self.awp_start:
            self.awp.perturb()

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

        return loss.detach() / self.args.gradient_accumulation_steps
