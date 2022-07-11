import re
from typing import Optional, Union, Callable, List

from tensorflow_addons.optimizers.weight_decay_optimizers import DecoupledWeightDecayExtension
from typeguard import typechecked

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike
from tensorflow.python.keras.optimizer_v2 import optimizer_v2


class AdaBelief(DecoupledWeightDecayExtension, optimizer_v2.OptimizerV2):
    """Optimizer that implements the Adabelief.
    The code is written so as to implement Decoupled Weight Decay.

    See paper [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients](https://arxiv.org/abs/2010.07468).
    """

    @typechecked
    def __init__(
            self,
            learning_rate: Union[FloatTensorLike, Callable] = 0.001,
            beta_1: FloatTensorLike = 0.9,
            beta_2: FloatTensorLike = 0.999,
            epsilon: FloatTensorLike = 1e-8,
            weight_decay: Union[FloatTensorLike, Callable] = 0.0,
            weight_decay_rate: FloatTensorLike = 0.0,
            exclude_from_weight_decay: Optional[List[str]] = None,
            remove_nans: Optional[List[str]] = None,
            remove_mean=False,
            name: str = "AdaBelief",
            weight_noise = None,
            **kwargs
    ):
        """Construct a new AdaBelief optimizer.

        Args:
            learning_rate: A `Tensor` or a floating point value. or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
                The learning rate.
            beta_1: A `float` value or a constant `float` tensor.
              The exponential decay rate for the 1st moment estimates.
            beta_2: A `float` value or a constant `float` tensor.
              The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            weight_decay_rate: weight decay rate.
            exclude_from_weight_decay: List of regex patterns of
              variables excluded from weight decay. Variables whose name
              contain a substring matching the pattern will be excluded.
            name: Optional name for the operations created when applying
              gradients. Defaults to "AdaBelief".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
              `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
              norm; `clipvalue` is clip gradients by value, `decay` is
              included for backward compatibility to allow time inverse
              decay of learning rate. `lr` is included for backward
              compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(
            weight_decay=weight_decay,
            name=name,
            **kwargs,
        )

        # super().__init__(name, **kwargs)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't interact
        # with the m/v parameters.
        self._set_hyper("weight_decay_rate", weight_decay_rate)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

        # This is learning rate decay for using keras learning rate schedule.
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon or tf.backend_config.epsilon()
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self.remove_nans = remove_nans
        self.remove_mean = remove_mean
        self.weight_noise = self._select_noise_type(weight_noise)

        assert self.remove_mean in [False, 0, 1]

    def _select_noise_type(self, weight_noise):
        if weight_noise is None:
            return lambda x: 0
        else:
            return lambda x: weight_noise*tf.random.uniform(x)


    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = tf.identity(self._get_hyper("beta_2", var_dtype))
        weight_decay_rate = tf.identity(self._get_hyper("weight_decay_rate", var_dtype))
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        apply_state[(var_device, var_dtype)].update(
            dict(
                weight_decay_rate=weight_decay_rate,
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t,
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * coefficients["one_minus_beta_1_t"]
        m_t = m * coefficients["beta_1_t"] + m_scaled_g_values
        m_t = m.assign(m_t, use_locking=self._use_locking)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad - m_t) * (grad - m_t) * coefficients["one_minus_beta_2_t"]
        v_t = v * coefficients["beta_2_t"] + v_scaled_g_values
        v_t = v.assign(v_t, use_locking=self._use_locking)

        m_t_hat = m_t / (1.0 - coefficients["beta_1_power"])
        v_t_hat = v_t / (1.0 - coefficients["beta_2_power"])

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + coefficients["epsilon"])

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += coefficients["weight_decay_rate"] * var

        update = self._check_nans(var_name, update)

        var_update = var - coefficients["lr_t"] * update

        if not self.remove_mean == False and len(tf.shape(var)) == 2:
            mean = tf.reduce_mean(var, axis=self.remove_mean)
            mean = tf.expand_dims(mean, axis=self.remove_mean)
            var_update = var - mean

        var_update += self.weight_noise(var.shape)
        return var.assign(var_update, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # return self._resource_apply_dense(grad, var, apply_state=apply_state)
        # raise NotImplementedError
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        m_scaled_g_values = grad * coefficients["one_minus_beta_1_t"]
        m_t = m.assign(m * coefficients["beta_1_t"], use_locking=self._use_locking)
        v_scaled_g_values = (grad - tf.gather(m_t, indices)) ** 2 * coefficients["one_minus_beta_2_t"]
        v_t = v.assign(v * coefficients["beta_2_t"], use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        # gm2 = tf.square(tf.math.subtract(grad, m))
        # v_scaled_g_values = tf.square(tf.math.subtract(grad, m_t)) * coefficients["one_minus_beta_2_t"]

        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        m_t_hat = m_t / (1.0 - coefficients["beta_1_power"])
        v_t_hat = v_t / (1.0 - coefficients["beta_2_power"])

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + coefficients["epsilon"])

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += coefficients["weight_decay_rate"] * var

        update = self._check_nans(var_name, update)

        var_update = var.assign_sub(
            coefficients["lr_t"] * update, use_locking=self._use_locking
        )

        if not self.remove_mean == False and len(tf.shape(var)) == 2:
            mean = tf.reduce_mean(var, axis=self.remove_mean)
            mean = tf.expand_dims(mean, axis=self.remove_mean)
            var_update = var.assign_sub(mean, use_locking=self._use_locking)

        var_update = var_update + self.weight_noise(var.shape)

        return tf.group(*[var_update, m_t, v_t])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "weight_decay_rate": self._serialize_hyperparameter(
                    "weight_decay_rate"
                ),
                "decay": self._serialize_hyperparameter("decay"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self.epsilon,
            }
        )
        return config

    def _check_nans(self, param_name, update):
        """Remove nans in the update rule."""
        if self.remove_nans:
            for r in self.remove_nans:
                if re.search(r, param_name) is not None:
                    non_nans = 1 - tf.cast(tf.math.is_nan(update), tf.float32)
                    update = tf.math.multiply_no_nan(update, non_nans)

            if 'all' in self.remove_nans:
                non_nans = 1 - tf.cast(tf.math.is_nan(update), tf.float32)
                update = tf.math.multiply_no_nan(update, non_nans)
        return update

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
