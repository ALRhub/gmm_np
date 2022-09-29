from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from metalearning_model_gmm_np.util import assert_shape


class Decoder(ABC):
    def __init__(
        self,
        d_x: int,
        d_y: int,
        d_z: int,
        n_hidden: int,
        d_hidden: int,
        activation: str,
    ):
        # check input
        self.d_x = d_x
        self.d_y = d_y
        self.d_z = d_z
        self.n_hidden = n_hidden
        self.d_hidden = d_hidden
        self.activation = activation

        # define mean MLP
        self._mean_mlp = tf.keras.Sequential()
        self._mean_mlp.add(keras.layers.Input(shape=(None, None, d_x + d_z)))
        for _ in range(n_hidden):
            self._mean_mlp.add(
                keras.layers.Dense(units=d_hidden, activation=activation)
            )
        self._mean_mlp.add(keras.layers.Dense(units=d_y, activation=None))

    @property
    @abstractmethod
    def trainable_weights_std(self) -> List[tf.Tensor]:
        pass

    @property
    @abstractmethod
    def trackables_std(self) -> List:
        pass

    @abstractmethod
    def _compute_variance(self, x: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        pass

    @property
    def trainable_weights_mean(self) -> List[tf.Tensor]:
        return self._mean_mlp.trainable_weights

    @property
    def trainable_weights(self) -> List[tf.Tensor]:
        return self.trainable_weights_mean + self.trainable_weights_std
    
    @property
    def trackables_mean(self) -> List:
        return [self._mean_mlp]

    @property
    def trackables(self) -> List:
        return self.trackables_mean + self.trackables_std

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        ]
    )
    def _compute_mean(self, x: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        # check inputs
        n_tasks = tf.shape(x)[0]
        n_points = tf.shape(x)[1]
        n_samples = tf.shape(z)[0]

        # compute mean
        x = tf.broadcast_to(x[None, ...], (n_samples, n_tasks, n_points, self.d_x))
        z = tf.broadcast_to(z[..., None, :], (n_samples, n_tasks, n_points, self.d_z))
        xz = tf.concat((x, z), axis=-1)
        assert_shape(xz, (n_samples, n_tasks, n_points, self.d_x + self.d_z))
        mu = self._mean_mlp(xz)

        return mu

    def compute_mean(self, x: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        # check inputs
        n_tasks = tf.shape(x)[0]
        n_points = tf.shape(x)[1]
        n_samples = tf.shape(z)[0]
        assert_shape(x, (n_tasks, n_points, self.d_x))
        assert_shape(z, (n_samples, n_tasks, self.d_z))

        mu = self._compute_mean(x=x, z=z)

        # check output
        assert_shape(mu, (n_samples, n_tasks, n_points, self.d_y))
        return mu

    def compute_variance(self, x: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        # check inputs
        n_tasks = tf.shape(x)[0]
        n_points = tf.shape(x)[1]
        n_samples = tf.shape(z)[0]
        assert_shape(x, (n_tasks, n_points, self.d_x))
        assert_shape(z, (n_samples, n_tasks, self.d_z))

        var = self._compute_variance(x=x, z=z)

        # check output
        assert_shape(var, (n_samples, n_tasks, n_points, self.d_y))
        return var

    def __call__(self, x: tf.Tensor, z: tf.Tensor) -> Tuple[tf.Tensor]:
        return self.compute_mean(x=x, z=z), self.compute_variance(x=x, z=z)


class DecoderWithFixedStdY(Decoder):
    def __init__(
        self,
        d_x: int,
        d_y: int,
        d_z: int,
        n_hidden: int,
        d_hidden: int,
        activation: str,
        fixed_std_y: float,
    ):
        # check input
        assert fixed_std_y > 0.0

        super().__init__(
            d_x=d_x,
            d_y=d_y,
            d_z=d_z,
            n_hidden=n_hidden,
            d_hidden=d_hidden,
            activation=activation,
        )
        self._fixed_var = tf.convert_to_tensor(fixed_std_y**2, dtype=tf.float32)

    @property
    def trainable_weights_std(self) -> List[tf.Tensor]:
        return []

    @property
    def trackables_std(self) -> List:
        return []

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        ]
    )
    def _compute_variance(self, x: tf.Tensor, z: tf.Tensor):
        # check input
        n_tasks = tf.shape(x)[0]
        n_points = tf.shape(x)[1]
        n_samples = tf.shape(z)[0]

        # compute output variance
        var = tf.broadcast_to(self._fixed_var, (n_samples, n_tasks, n_points, self.d_y))

        return var


class DecoderWithLearnedVectorStdY(Decoder):
    def __init__(
        self,
        d_x: int,
        d_y: int,
        d_z: int,
        n_hidden: int,
        d_hidden: int,
        activation: str,
        std_y_lower_bound: float,
    ):
        # check input
        assert std_y_lower_bound > 0.0
        super().__init__(
            d_x=d_x,
            d_y=d_y,
            d_z=d_z,
            n_hidden=n_hidden,
            d_hidden=d_hidden,
            activation=activation,
        )

        # define trainable std_y vector
        self._var_lower_bound = tf.convert_to_tensor(
            std_y_lower_bound**2, dtype=tf.float32
        )
        var_init = tf.math.maximum(self._var_lower_bound, 1.0)
        var_init = var_init * tf.ones((self.d_y,), dtype=tf.float32)
        var_unconstrained_init = self.constrained_to_unconstrained(var_init)
        self._var_unconstrained = tf.Variable(
            var_unconstrained_init,
            trainable=True,
            shape=((self.d_y,)),
            dtype=tf.float32,
        )

    @property
    def trainable_weights_std(self) -> List[tf.Tensor]:
        return [self._var_unconstrained]

    @property
    def trackables_std(self) -> List:
        return [self._var_unconstrained]

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def constrained_to_unconstrained(self, var_constrained: tf.Tensor) -> tf.Tensor:
        return tfp.math.softplus_inverse(var_constrained - self._var_lower_bound)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def unconstrained_to_constrained(self, var_unconstrained: tf.Tensor) -> tf.Tensor:
        return self._var_lower_bound + tf.math.softplus(var_unconstrained)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        ]
    )
    def _compute_variance(self, x: tf.Tensor, z: tf.Tensor):
        # check input
        n_tasks = tf.shape(x)[0]
        n_points = tf.shape(x)[1]
        n_samples = tf.shape(z)[0]

        # compute output variance
        var = self.unconstrained_to_constrained(self._var_unconstrained)
        var = tf.broadcast_to(var, (n_samples, n_tasks, n_points, self.d_y))

        return var


class DecoderWithXFeatureStdY(Decoder):
    def __init__(
        self,
        d_x: int,
        d_y: int,
        d_z: int,
        n_hidden: int,
        d_hidden: int,
        activation: str,
        std_y_lower_bound: float,
    ):
        # check input
        assert std_y_lower_bound > 0.0
        super().__init__(
            d_x=d_x,
            d_y=d_y,
            d_z=d_z,
            n_hidden=n_hidden,
            d_hidden=d_hidden,
            activation=activation,
        )

        # define variance MLP
        self._variance_mlp = tf.keras.Sequential()
        self._variance_mlp.add(keras.layers.Input(shape=(None, d_x)))
        for _ in range(n_hidden):
            self._variance_mlp.add(
                keras.layers.Dense(units=d_hidden, activation=activation)
            )
        self._variance_mlp.add(keras.layers.Dense(units=d_y, activation=None))
        self._var_lower_bound = tf.convert_to_tensor(
            std_y_lower_bound**2, dtype=tf.float32
        )

    @property
    def trainable_weights_std(self) -> List[tf.Tensor]:
        return self._variance_mlp.trainable_weights

    @property
    def trackables_std(self) -> List:
        return [self._variance_mlp]

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)]
    )
    def unconstrained_to_constrained(self, var_unconstrained: tf.Tensor) -> tf.Tensor:
        return self._var_lower_bound + tf.math.softplus(var_unconstrained)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        ]
    )
    def _compute_variance(self, x: tf.Tensor, z: tf.Tensor):
        # check inputs
        n_tasks = tf.shape(x)[0]
        n_points = tf.shape(x)[1]
        n_samples = tf.shape(z)[0]

        # compute variance
        var = self._variance_mlp(x)
        assert_shape(var, (n_tasks, n_points, self.d_y))
        var = self.unconstrained_to_constrained(var)
        var = tf.broadcast_to(var[None, ...], (n_samples, n_tasks, n_points, self.d_y))

        return var

class DecoderWithXZFeatureStdY(Decoder):
    def __init__(
        self,
        d_x: int,
        d_y: int,
        d_z: int,
        n_hidden: int,
        d_hidden: int,
        activation: str,
        std_y_lower_bound: float,
    ):
        # check input
        assert std_y_lower_bound > 0.0
        super().__init__(
            d_x=d_x,
            d_y=d_y,
            d_z=d_z,
            n_hidden=n_hidden,
            d_hidden=d_hidden,
            activation=activation,
        )

        # define variance MLP
        self._variance_mlp = tf.keras.Sequential()
        self._variance_mlp.add(keras.layers.Input(shape=(None, None, d_x+d_z)))
        for _ in range(n_hidden):
            self._variance_mlp.add(
                keras.layers.Dense(units=d_hidden, activation=activation)
            )
        self._variance_mlp.add(keras.layers.Dense(units=d_y, activation=None))
        self._var_lower_bound = tf.convert_to_tensor(
            std_y_lower_bound**2, dtype=tf.float32
        )

    @property
    def trainable_weights_std(self) -> List[tf.Tensor]:
        return self._variance_mlp.trainable_weights
        
    @property
    def trackables_std(self) -> List:
        return [self._variance_mlp]

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)]
    )
    def unconstrained_to_constrained(self, var_unconstrained: tf.Tensor) -> tf.Tensor:
        return self._var_lower_bound + tf.math.softplus(var_unconstrained)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        ]
    )
    def _compute_variance(self, x: tf.Tensor, z: tf.Tensor):
        # check inputs
        n_tasks = tf.shape(x)[0]
        n_points = tf.shape(x)[1]
        n_samples = tf.shape(z)[0]

        # compute variance
        x = tf.broadcast_to(x[None, ...], (n_samples, n_tasks, n_points, self.d_x))
        z = tf.broadcast_to(z[..., None, :], (n_samples, n_tasks, n_points, self.d_z))
        xz = tf.concat((x, z), axis=-1)
        assert_shape(xz, (n_samples, n_tasks, n_points, self.d_x + self.d_z))
        var = self._variance_mlp(xz)
        var = self.unconstrained_to_constrained(var)


        return var

def decoder_builder(
    d_x: int,
    d_y: int,
    d_z: int,
    n_hidden: int,
    d_hidden: int,
    activation: str,
    std_y_features: str,
    std_y_lower_bound: Optional[float] = None,
    fixed_std_y: Optional[np.ndarray] = None,
) -> Decoder:
    match std_y_features:
        case "fixed":
            return DecoderWithFixedStdY(
                d_x=d_x,
                d_y=d_y,
                d_z=d_z,
                n_hidden=n_hidden,
                d_hidden=d_hidden,
                activation=activation,
                fixed_std_y=fixed_std_y,
            )
        case "learned_float":
            return DecoderWithLearnedVectorStdY(
                d_x=d_x,
                d_y=d_y,
                d_z=d_z,
                n_hidden=n_hidden,
                d_hidden=d_hidden,
                activation=activation,
                std_y_lower_bound=std_y_lower_bound,
            )
        case "x":
            return DecoderWithXFeatureStdY(
                d_x=d_x,
                d_y=d_y,
                d_z=d_z,
                n_hidden=n_hidden,
                d_hidden=d_hidden,
                activation=activation,
                std_y_lower_bound=std_y_lower_bound,
            )
        case "xz":
            return DecoderWithXZFeatureStdY(
                d_x=d_x,
                d_y=d_y,
                d_z=d_z,
                n_hidden=n_hidden,
                d_hidden=d_hidden,
                activation=activation,
                std_y_lower_bound=std_y_lower_bound,
            )
        case _:
            raise NotImplementedError
