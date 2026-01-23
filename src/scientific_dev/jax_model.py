import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
import optax
from typing import Tuple, List
from .schema import TrainingConfig, ModelConfig

def init_mlp_params(layer_widths: List[int], input_dim: int, key: jax.random.PRNGKey):
    """Initializes weights and biases for an MLP."""
    params = []
    keys = jax.random.split(key, len(layer_widths) + 1)
    
    current_dim = input_dim
    for i, next_dim in enumerate(layer_widths):
        w_key = keys[i]
        # Xavier/Glorot initialization
        scale = jnp.sqrt(2.0 / (current_dim + next_dim))
        w = jax.random.normal(w_key, (current_dim, next_dim)) * scale
        b = jnp.zeros((next_dim,))
        params.append((w, b))
        current_dim = next_dim
        
    # Output layer (Single node for binary classification)
    w_out = jax.random.normal(keys[-1], (current_dim, 1)) * jnp.sqrt(2.0 / (current_dim + 1))
    b_out = jnp.zeros((1,))
    params.append((w_out, b_out))
    
    return params

def mlp_forward(params, x):
    """Forward pass through the MLP."""
    for w, b in params[:-1]:
        x = jnp.dot(x, w) + b
        x = jax.nn.relu(x)
    
    w_last, b_last = params[-1]
    logits = jnp.dot(x, w_last) + b_last
    return logits.reshape(-1)

def binary_cross_entropy_loss(params, x, y):
    """Calculates binary cross-entropy loss."""
    logits = mlp_forward(params, x)
    # Using optax's stable sigmoid cross entropy
    loss = optax.sigmoid_binary_cross_entropy(logits, y)
    return jnp.mean(loss)

@jit
def train_step(params, opt_state, x, y, optimizer):
    """A single JIT-compiled training step."""
    loss, grads = value_and_grad(binary_cross_entropy_loss)(params, x, y)
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    next_params = optax.apply_updates(params, updates)
    return next_params, next_opt_state, loss

def evaluate_model(params, x, y):
    """Evaluates the model on a dataset."""
    logits = mlp_forward(params, x)
    preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
    accuracy = jnp.mean(preds == y)
    return accuracy

if __name__ == "__main__":
    # Simple test with synthetic data
    from .schema import ExperimentConfig
    config = ExperimentConfig(name="test_run")
    
    key = jax.random.PRNGKey(config.training.seed)
    input_dim = 28
    n_samples = 1000
    
    X = jax.random.normal(key, (n_samples, input_dim))
    y = jax.random.bernoulli(key, 0.5, (n_samples,)).astype(jnp.float32)
    
    params = init_mlp_params(config.model.hidden_dims, input_dim, key)
    optimizer = optax.adam(config.training.learning_rate)
    opt_state = optimizer.init(params)
    
    print(f"Starting training on synthetic data...")
    for i in range(100):
        params, opt_state, loss = train_step(params, opt_state, X, y, optimizer)
        if i % 20 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")
    
    acc = evaluate_model(params, X, y)
    print(f"Final Synthetic Accuracy: {acc:.4f}")
