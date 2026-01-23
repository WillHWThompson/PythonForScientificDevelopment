import jax
import jax.numpy as jnp
import optax
import pytest
from src.scientific_dev.jax_model import (
    init_mlp_params, mlp_forward, binary_cross_entropy_loss, train_step
)

def test_mlp_initialization():
    key = jax.random.PRNGKey(0)
    input_dim = 28
    hidden_dims = [64, 32]
    params = init_mlp_params(hidden_dims, input_dim, key)
    
    # Check number of layers (hidden + output)
    assert len(params) == len(hidden_dims) + 1
    
    # Check shapes
    # First hidden layer: (input, 64)
    assert params[0][0].shape == (28, 64)
    # Last output layer: (32, 1)
    assert params[-1][0].shape == (32, 1)

def test_mlp_forward_pass():
    key = jax.random.PRNGKey(0)
    input_dim = 28
    hidden_dims = [64]
    params = init_mlp_params(hidden_dims, input_dim, key)
    
    x = jax.random.normal(key, (10, input_dim))
    logits = mlp_forward(params, x)
    
    # Check output shape (batch_size,)
    assert logits.shape == (10,)

def test_train_step():
    key = jax.random.PRNGKey(0)
    input_dim = 28
    params = init_mlp_params([16], input_dim, key)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    
    X = jax.random.normal(key, (32, input_dim))
    y = jax.random.bernoulli(key, 0.5, (32,)).astype(jnp.float32)
    
    # Perform a few steps and ensure loss decreases or at least returns a valid number
    _, _, initial_loss = train_step(params, opt_state, X, y, optimizer)
    assert not jnp.isnan(initial_loss)
    assert initial_loss > 0
