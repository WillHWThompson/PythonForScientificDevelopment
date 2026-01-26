import pytest
from src.core.schema import NLPModelConfig, NLPExperimentConfig
from src.data.loader import TextClassifierDataLoader

def test_model_config_validation():
    """Unit Test: Ensures Pydantic catches invalid hyperparams."""
    # Valid config
    config = NLPModelConfig(adapter_dim=128)
    assert config.adapter_dim == 128
    
    # Invalid config (negative dimension)
    with pytest.raises(ValueError):
        NLPModelConfig(adapter_dim=-1)

def test_dataloader_init():
    """Integration Test: Ensures the 'plumbing' between schema and loader works."""
    exp_config = NLPExperimentConfig()
    
    # Check if we can instantiate the loader with the default schema
    loader = TextClassifierDataLoader(
        data_config=exp_config.data,
        train_config=exp_config.training,
        model_name=exp_config.model.model_name
    )
    
    assert loader.tokenizer is not None
    assert exp_config.data.dataset_name == "ag_news"

if __name__ == "__main__":
    # Allow running directly
    test_model_config_validation()
    test_dataloader_init()
    print("All examples tests passed!")
