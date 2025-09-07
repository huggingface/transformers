import torch
import pytest
from torch import nn
from transformers import TrainingArguments, Trainer


class DummyModel(nn.Module):
    """Simple dummy model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, **kwargs):
        return torch.tensor(1.0)


class DummyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call_reduce_loss(self, loss, num_items_in_batch=None):
        return self._reduce_loss(loss, num_items_in_batch=num_items_in_batch)


@pytest.fixture
def dummy_trainer(tmp_path):
    args = TrainingArguments(output_dir=tmp_path, per_device_train_batch_size=2)
    model = DummyModel()
    return DummyTrainer(model=model, args=args)


def test_reduce_loss_single_gpu_no_reduction(dummy_trainer):
    """Test that on single GPU, loss is returned unchanged."""
    dummy_trainer.args._n_gpu = 1
    loss = torch.tensor([2.0, 4.0])
    reduced = dummy_trainer.call_reduce_loss(loss)
    assert torch.equal(reduced, loss)


def test_reduce_loss_multi_gpu_mean(dummy_trainer):
    """Test that on multi-GPU without token count, loss.mean() is used."""
    dummy_trainer.args._n_gpu = 2
    loss = torch.tensor([2.0, 4.0])
    reduced = dummy_trainer.call_reduce_loss(loss)
    expected = loss.mean()  # Should be 3.0
    assert torch.isclose(reduced, expected)


def test_reduce_loss_multi_gpu_sum_with_tokens(dummy_trainer):
    """Test that on multi-GPU with token count, loss.sum() / num_items_in_batch is used."""
    dummy_trainer.args._n_gpu = 2
    # Simulate two partial losses on two GPUs
    loss = torch.tensor([2.0, 4.0])
    num_items_in_batch = 2
    reduced = dummy_trainer.call_reduce_loss(loss, num_items_in_batch=num_items_in_batch)
    # Should do sum / num_items_in_batch = (2+4)/2 = 3
    expected = loss.sum() / num_items_in_batch
    assert torch.isclose(reduced, expected)


def test_reduce_loss_multi_gpu_different_batch_sizes(dummy_trainer):
    """Test loss reduction with different batch sizes."""
    dummy_trainer.args._n_gpu = 4
    loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
    num_items_in_batch = 10  # Total tokens across all devices
    reduced = dummy_trainer.call_reduce_loss(loss, num_items_in_batch=num_items_in_batch)
    # Should do sum / num_items_in_batch = (1+2+3+4)/10 = 1.0
    expected = loss.sum() / num_items_in_batch
    assert torch.isclose(reduced, expected)


def test_reduce_loss_single_element_tensor(dummy_trainer):
    """Test with single element tensor (common case)."""
    dummy_trainer.args._n_gpu = 2
    loss = torch.tensor(5.0)
    reduced = dummy_trainer.call_reduce_loss(loss)
    # Single element mean is the element itself
    assert torch.isclose(reduced, loss)


def test_reduce_loss_zero_loss(dummy_trainer):
    """Test with zero loss values."""
    dummy_trainer.args._n_gpu = 2
    loss = torch.tensor([0.0, 0.0])
    num_items_in_batch = 5
    reduced = dummy_trainer.call_reduce_loss(loss, num_items_in_batch=num_items_in_batch)
    assert torch.isclose(reduced, torch.tensor(0.0))


def test_reduce_loss_preserves_gradients(dummy_trainer):
    """Test that gradient information is preserved."""
    dummy_trainer.args._n_gpu = 2
    loss = torch.tensor([2.0, 4.0], requires_grad=True)
    reduced = dummy_trainer.call_reduce_loss(loss)
    assert reduced.requires_grad
    
    # Test backward pass works
    reduced.backward()
    assert loss.grad is not None


def test_reduce_loss_with_tensor_token_count(dummy_trainer):
    """Test with tensor token count (as would come from actual training)."""
    dummy_trainer.args._n_gpu = 2
    loss = torch.tensor([3.0, 6.0])
    num_items_in_batch = torch.tensor(3)  # Tensor instead of int
    reduced = dummy_trainer.call_reduce_loss(loss, num_items_in_batch=num_items_in_batch)
    expected = loss.sum() / num_items_in_batch
    assert torch.isclose(reduced, expected)
