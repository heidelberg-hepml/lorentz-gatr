import torch

from gatr.utils.tensors import block_stack


def test_block_stack():
    """Test block stacking."""
    x = torch.rand(4, 2, 3, 5)
    y = torch.rand(4, 4, 5, 5)
    z = torch.rand(4, 3, 3, 5)
    out = block_stack([x, y, z], 1, 2)
    assert list(out.shape) == [4, 9, 11, 5]
    torch.testing.assert_close(out[:, 0:2, 0:3, :], x)
    torch.testing.assert_close(out[:, 2:6, 3:8, :], y)
    torch.testing.assert_close(out[:, 6:9, 8:11, :], z)
    out[:, 0:2, 0:3, :] = 0
    out[:, 2:6, 3:8, :] = 0
    out[:, 6:9, 8:11, :] = 0
    torch.testing.assert_close(out, torch.zeros_like(out))
