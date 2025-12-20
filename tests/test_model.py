import unittest
import torch
from config import ModelArgs
from model import LightronTransformer


class TestLightronModel(unittest.TestCase):
    def setUp(self):
        self.args = ModelArgs(
            dim=256,
            n_layers=2,
            n_heads=4,
            vocab_size=1000,
            max_seq_len=64
        )
        self.model = LightronTransformer(self.args)

    def test_forward_shape(self):
        batch_size = 2
        seq_len = 32
        x = torch.randint(0, self.args.vocab_size, (batch_size, seq_len))

        output = self.model(x)

        # 验证输出形状 [B, S, Vocab]
        self.assertEqual(output.shape, (batch_size, seq_len, self.args.vocab_size))

    def test_loss_backward(self):
        # 验证反向传播是否通畅
        x = torch.randint(0, self.args.vocab_size, (2, 10))
        output = self.model(x)
        loss = output.sum()
        loss.backward()

        # 检查梯度是否存在
        self.assertIsNotNone(self.model.output.weight.grad)


if __name__ == '__main__':
    unittest.main()
