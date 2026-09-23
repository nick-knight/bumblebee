import tempfile
import unittest
from pathlib import Path

import torch

from checkpoint import load_checkpoint, save_checkpoint
from dataloader import ByteTokenizer, text_dataloader
from model import GPTModel, ModelConfig
from optimizer import OptimConfig, Optimizer
from train import TrainConfig


def tiny_model_config():
    return ModelConfig(
        num_layers=2,
        hidden_dim=16,
        num_heads=4,
        num_query_groups=2,
        qk_dim=4,
        v_dim=3,
        mlp_dim=32,
        max_seq_len=8,
        vocab_size=32,
    )


class RecordingTransform(torch.nn.Module):

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.last_shape = None

    def forward(self, values):
        self.last_shape = values.shape
        return self.scale * values


class CharacterTokenizer:

    def encode_ordinary(self, text):
        return [ord(character) for character in text]


class ModelOptSupportTest(unittest.TestCase):

    def test_byte_tokenizer_uses_a_compact_fixed_vocabulary(self):
        tokenizer = ByteTokenizer()

        self.assertEqual(tokenizer.n_vocab, 256)
        self.assertEqual(tokenizer.encode_ordinary('Aé'), list('Aé'.encode('utf-8')))

    def test_local_text_dataloader_produces_deterministic_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'corpus.txt'
            path.write_text('abcdefghijklmnopqrstuvwxyz' * 10, encoding='utf-8')
            first = text_dataloader(path, CharacterTokenizer(), 2, 8, 'cpu', 3, 'train')
            second = text_dataloader(path, CharacterTokenizer(), 2, 8, 'cpu', 3, 'train')

            torch.testing.assert_close(next(first), next(second))

    def test_key_value_transforms_receive_per_group_activations(self):
        torch.manual_seed(1)
        model = GPTModel(tiny_model_config())
        inputs = torch.randint(0, model.vocab_size, (2, 5))
        baseline = model(inputs)
        key_transform = RecordingTransform(0.0)
        value_transform = RecordingTransform(0.0)

        model.set_key_value_transforms(0, key=key_transform, value=value_transform)
        transformed = model(inputs)

        self.assertEqual(key_transform.last_shape, (2, 2, 5, 4))
        self.assertEqual(value_transform.last_shape, (2, 2, 5, 3))
        self.assertFalse(torch.equal(baseline, transformed))

    def test_checkpoint_round_trip_restores_model_and_optimizer(self):
        torch.manual_seed(2)
        model_config = tiny_model_config()
        train_config = TrainConfig(dataset_name='synthetic', torch_compile=False)
        optim_config = OptimConfig()
        model = GPTModel(model_config)
        optimizer = Optimizer(optim_config, model)
        for state in optimizer.adamw_m.values():
            state.normal_()
        expected_model = {name: value.clone() for name, value in model.state_dict().items()}
        expected_m = {name: value.clone() for name, value in optimizer.adamw_m.items()}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                next_step=17,
                model_config=model_config,
                train_config=train_config,
                optim_config=optim_config,
            )
            restored_model = GPTModel(model_config)
            restored_optimizer = Optimizer(optim_config, restored_model)
            checkpoint = load_checkpoint(
                path,
                model=restored_model,
                optimizer=restored_optimizer,
                map_location='cpu',
            )

        self.assertEqual(checkpoint['next_step'], 17)
        self.assertEqual(checkpoint['model_config']['hidden_dim'], 16)
        for name, value in restored_model.state_dict().items():
            torch.testing.assert_close(value, expected_model[name])
        for name, value in restored_optimizer.adamw_m.items():
            torch.testing.assert_close(value, expected_m[name])


if __name__ == '__main__':
    unittest.main()
