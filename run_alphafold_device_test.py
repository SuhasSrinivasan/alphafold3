# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Tests device selection and validation in the AlphaFold 3 launcher."""

import dataclasses
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import run_alphafold


@dataclasses.dataclass(frozen=True)
class _FakeDevice:
  platform: str
  compute_capability: str | None = None


class _MpsDeviceWithForbiddenComputeCapability:
  platform = 'mps'

  @property
  def compute_capability(self) -> str:
    raise AssertionError('MPS compute capability must not be inspected')


class DeviceSelectionTest(parameterized.TestCase):

  def test_default_accelerator_backend_remains_gpu(self):
    self.assertEqual(run_alphafold._JAX_BACKEND.default, 'gpu')

  def test_cpu_only_selects_first_cpu_and_ignores_accelerator_index(self):
    cpu = _FakeDevice('cpu')
    with mock.patch.object(
        run_alphafold.jax, 'local_devices', return_value=[cpu]
    ) as local_devices:
      actual = run_alphafold._select_inference_device(
          use_cpu_only=True,
          accelerator_backend='mps',
          accelerator_device_index=123,
      )

    self.assertIs(actual, cpu)
    local_devices.assert_called_once_with(backend='cpu')

  def test_gpu_backend_selects_gpu_without_querying_mps(self):
    cuda = _FakeDevice('gpu', '9.0')
    with mock.patch.object(
        run_alphafold.jax, 'local_devices', return_value=[cuda]
    ) as local_devices:
      actual = run_alphafold._select_inference_device(
          use_cpu_only=False,
          accelerator_backend='gpu',
          accelerator_device_index=0,
      )

    self.assertIs(actual, cuda)
    local_devices.assert_called_once_with(backend='gpu')

  def test_mps_backend_selects_mps_without_querying_gpu(self):
    mps_devices = [_FakeDevice('mps'), _FakeDevice('mps')]
    with mock.patch.object(
        run_alphafold.jax, 'local_devices', return_value=mps_devices
    ) as local_devices:
      actual = run_alphafold._select_inference_device(
          use_cpu_only=False,
          accelerator_backend='mps',
          accelerator_device_index=1,
      )

    self.assertIs(actual, mps_devices[1])
    local_devices.assert_called_once_with(backend='mps')

  @parameterized.parameters(
      "Unknown backend: 'mps'",
      'Unknown backend mps. Available platforms are: cpu',
  )
  def test_unknown_backend_has_actionable_error(self, backend_error):
    with mock.patch.object(
        run_alphafold.jax,
        'local_devices',
        side_effect=RuntimeError(backend_error),
    ) as local_devices:
      with self.assertRaisesRegex(RuntimeError, 'No local MPS device'):
        run_alphafold._select_inference_device(
            use_cpu_only=False,
            accelerator_backend='mps',
            accelerator_device_index=0,
        )

    local_devices.assert_called_once_with(backend='mps')

  @parameterized.parameters(-1, 2)
  def test_accelerator_index_must_be_in_range(self, device_index):
    with mock.patch.object(
        run_alphafold.jax,
        'local_devices',
        return_value=[_FakeDevice('mps'), _FakeDevice('mps')],
    ):
      with self.assertRaisesRegex(ValueError, 'out of range'):
        run_alphafold._select_inference_device(
            use_cpu_only=False,
            accelerator_backend='mps',
            accelerator_device_index=device_index,
        )

  def test_no_device_for_selected_backend_has_actionable_error(self):
    with mock.patch.object(
        run_alphafold.jax, 'local_devices', return_value=[]
    ) as local_devices:
      with self.assertRaisesRegex(RuntimeError, 'No local MPS device'):
        run_alphafold._select_inference_device(
            use_cpu_only=False,
            accelerator_backend='mps',
            accelerator_device_index=0,
        )

    local_devices.assert_called_once_with(backend='mps')

  def test_installed_backend_initialization_error_propagates(self):
    error = RuntimeError('MPS backend failed to initialize')
    with mock.patch.object(
        run_alphafold.jax, 'local_devices', side_effect=error
    ) as local_devices:
      with self.assertRaisesRegex(RuntimeError, 'failed to initialize'):
        run_alphafold._select_inference_device(
            use_cpu_only=False,
            accelerator_backend='mps',
            accelerator_device_index=0,
        )

    local_devices.assert_called_once_with(backend='mps')


class DeviceValidationTest(parameterized.TestCase):

  @parameterized.product(
      platform=('cpu', 'mps'), implementation=('xla', 'xla_chunked')
  )
  def test_portable_attention_is_accepted(self, platform, implementation):
    run_alphafold._validate_inference_device(
        _FakeDevice(platform), implementation
    )

  @parameterized.product(
      platform=('cpu', 'mps'), implementation=('triton', 'cudnn')
  )
  def test_nonportable_attention_is_rejected(self, platform, implementation):
    with self.assertRaisesRegex(
        ValueError, f'For {platform.upper()} inference'
    ):
      run_alphafold._validate_inference_device(
          _FakeDevice(platform), implementation
      )

  def test_mps_does_not_inspect_cuda_compute_capability(self):
    run_alphafold._validate_inference_device(
        _MpsDeviceWithForbiddenComputeCapability(), 'xla'
    )

  def test_old_cuda_compute_capability_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'compute capability 6.0'):
      run_alphafold._validate_inference_device(
          _FakeDevice('cuda', '5.2'), 'xla'
      )

  def test_cuda_7_requires_workaround_flag(self):
    with mock.patch.dict(os.environ, {'XLA_FLAGS': ''}):
      with self.assertRaisesRegex(ValueError, 'XLA_FLAGS'):
        run_alphafold._validate_inference_device(
            _FakeDevice('cuda', '7.5'), 'xla'
        )

  @parameterized.parameters('xla', 'xla_chunked')
  def test_cuda_7_accepts_portable_attention(self, implementation):
    with mock.patch.dict(
        os.environ,
        {
            'XLA_FLAGS': (
                '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'
            )
        },
    ):
      run_alphafold._validate_inference_device(
          _FakeDevice('cuda', '7.5'), implementation
      )

  def test_cuda_7_rejects_nonportable_attention(self):
    with mock.patch.dict(
        os.environ,
        {
            'XLA_FLAGS': (
                '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'
            )
        },
    ):
      with self.assertRaisesRegex(ValueError, 'xla_chunked'):
        run_alphafold._validate_inference_device(
            _FakeDevice('cuda', '7.5'), 'triton'
        )

  def test_modern_cuda_preserves_existing_behavior(self):
    run_alphafold._validate_inference_device(
        _FakeDevice('cuda', '9.0'), 'triton'
    )

  def test_legacy_gpu_platform_with_compute_capability_is_cuda(self):
    run_alphafold._validate_inference_device(
        _FakeDevice('gpu', '9.0'), 'triton'
    )

  @parameterized.parameters('gpu', 'rocm', 'tpu')
  def test_unvalidated_accelerator_is_rejected(self, platform):
    with self.assertRaisesRegex(ValueError, 'Unsupported accelerator platform'):
      run_alphafold._validate_inference_device(
          _FakeDevice(platform), 'xla'
      )

  def test_rocm_is_rejected_even_if_it_exposes_compute_capability(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported accelerator platform'):
      run_alphafold._validate_inference_device(
          _FakeDevice('rocm', '9.0'), 'xla'
      )

  def test_xla_chunked_is_carried_through_model_config(self):
    config = run_alphafold.make_model_config(
        flash_attention_implementation='xla_chunked'
    )
    self.assertEqual(
        config.global_config.flash_attention_implementation, 'xla_chunked'
    )


if __name__ == '__main__':
  absltest.main()
