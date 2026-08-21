# FlashAttention for GCU

This repository contains a GCU-oriented fork of FlashAttention for PyTorch.
It keeps the familiar `flash_attn` Python API shape while providing a GCU
backend, GCU-specific tests, and GCU-focused packaging/build scripts.

## Overview

This branch is intended for environments that already provide the GCU runtime
stack required by the extension build and execution path.

Exposed Python APIs include:
- `flash_attn_func`
- `flash_attn_qkvpacked_func`
- `flash_attn_kvpacked_func`
- `flash_attn_varlen_func`
- `flash_attn_varlen_qkvpacked_func`
- `flash_attn_varlen_kvpacked_func`
- `flash_attn_with_kvcache`

The implementation and tests in this branch target the GCU device path rather
than the upstream generic release flow.

## Requirements

- Linux
- PyTorch
- `torch_gcu`
- A GCU software stack that provides the required runtime headers and libraries
- A C++17-capable compiler toolchain

## Installation

### Option 1: Docker (Recommended)

The easiest way to get started is using the pre-built Docker image with all
dependencies included.

1. **Pull and start the container**:

   ```bash
   IMAGE=registry-egc.enflame-tech.com/artifacts/public_pytorch:v2.11.0-TR3.8.106-ubuntu2204

   docker run --name flash_attn -d \
     -v /home:/home \
     --shm-size 8G \
     --ipc=host --network host \
     --cap-add SYS_PTRACE \
     --security-opt seccomp=unconfined \
     --privileged \
     "$IMAGE" \
     tail -f /dev/null
   ```
2. **Update the host GCU driver** (to match the image's software version):

   ```bash
   # Extract the matching driver from the container
   docker cp flash_attn:/enflame/driver ./

   # Install the driver on the host
   sudo driver/enflame-x86_64-gcc-*.run

   # Restart the container to pick up the new driver
   docker restart flash_attn
   ```
3. **Clone the source code**:

   ```bash
   cd /home
   git clone git@github.com:EnflameTechnology/flash-attention.git
   ```
4. **Enter the container and build**:

   ```bash
   docker exec -it flash_attn bash
   cd /home/flash-attention
   ./install.sh
   ```

### Option 2: build and install with the helper script

Build from source in an environment where the GCU runtime stack is already
installed.

```sh
./install.sh
```

### Option 3: build a wheel manually

```sh
python setup.py bdist_wheel
pip install --force-reinstall --no-deps dist/*.whl
```

### Optional build environment variables

The build script reads the following environment variables when present:

- `BASE_DIR`
- `COMPILE_CXX_FLAGS`
- `LINK_FLAGS`
- `LINK_LIBS`
- `CMAKE_BUILD_TYPE`
- `PACKAGE_VERSION`

Use these only when your local toolchain/runtime layout requires overrides.

## Quick Start

The following snippets mirror the APIs exercised by the test suite under
`tests/` (`flash_attn_func`, `flash_attn_varlen_func`).

### Standard attention (`flash_attn_func`)

```python
import torch
import torch_gcu
from torch_gcu import transfer_to_gcu  # monkeypatches cuda -> gcu

from flash_attn import flash_attn_func

batch_size, seqlen = 2, 128
num_heads, head_dim = 8, 64

q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float16)
k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float16)
v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float16)

out = flash_attn_func(q.gcu(), k.gcu(), v.gcu(), dropout_p=0.0, causal=False)
print(out.shape)  # torch.Size([2, 128, 8, 64])
```

GQA (grouped-query attention) is supported by passing different head counts for
`k` / `v`:

```python
num_kv_heads = 2
q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.float16)
k = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, dtype=torch.float16)
v = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, dtype=torch.float16)

out = flash_attn_func(q.gcu(), k.gcu(), v.gcu(), dropout_p=0.0, causal=True)
```

### Variable-length attention (`flash_attn_varlen_func`)

```python
import torch
import torch_gcu
from torch_gcu import transfer_to_gcu

from flash_attn import flash_attn_varlen_func

batch_size, seqlen = 2, 128
num_heads, head_dim = 8, 64
total = batch_size * seqlen

q = torch.randn(total, num_heads, head_dim, dtype=torch.float16)
k = torch.randn(total, num_heads, head_dim, dtype=torch.float16)
v = torch.randn(total, num_heads, head_dim, dtype=torch.float16)

cu_seqlens = torch.arange(
    0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32
).gcu()

out = flash_attn_varlen_func(
    q.gcu(), k.gcu(), v.gcu(), cu_seqlens, cu_seqlens, seqlen, seqlen,
    causal=True, window_size=(-1, -1), softcap=0.0,
)
```

For paged KV-cache / varlen inference, see
`tests/test_flash_attn_vllm_gcu.py::test_varlen_with_paged_kv`, which uses
`flash_attn.vllm_flash_attn.flash_attn_varlen_func` with `block_table` and
`seqused_k`.

## Testing

The test suite under `tests/` is intentionally minimal and focused on the GCU
device path. Run it from the repository root:

```sh
# Standard and GQA forward
pytest -q tests/test_flash_attn_s60.py

# Variable-length forward
pytest -q tests/test_flash_attn_varlen_s60.py

# Paged KV-cache varlen (vLLM path)
pytest -q tests/test_flash_attn_vllm_gcu.py::test_varlen_with_paged_kv
```

Or run all of them together:

```sh
pytest -q tests/
```

## Project Structure

- `flash_attn/` — Python package and public API wrappers
- `csrc_gcu/` — GCU C++ extension sources
- `tests/` — GCU test coverage (`flash_attn_func`, `flash_attn_varlen_func`, and the vLLM paged-KV varlen path)

## Notes

- This branch is a GCU-targeted fork and is not documented as a generic
  accelerator release.
- Public dependency names such as `torch_gcu` and `topsaten` are kept as-is in
  code paths where they are part of the runtime/build contract.
- For concrete usage patterns, see the tests in `tests/`.

## License

This repository is released under the terms of the BSD 3-Clause License. See
[`LICENSE`](LICENSE).

## Attribution

This repository is based on the FlashAttention project and preserves upstream
attribution present in source files and repository metadata.
