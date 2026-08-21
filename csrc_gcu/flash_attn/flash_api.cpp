/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all
// of the torch headers. #include <torch/python.h> #include
// <torch/nn/functional.h>
#include <ATen.h>
#include <c10/util/Logging.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include <fstream>

#include "torch_gcu.h"
// #include "flash.h"
// #include "static_switch.h"

#include "gcu/topsaten/topsaten_fa.h"
#include "gcu/topsaten/topsaten_ops.h"
#include "gcu/topsaten/topsaten_te.h"

#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_DEVICE(x) TORCH_CHECK(x.is_gcu(), #x " must be on gcu")
#define TOPSATEN_CHECK(call)                                         \
  {                                                                  \
    const topsatenStatus_t error = call;                             \
    if (error != TOPSATEN_STATUS_SUCCESS)                            \
    {                                                                \
      printf("Error: %s:%d, error=%d\n", __FILE__, __LINE__, error); \
      exit(1);                                                       \
    }                                                                \
  }
typedef struct alignas(2)
{
  uint16_t x;
} __bfloat16_raw;

class alignas(2) bfloat16
{
public:
  bfloat16() = default;

  struct from_bits_t
  {
  };
  static constexpr from_bits_t from_bits() { return from_bits_t(); }
  constexpr bfloat16(uint16_t bits, from_bits_t) : __x(bits) {}

  /* Convert to/from __bfloat16_raw */
  bfloat16(const __bfloat16_raw &hr) : __x(hr.x) {}
  bfloat16 &operator=(const __bfloat16_raw &hr)
  {
    __x = hr.x;
    return *this;
  }
  volatile bfloat16 &operator=(const __bfloat16_raw &hr) volatile
  {
    __x = hr.x;
    return *this;
  }
  volatile bfloat16 &operator=(const volatile __bfloat16_raw &hr) volatile
  {
    __x = hr.x;
    return *this;
  }
  operator __bfloat16_raw() const
  {
    __bfloat16_raw ret;
    ret.x = __x;
    return ret;
  }
  operator __bfloat16_raw() const volatile
  {
    __bfloat16_raw ret;
    ret.x = __x;
    return ret;
  }

  /* Construct from float/double */
  bfloat16(const float f) { __x = __float2bfloat16(f).__x; }
  bfloat16(const double f) { __x = __double2bfloat16(f).__x; }
  bfloat16(const int f) { __x = __float2bfloat16(f).__x; }
  bfloat16(const int64_t f) { __x = __float2bfloat16(f).__x; }
  bfloat16(const uint32_t f)
  {
    __x = __float2bfloat16(static_cast<float>(f)).__x;
  }
  bfloat16(const uint64_t f)
  {
    __x = __float2bfloat16(static_cast<float>(f)).__x;
  }
  bfloat16(const bool f) { __x = __float2bfloat16(f).__x; }

  operator float() const { return __bfloat162float(*this); }
  bfloat16 &operator=(const float f)
  {
    __x = __float2bfloat16(f).__x;
    return *this;
  }

  /* We omit "cast to double" operator, so as to not be ambiguous about up-cast
   */
  bfloat16 &operator=(const double f)
  {
    __x = __double2bfloat16(f).__x;
    return *this;
  }

  bfloat16 &operator=(const bool f)
  {
    __x = __float2bfloat16(f).__x;
    return *this;
  }

  bfloat16 &operator+=(const float f)
  {
    __x = __float2bfloat16(__bfloat162float(*this) + f).__x;
    return *this;
  }
  bfloat16 &operator-=(const float f)
  {
    __x = __float2bfloat16(__bfloat162float(*this) - f).__x;
    return *this;
  }
  bfloat16 &operator*=(const float f)
  {
    __x = __float2bfloat16(__bfloat162float(*this) * f).__x;
    return *this;
  }
  bfloat16 &operator/=(const float f)
  {
    __x = __float2bfloat16(__bfloat162float(*this) / f).__x;
    return *this;
  }

  // protected:
  static uint16_t __internal_float2bfloat16(const float f, uint32_t &sign,
                                            uint32_t &remainder)
  {
    uint32_t x;
    (void)std::memcpy(&x, &f, sizeof(f));
    if ((x & 0x7fffffffU) > 0x7f800000U)
    {
      sign = 0U;
      remainder = 0U;
      return static_cast<uint16_t>(0x7fffU);
    }
    sign = x >> 31U;
    remainder = x << 16U;
    return static_cast<uint16_t>(x >> 16U);
  }

  static bfloat16 __double2bfloat16(const double x)
  {
    float f = static_cast<float>(x);
    const double d = static_cast<double>(f);
    uint32_t u;
    (void)std::memcpy(&u, &f, sizeof(f));
    bool x_is_not_nan = ((u << (unsigned)1U) <= (unsigned)0xFF000000U);

    if ((x > 0.0) && (d > x))
    {
      u--;
    }
    if ((x < 0.0) && (d < x))
    {
      u--;
    }
    if ((d != x) && x_is_not_nan)
    {
      u |= 1U;
    }

    (void)std::memcpy(&f, &u, sizeof(f));
    return __float2bfloat16(f);
  }

  static bfloat16 __float2bfloat16(const float a)
  {
    bfloat16 val;
    __bfloat16_raw r;
    uint32_t sign = 0U;
    uint32_t remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    if ((remainder > 0x80000000U) ||
        ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U)))
    {
      r.x++;
    }
    val = r;
    return val;
  }

  static bfloat16 __float2bfloat16_rz(const float a)
  {
    bfloat16 val;
    __bfloat16_raw r;
    uint32_t sign = 0U;
    uint32_t remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    val = r;
    return val;
  }

  static bfloat16 __float2bfloat16_rd(const float a)
  {
    bfloat16 val;
    __bfloat16_raw r;
    uint32_t sign = 0U;
    uint32_t remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    if ((remainder != 0U) && (sign != 0U))
    {
      r.x++;
    }
    val = r;
    return val;
  }

  static bfloat16 __float2bfloat16_ru(const float a)
  {
    bfloat16 val;
    __bfloat16_raw r;
    uint32_t sign = 0U;
    uint32_t remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    if ((remainder != 0U) && (sign == 0U))
    {
      r.x++;
    }
    val = r;
    return val;
  }

  static float __internal_bfloat162float(const uint16_t h)
  {
    float f;
    uint32_t u = static_cast<uint32_t>(h) << 16;
    (void)std::memcpy(&f, &u, sizeof(f));
    return f;
  }

  static float __bfloat162float(const bfloat16 a)
  {
    return __internal_bfloat162float(static_cast<__bfloat16_raw>(a).x);
  }

  static bfloat16 __float2bfloat16_rn(const float a)
  {
    bfloat16 val;
    __bfloat16_raw r;
    uint32_t sign = 0U;
    uint32_t remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    if ((remainder > 0x80000000U) ||
        ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U)))
    {
      r.x++;
    }
    val = r;
    return val;
  }

protected:
  uint16_t __x;
};

void dump_tensor(const char *name, size_t size, void *data)
{
  std::string name_str(name);
  std::ofstream fout(name_str, std::ios::binary);
  std::cout << "name: " << name << std::endl;
  fout.write(reinterpret_cast<char *>(data), size);
  fout.close();
  return;
}

void bf16_to_fp32(float *dst, __bfloat16_raw *src, int size)
{
  for (int i = 0; i < size; i++)
  {
    dst[i] = bfloat16::__bfloat162float(src[i]);
  }
}

topsatenTensor makeTopsatenTensor(const at::Tensor &tensor,
                                  const topsatenDataType_t &ddtype)
{
  TORCH_INTERNAL_ASSERT(tensor.device().is_privateuseone(),
                        "only gcu tensor could create topsatenTensor.");

  auto dtype = tensor.scalar_type();
  // auto topsaten_dtype = topsatenDataType_t::TOPSATEN_DATA_BF16;

  auto itemsize = tensor.itemsize();
  if (dtype == c10::ScalarType::Double || dtype == c10::ScalarType::Long)
  {
    itemsize = itemsize >> 1;
  }
  void *data_ptr = static_cast<char *>(
                       tensor.unsafeGetTensorImpl()->storage().mutable_data()) +
                   itemsize * tensor.storage_offset();

  auto rank = tensor.dim() == 0 ? 1 : tensor.dim();

  auto dims = tensor.sizes().vec();
  auto topsaten_dims = topsatenSize_t{dims.data(), rank};

  auto strides = tensor.strides().vec();
  auto topsaten_strides = topsatenSize_t{strides.data(), rank};

  return topsatenTensor(topsaten_dims, topsaten_strides, ddtype, data_ptr);
}

topsatenTensor makeOptionalTopsatenTensor(
    const c10::optional<at::Tensor> &tensor,
    const topsatenDataType_t &ddtype)
{
  if (tensor.has_value())
  {
    return makeTopsatenTensor(tensor.value(), ddtype);
  }
  else
  {
    return topsatenTensor();
  }
}

topsatenTensor makeOptionalConstTopsatenTensor(
    const c10::optional<const at::Tensor> &tensor,
    const topsatenDataType_t &ddtype)
{
  if (tensor.has_value())
  {
    return makeTopsatenTensor(tensor.value(), ddtype);
  }
  else
  {
    return topsatenTensor();
  }
}

topsatenTensor makeOptionalConstTopsatenTensor(
    const c10::optional<at::Tensor> &tensor,
    const topsatenDataType_t &ddtype)
{
  if (tensor.has_value())
  {
    return makeTopsatenTensor(tensor.value(), ddtype);
  }
  else
  {
    return topsatenTensor();
  }
}

inline topsatenDataType_t get_gcu_dtype(const at::Tensor &t)
{
  auto st = t.scalar_type();
  switch (st)
  {
  case at::kHalf:
    return topsatenDataType_t::TOPSATEN_DATA_FP16;
  case at::kFloat:
    return topsatenDataType_t::TOPSATEN_DATA_FP32;
  case at::kBFloat16:
    return topsatenDataType_t::TOPSATEN_DATA_BF16;
  case at::kBool:
    return topsatenDataType_t::TOPSATEN_DATA_U8;
  case torch::kByte:
    return topsatenDataType_t::TOPSATEN_DATA_U8;
  case torch::kInt32:
    return topsatenDataType_t::TOPSATEN_DATA_I32;
  case torch::kUInt32:
    return topsatenDataType_t::TOPSATEN_DATA_U32;
  case torch::kInt64:
    return topsatenDataType_t::TOPSATEN_DATA_I64;
  case torch::kFloat8_e4m3fn:
    return topsatenDataType_t::TOPSATEN_DATA_FP8E4M3;
  case torch::kFloat8_e5m2:
    return topsatenDataType_t::TOPSATEN_DATA_FP8E5M2;
  case torch::kInt8:
    return topsatenDataType_t::TOPSATEN_DATA_I8;
  default:
    std::cerr << "Unsupported dtype:" << t;
    std::abort();
  }
}

std::vector<at::Tensor> mha_fwd(
    at::Tensor
        &q,                          // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &k,             // batch_size x seqlen_k x num_heads_k x
                                     // round_multiple(head_size, 8)
    const at::Tensor &v,             // batch_size x seqlen_k x num_heads_k x
                                     // round_multiple(head_size, 8)
    c10::optional<at::Tensor> &out_, // batch_size x seqlen_q x num_heads x
                                     // round_multiple(head_size, 8)
    c10::optional<at::Tensor>
        &alibi_slopes_, // num_heads or batch_size x num_heads
    const float p_dropout, const float softmax_scale, bool is_causal,
    int window_size_left, int window_size_right, const float softcap,
    const bool return_softmax, c10::optional<at::Generator> gen_)
{
  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention GCU only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

  // CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  const int head_size = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);
  const int head_size_v = v.size(-1);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(
      head_size <= 256,
      "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(head_size % 8 == 0,
              "query, key, value, and out_ must have a head_size that is a "
              "multiple of 8");
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");

  if (softcap > 0.f)
  {
    TORCH_CHECK(p_dropout == 0.f,
                "Softcapping does not support dropout for now");
  }

  if (window_size_left >= seqlen_k)
  {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_k)
  {
    window_size_right = -1;
  }

  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1 && !alibi_slopes_.has_value())
  {
    is_causal = false;
  }
  if (is_causal)
  {
    window_size_right = 0;
  }

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_v);

  // auto opts = q.options();
  auto opts = q.options().dtype(at::kFloat).device(at::kPrivateUse1);
  at::Tensor out;
  if (out_.has_value())
  {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype,
                "Output must have the same dtype as inputs");
    // CHECK_DEVICE(out);
    TORCH_CHECK(out.stride(-1) == 1,
                "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size_v);
  }
  else
  {
    // out = torch::empty_like(q);
    out = torch::empty({batch_size, sizes[1], sizes[2], head_size_v},
                       opts.dtype(q_dtype));
  }

  auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts);
  at::Tensor p;
  // Only return softmax if there's dropout to reduce compilation time
  if (return_softmax)
  {
    TORCH_CHECK(p_dropout > 0.0f,
                "return_softmax is only supported when p_dropout > 0.0");
    p = torch::empty({batch_size, num_heads, seqlen_q, seqlen_k}, opts);
  }
  else
  {
    p = torch::empty({0}, opts);
  }

  // auto options =
  //   torch::TensorOptions().dtype(torch::kFloat32).device(torch::kGCU);
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(at::kPrivateUse1);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));

  topsatenDataType_t q_data_type = q_dtype == torch::kBFloat16 ? topsatenDataType_t::TOPSATEN_DATA_BF16
                                                               : topsatenDataType_t::TOPSATEN_DATA_FP16;

  TORCH_CHECK(q.numel() > 0, "q is null tensor");
  topsatenTensor q_tensor = makeTopsatenTensor(q, q_data_type);

  TORCH_CHECK(k.numel() > 0, "k is null tensor");
  topsatenTensor k_tensor = makeTopsatenTensor(k, q_data_type);

  TORCH_CHECK(v.numel() > 0, "v is null tensor");
  topsatenTensor v_tensor = makeTopsatenTensor(v, q_data_type);

  topsatenTensor alibi_slopes_tensor = makeOptionalTopsatenTensor(
      alibi_slopes_, topsatenDataType_t::TOPSATEN_DATA_FP32);

  topsatenSize_t tensor_dims;
  topsatenSize_t tensor_strides;

  TORCH_CHECK(rng_state.numel() > 0, "rng_state is null tensor");
  // construct rng state params
  int64_t dims_rng_state[] = {2};
  int64_t strides_rng_state[] = {1};
  tensor_dims.data = dims_rng_state;
  tensor_dims.len = 1;
  tensor_strides.data = strides_rng_state;
  tensor_strides.len = 1;
  topsatenTensor rng_state_tensor(tensor_dims, tensor_strides,
                                  topsatenDataType_t::TOPSATEN_DATA_U64,
                                  rng_state.data_ptr());

  // construct out tensor params
  auto out_tensor = makeTopsatenTensor(out, q_data_type);

  // construct softmax_lse tensor params
  topsatenSize_t softmax_lse_tensor_dims, softmax_lse_tensor_strides;
  int64_t dims_softmax_lse[3] = {batch_size, num_heads, seqlen_q};
  int64_t stride_softmax_lse[3] = {num_heads * seqlen_q, seqlen_q, 1};
  topsatenDataType_t softmax_lse_data_type =
      topsatenDataType_t::TOPSATEN_DATA_FP32;
  softmax_lse_tensor_dims.data = dims_softmax_lse;
  softmax_lse_tensor_dims.len = 3;
  softmax_lse_tensor_strides.data = stride_softmax_lse;
  softmax_lse_tensor_strides.len = 3;

  topsatenTensor softmax_lse_tensor(
      softmax_lse_tensor_dims, softmax_lse_tensor_strides,
      softmax_lse_data_type, softmax_lse.data_ptr());

  topsatenTensor S_dmask_tensor;

  std::tuple<topsatenTensor &, topsatenTensor &, topsatenTensor &,
             topsatenTensor &>
      outputs(out_tensor, softmax_lse_tensor, S_dmask_tensor, rng_state_tensor);

  // scale value
  topsatenScalar_t p_dropout_scalar;
  p_dropout_scalar.dtype = TOPSATEN_DATA_FP32;
  p_dropout_scalar.fval = p_dropout;

  topsatenScalar_t softmax_scale_scalar;
  softmax_scale_scalar.dtype = TOPSATEN_DATA_FP32;
  softmax_scale_scalar.fval = softmax_scale;

  topsatenScalar_t window_size_left_scalar;
  window_size_left_scalar.dtype = TOPSATEN_DATA_I32;
  window_size_left_scalar.ival = window_size_left;

  topsatenScalar_t window_size_right_scalar;
  window_size_right_scalar.dtype = TOPSATEN_DATA_I32;
  window_size_right_scalar.ival = window_size_right;

  topsatenScalar_t softcap_scalar;
  softcap_scalar.dtype = TOPSATEN_DATA_FP32;
  softcap_scalar.fval = softcap;

  // fixme is the same generator with the PhiloxState
  auto torch_generator =
      at::get_generator_or_default<torch_gcu::GCUGeneratorImpl>(
          gen_, torch_gcu::getDefaultGCUGenerator(q.device().index()));

  bool is_dropout = p_dropout > 0.0;
  uint64_t drop_seed = is_dropout > 0.0 ? torch_generator->current_seed() : 0;
  uint64_t drop_offset = is_dropout > 0.0 ? torch_generator->get_offset() : 0;

  // if (p_dropout > 0.0)  {
  // int64_t counter_offset = params.b * params.h * 32;
  // See Note [Acquire lock when using random generators]
  // std::lock_guard<std::mutex> lock(torch_generator->mutex_);
  // at::PhiloxCudaState philox_args = torch_generator->philox_gcu_state(counter_offset);
  // std::tie(drop_seed, drop_offset) = flash::unpack(philox_args);
  // todo using different way if (arg.captured_)
  // topsatenPhiloxState_t philox_state;
  // philox_state.seed.val = torch_generator->current_seed();
  // philox_state.offset.val = torch_generator->get_offset();
  // }

  topsatenGenerator_t generator{drop_seed, drop_offset};

  if (seqlen_k > 0)
  {
    auto stream = torch_gcu::getCurrentGCUStream();
    TOPSATEN_CHECK(topsfa::topsfaFlashAttnFwd(
        outputs, q_tensor, k_tensor, v_tensor, out_tensor, alibi_slopes_tensor,
        p_dropout_scalar, softmax_scale_scalar, is_causal,
        window_size_left_scalar, window_size_right_scalar, softcap_scalar,
        return_softmax, generator, stream));
    // stream.synchronize();
  }
  else
  {
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output
    // to 0.
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
  }
  return {out, softmax_lse, p, rng_state};
}

std::vector<at::Tensor> mha_bwd(
    const at::Tensor
        &dout,                     // batch_size x seqlen_q x num_heads, x head_size_og
    const at::Tensor &q,           // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &k,           // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor &v,           // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor &out,         // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &softmax_lse, // b x h x seqlen_q
    c10::optional<at::Tensor>
        &dq_, // batch_size x seqlen_q x num_heads x head_size
    c10::optional<at::Tensor>
        &dk_, // batch_size x seqlen_k x num_heads_k x head_size
    c10::optional<at::Tensor>
        &dv_, // batch_size x seqlen_k x num_heads_k x head_size
    __attribute__((unused)) c10::optional<at::Tensor>
        &alibi_slopes_,    // num_heads or batch_size x num_heads
    const float p_dropout, // probability to drop
    const float softmax_scale, const bool is_causal, int window_size_left,
    int window_size_right, const float softcap,
    __attribute__((unused)) const bool deterministic,
    c10::optional<at::Generator> gen_,
    c10::optional<at::Tensor> &rng_state)
{
#ifdef FLASHATTENTION_DISABLE_BACKWARD
  TORCH_CHECK(false, "This flash attention build does not support backward.");
#endif
  if (is_causal)
  {
    window_size_right = 0;
  }

  // auto stream = torch_gcu::getCurrentGCUStream();

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention GCU only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
  TORCH_CHECK(dout.dtype() == q_dtype,
              "query and dout must have the same dtype");

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(out.stride(-1) == 1,
              "out tensor must have contiguous last dimension");
  TORCH_CHECK(dout.stride(-1) == 1,
              "dout tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  const int seqlen_q = sizes[1];
  const int num_heads = sizes[2];
  const int head_size_og = dout.size(3);
  const int head_size = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);
  const int head_size_v = v.size(-1);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(head_size_v % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(
      head_size <= 256,
      "FlashAttention backward only supports head dimension at most 256");
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");

  auto round_multiple = [](int x, int m)
  { return (x + m - 1) / m * m; };
  const int head_size_rounded =
      head_size <= 192 ? round_multiple(head_size, 32) : 256;
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  // const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
  TORCH_CHECK(head_size_v == round_multiple(head_size_og, 8),
              "head_size must be head_size_og rounded to a multiple of 8");
  if (softcap > 0.f)
  {
    TORCH_CHECK(p_dropout == 0.f,
                "Softcapping does not support dropout for now");
  }

  if (window_size_left >= seqlen_k)
  {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_k)
  {
    window_size_right = -1;
  }

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_v);
  CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_v);
  CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_og);

  at::Tensor dq, dk, dv;
  if (dq_.has_value())
  {
    dq = dq_.value();
    TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
    TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
    CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
  }
  else
  {
    dq = torch::empty_like(q);
  }
  if (dk_.has_value())
  {
    dk = dk_.value();
    TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
    TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
  }
  else
  {
    dk = torch::empty_like(k);
  }
  if (dv_.has_value())
  {
    dv = dv_.value();
    TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
    TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
    CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size_v);
  }
  else
  {
    dv = torch::empty_like(v);
  }

  at::Tensor dout_padded;
  if (head_size_og % 8 != 0)
  {
    dout_padded = torch::nn::functional::pad(
        dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
  }
  else
  {
    dout_padded = dout;
  }

  // bool loop = seqlen_k > blocksize_c;
  // TODO: change later, for now set to true for simplicity
  // bool loop = true;
  auto opts = q.options();
  auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded},
                                opts.dtype(at::kFloat));
  at::Tensor dq_accum;
  at::Tensor dk_accum, dv_accum;
  // if (loop) {
  //     if (!deterministic) {
  //         dq_accum = torch::empty({batch_size, seqlen_q_rounded, num_heads,
  //         head_size_rounded}, opts.dtype(at::kFloat));
  //     } else {
  //         const int nsplits = (dprops->multiProcessorCount + batch_size *
  //         num_heads - 1) / (batch_size * num_heads); dq_accum =
  //         torch::zeros({nsplits, batch_size, seqlen_q_rounded, num_heads,
  //         head_size_rounded}, opts.dtype(at::kFloat));
  //     }
  //     // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded,
  //     head_size_rounded}, opts.dtype(at::kFloat));
  //     // dv_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded,
  //     head_size_rounded}, opts.dtype(at::kFloat));
  // }
  dq_accum =
      torch::empty({batch_size, seqlen_q_rounded, num_heads, head_size_rounded},
                   opts.dtype(at::kFloat));
  // at::Tensor dk_expanded, dv_expanded;
  // if (num_heads_k != num_heads) {  // MQA / GQA
  //     dk_expanded = torch::empty({batch_size, seqlen_k, num_heads,
  //     head_size}, opts); dv_expanded = torch::empty({batch_size, seqlen_k,
  //     num_heads, head_size}, opts);
  // } else {
  //     dk_expanded = dk;
  //     dv_expanded = dv;
  // }

  // params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);

  // auto launch = &run_mha_bwd;

  // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
  //     gen_, at::cuda::detail::getDefaultCUDAGenerator());

  // We use a custom RNG that increases the offset by batch_size * nheads * 32.
  // int64_t counter_offset = params.b * params.h * 32;

  // set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

  // ---------------------start gcu tensor build--------------------------------
  topsatenDataType_t q_data_type = q_dtype == torch::kBFloat16 ? topsatenDataType_t::TOPSATEN_DATA_BF16
                                                               : topsatenDataType_t::TOPSATEN_DATA_FP16;

  TORCH_CHECK(q.numel() > 0, "q is null tensor");
  topsatenTensor q_tensor = makeTopsatenTensor(q, q_data_type);

  TORCH_CHECK(k.numel() > 0, "k is null tensor");
  topsatenTensor k_tensor = makeTopsatenTensor(k, q_data_type);

  TORCH_CHECK(v.numel() > 0, "v is null tensor");
  topsatenTensor v_tensor = makeTopsatenTensor(v, q_data_type);

  // construct out tensor params
  topsatenTensor out_tensor = makeTopsatenTensor(out, q_data_type);

  // construct dout tensor params
  topsatenTensor dout_tensor = makeTopsatenTensor(dout, q_data_type);

  // construct dq tensor params
  topsatenTensor dq_tensor = makeTopsatenTensor(dq, q_data_type);

  // construct dk tensor params
  topsatenTensor dk_tensor = makeTopsatenTensor(dk, q_data_type);

  // construct dv tensor params
  topsatenTensor dv_tensor = makeTopsatenTensor(dv, q_data_type);

  // construct softmax_lse tensor params
  topsatenSize_t softmax_lse_tensor_dims, softmax_lse_tensor_strides;
  int64_t dims_softmax_lse[3] = {batch_size, num_heads, seqlen_q};
  int64_t stride_softmax_lse[3] = {seqlen_q * num_heads, seqlen_q, 1};
  topsatenDataType_t softmax_lse_data_type =
      topsatenDataType_t::TOPSATEN_DATA_FP32;
  softmax_lse_tensor_dims.data = dims_softmax_lse;
  softmax_lse_tensor_dims.len = 3;
  softmax_lse_tensor_strides.data = stride_softmax_lse;
  softmax_lse_tensor_strides.len = 3;

  topsatenTensor softmax_lse_tensor(
      softmax_lse_tensor_dims, softmax_lse_tensor_strides,
      softmax_lse_data_type, softmax_lse.data_ptr());

  topsatenTensor alibi_slopes_tensor = makeOptionalTopsatenTensor(
      alibi_slopes_, topsatenDataType_t::TOPSATEN_DATA_FP32);

  topsatenTensor cum_seq_q_tensor;
  topsatenTensor cum_seq_k_tensor;

  // result tensor contruct
  std::tuple<topsatenTensor, topsatenTensor, topsatenTensor, topsatenTensor>
      result(dq_tensor, dk_tensor, dv_tensor, softmax_lse_tensor);

  // scale value
  topsatenScalar_t p_dropout_scalar;
  p_dropout_scalar.dtype = TOPSATEN_DATA_FP32;
  p_dropout_scalar.fval = p_dropout;

  topsatenScalar_t softmax_scale_scalar;
  softmax_scale_scalar.dtype = TOPSATEN_DATA_FP32;
  softmax_scale_scalar.fval = softmax_scale;

  topsatenScalar_t window_size_left_scalar;
  window_size_left_scalar.dtype = TOPSATEN_DATA_I32;
  window_size_left_scalar.ival = window_size_left;

  topsatenScalar_t window_size_right_scalar;
  window_size_right_scalar.dtype = TOPSATEN_DATA_I32;
  window_size_right_scalar.ival = window_size_right;

  topsatenScalar_t softcap_scalar;
  softcap_scalar.dtype = TOPSATEN_DATA_FP32;
  softcap_scalar.fval = softcap;

  // fixme is the same generator with the PhiloxState
  auto torch_generator =
      at::get_generator_or_default<torch_gcu::GCUGeneratorImpl>(
          gen_, torch_gcu::getDefaultGCUGenerator(q.device().index()));

  bool is_dropout = p_dropout > 0.0;
  uint64_t drop_seed = is_dropout > 0.0 ? torch_generator->current_seed() : 0;
  uint64_t drop_offset = is_dropout > 0.0 ? torch_generator->get_offset() : 0;

  topsatenGenerator_t generator{drop_seed, drop_offset};

  topsatenTensor rng_state_tensor = makeOptionalTopsatenTensor(
      rng_state, topsatenDataType_t::TOPSATEN_DATA_U64);

  // ---------------------end gcu tensor build--------------------------------
  if (seqlen_q > 0)
  {
    auto stream = torch_gcu::getCurrentGCUStream();
    TOPSATEN_CHECK(topsfa::topsfaFlashAttnBwd(
        result, dout_tensor, q_tensor, k_tensor, v_tensor, out_tensor,
        softmax_lse_tensor, dq_tensor, dk_tensor, dv_tensor,
        alibi_slopes_tensor, p_dropout_scalar, softmax_scale_scalar,
        is_causal, window_size_left_scalar, window_size_right_scalar,
        softcap_scalar, deterministic,
        generator, rng_state_tensor, stream));
  }
  else
  {
    // If seqlen_q == 0, then we have an empty tensor. We need to set the output
    // to 0.
    dk.zero_();
    dv.zero_();
    softmax_d.zero_();
  }
  // For MQA/GQA we need to sum dK and dV across the groups
  // if (num_heads_k != num_heads) {
  //     at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k,
  //     num_heads_k, num_heads / num_heads_k, head_size}), {3});
  //     at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k,
  //     num_heads_k, num_heads / num_heads_k, head_size}), {3});
  // }
  if (head_size_og % 8 != 0)
  {
    dq = dq.index(
        {"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    dk = dk.index(
        {"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    dv = dv.index(
        {"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
  }
  return {dq, dk, dv, softmax_d};
}

std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor
        &q,              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor &k, // total_k x num_heads_k x head_size, total_k :=
                         // \sum_{i=0}^{b} s_i or num_blocks x page_block_size
                         // x num_heads_k x head_size if there's a block_table.
    const at::Tensor &v, // total_k x num_heads_k x head_size, total_k :=
                         // \sum_{i=0}^{b} s_i or num_blocks x page_block_size
                         // x num_heads_k x head_size if there's a block_table.
    c10::optional<at::Tensor> &
        out_,                       // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &cu_seqlens_q, // b+1
    const at::Tensor &cu_seqlens_k, // b+1
    c10::optional<at::Tensor>
        &seqused_k,                              // b. If given, only this many elements of each batch
                                                 // element's keys are used.
    c10::optional<const at::Tensor> &leftpad_k_, // batch_size
    c10::optional<at::Tensor>
        &block_table_,                        // batch_size x max_num_blocks_per_seq
    c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
    int max_seqlen_q, const int max_seqlen_k, const float p_dropout,
    const float softmax_scale, const bool zero_tensors, bool is_causal,
    int window_size_left, int window_size_right, const float softcap,
    const bool return_softmax, c10::optional<at::Generator> gen_,
    c10::optional<at::Tensor> &s_aux_)
{
  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention GCU only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32,
              "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32,
              "cu_seqlens_k must have dtype int32");

  at::Tensor block_table;
  const bool paged_KV = block_table_.has_value();
  if (paged_KV)
  {
    block_table = block_table_.value();
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
  }

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");

  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);

  const auto sizes = q.sizes();
  LOG(WARNING) << "q sizes: " << sizes;

  const int batch_size = cu_seqlens_q.numel() - 1;
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  int num_heads = sizes[1];
  const int head_size = sizes[2];
  const int num_heads_k = paged_KV ? k.size(2) : k.size(1);
  // const int num_heads_k = k.size(1);
  const int head_size_v = v.size(-1);

  if (softcap > 0.f)
  {
    TORCH_CHECK(p_dropout == 0.f,
                "Softcapping does not support dropout for now");
  }

  const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
  const int num_blocks = !paged_KV ? 0 : k.size(0);
  const int page_block_size = !paged_KV ? 1 : k.size(1);
  TORCH_CHECK(!paged_KV || page_block_size % 16 == 0, "Paged KV cache block size must be divisible by 16");

  // const int max_num_blocks_per_seq = 0;
  // const int num_blocks = 0;
  // const int page_block_size = 1;

  if (window_size_left >= max_seqlen_k)
  {
    window_size_left = -1;
  }
  if (window_size_right >= max_seqlen_k)
  {
    window_size_right = -1;
  }

  // causal=true is the same as causal=false in this case
  if (max_seqlen_q == 1 && !alibi_slopes_.has_value())
  {
    is_causal = false;
  }
  if (is_causal)
  {
    window_size_right = 0;
  }

  // void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

  const int total_q = q.size(0);
  const int total_k = k.size(0);

  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(
      head_size <= 256,
      "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(head_size % 8 == 0,
              "query, key, value, and out_ must have a head_size that is a "
              "multiple of 8");
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");

  CHECK_SHAPE(q, total_q, num_heads, head_size);
  if (!paged_KV)
  {
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
  }
  else
  {
    CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size);
    CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size_v);
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
  }

  if (seqused_k.has_value())
  {
    auto seqused_k_ = seqused_k.value();
    TORCH_CHECK(seqused_k_.dtype() == torch::kInt32,
                "seqused_k must have dtype int32");
    // TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
    TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
    CHECK_SHAPE(seqused_k_, batch_size);
  }

  // auto opts = q.options();
  auto opts = q.options().dtype(at::kFloat).device(at::kPrivateUse1);
  at::Tensor out;
  if (out_.has_value())
  {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype,
                "Output must have the same dtype as inputs");
    TORCH_CHECK(out.stride(-1) == 1,
                "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, sizes[0], sizes[1], head_size_v);
  }
  else
  {
    // out = torch::empty_like(q);
    out = torch::empty({sizes[0], sizes[1], head_size_v}, opts.dtype(q_dtype));
  }

  LOG(WARNING) << "out sizes: " << out.sizes();

  auto softmax_lse = torch::empty({num_heads, total_q}, opts);
  at::Tensor p;
  // Only return softmax if there's dropout to reduce compilation time
  if (return_softmax)
  {
    TORCH_CHECK(p_dropout > 0.0f,
                "return_softmax is only supported when p_dropout > 0.0");
    p = torch::empty({batch_size, num_heads, max_seqlen_q, max_seqlen_k}, opts);
  }
  else
  {
    p = torch::empty({0}, opts);
  }
  ///////////////////

  ///////////////////
  // if (paged_KV) {
  //     params.block_table = block_table.data_ptr<int>();
  //     params.block_table_batch_stride = block_table.stride(0);
  //     params.k_batch_stride = k_padded.stride(0);
  //     params.v_batch_stride = v_padded.stride(0);
  // }
  // params.page_block_size = page_block_size;
  // Keep references to these tensors to extend their lifetime
  // at::Tensor softmax_lse_accum, out_accum;
  at::Tensor leftpad_k;
  if (leftpad_k_.has_value())
  {
    leftpad_k = leftpad_k_.value();
    // TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running
    // at the same time yet");
    TORCH_CHECK(leftpad_k.dtype() == torch::kInt32,
                "leftpad_k must have dtype int32");
    CHECK_CONTIGUOUS(leftpad_k);
    CHECK_SHAPE(leftpad_k, batch_size);
    // params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
  }

  // number of times random will be generated per thread, to offset philox
  // counter in thc random state We use a custom RNG that increases the offset
  // by batch_size * nheads * 32. int64_t counter_offset = params.b * params.h *
  // 32;
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(at::kPrivateUse1);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));

  // Forward kernel will populate memory with the seed and offset.
  /* In put data handle */

  auto q_num_heads = num_heads;
  auto kv_num_heads = num_heads_k;

  topsatenDataType_t q_data_type = q_dtype == torch::kBFloat16 ? topsatenDataType_t::TOPSATEN_DATA_BF16
                                                               : topsatenDataType_t::TOPSATEN_DATA_FP16;

  int batch = batch_size;
  // int total_q = 0;
  // int total_k = 0;
  // int num_queries_per_kv = q_num_heads / kv_num_heads;
  // for (int i = 0; i < batch; ++i) {
  //   total_q += q_lens.shape[i];
  //   total_k += kv_lens.shape[i];
  // }

  int ele_cu_seqlens = batch + 1;
  int ele_seqused_k = batch;

  int max_block_per_request = max_num_blocks_per_seq;

  int64_t dims_rng_state[] = {2};
  int64_t strides_rng_state[] = {1};

  int64_t dims_softmax_lse[] = {q_num_heads, total_q};
  int64_t strides_softmax_lse[] = {total_q, 1};

  LOG(WARNING) << "This is an informational log message begin.";
  LOG(WARNING) << "q_num_heads is " << q_num_heads;
  LOG(WARNING) << "kv_num_heads is " << kv_num_heads;
  LOG(WARNING) << "batch_size is " << batch_size;
  LOG(WARNING) << "ele_cu_seqlens is " << ele_cu_seqlens;
  LOG(WARNING) << "ele_seqused_k is " << ele_seqused_k;
  LOG(WARNING) << "max_block_per_request is " << max_block_per_request;
  LOG(WARNING) << "total_q is " << total_q;
  LOG(WARNING) << "total_k is " << total_k;
  LOG(WARNING) << "head_size is " << head_size;

  topsatenSize_t tensor_dims;
  topsatenSize_t tensor_strides;

  TORCH_CHECK(rng_state.numel() > 0, "rng_state is null tensor");
  // construct rng state params
  tensor_dims.data = dims_rng_state;
  tensor_dims.len = 1;
  tensor_strides.data = strides_rng_state;
  tensor_strides.len = 1;
  topsatenTensor rng_state_tensor(tensor_dims, tensor_strides,
                                  topsatenDataType_t::TOPSATEN_DATA_U64,
                                  rng_state.data_ptr());
  if (softmax_lse.numel() <= 0)
  {
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
    return {out, softmax_lse, p, rng_state};
  }
  // TORCH_CHECK(softmax_lse.numel() > 0, "softmax_lse is null tensor");
  // construct softmax_lse tensor params
  tensor_dims.data = dims_softmax_lse;
  tensor_dims.len = 2;
  tensor_strides.data = strides_softmax_lse;
  tensor_strides.len = 2;
  topsatenTensor softmax_lse_tensor(tensor_dims, tensor_strides,
                                    topsatenDataType_t::TOPSATEN_DATA_FP32,
                                    softmax_lse.data_ptr());

  TORCH_CHECK(out.numel() > 0, "out is null tensor");
  auto out_tensor = makeTopsatenTensor(out, q_data_type);

  TORCH_CHECK(q.numel() > 0, "q is null tensor");
  topsatenTensor q_tensor = makeTopsatenTensor(q, q_data_type);

  TORCH_CHECK(k.numel() > 0, "k is null tensor");
  topsatenTensor k_tensor = makeTopsatenTensor(k, q_data_type);

  TORCH_CHECK(v.numel() > 0, "v is null tensor");
  topsatenTensor v_tensor = makeTopsatenTensor(v, q_data_type);

  TORCH_CHECK(cu_seqlens_q.numel() > 0, "cu_seqlens_q is null tensor");
  auto cu_seqlens_q_tensor =
      makeTopsatenTensor(cu_seqlens_q, topsatenDataType_t::TOPSATEN_DATA_I32);

  TORCH_CHECK(cu_seqlens_k.numel() > 0, "cu_seqlens_k is null tensor");
  auto cu_seqlens_k_tensor =
      makeTopsatenTensor(cu_seqlens_k, topsatenDataType_t::TOPSATEN_DATA_I32);

  topsatenTensor block_table_tensor = makeOptionalTopsatenTensor(
      block_table_, topsatenDataType_t::TOPSATEN_DATA_I32);

  topsatenTensor seqused_k_tensor = makeOptionalTopsatenTensor(
      seqused_k, topsatenDataType_t::TOPSATEN_DATA_I32);
  topsatenTensor leftpad_k_tensor =
      leftpad_k_.has_value()
          ? makeTopsatenTensor(leftpad_k, topsatenDataType_t::TOPSATEN_DATA_I32)
          : topsatenTensor();
  topsatenTensor alibi_slopes_tensor = makeOptionalTopsatenTensor(
      alibi_slopes_, topsatenDataType_t::TOPSATEN_DATA_FP32);

  topsatenTensor s_aux_tensor = makeOptionalTopsatenTensor(
      s_aux_, q_data_type);

  // fixme is the same generator with the PhiloxState
  auto torch_generator =
      at::get_generator_or_default<torch_gcu::GCUGeneratorImpl>(
          gen_, torch_gcu::getDefaultGCUGenerator(q.device().index()));

  bool is_dropout = p_dropout > 0.0;
  uint64_t drop_seed = is_dropout > 0.0 ? torch_generator->current_seed() : 0;
  uint64_t drop_offset = is_dropout > 0.0 ? torch_generator->get_offset() : 0;

  // if (p_dropout > 0.0)  {
  // int64_t counter_offset = params.b * params.h * 32;
  // See Note [Acquire lock when using random generators]
  // std::lock_guard<std::mutex> lock(torch_generator->mutex_);
  // at::PhiloxCudaState philox_args = torch_generator->philox_gcu_state(counter_offset);
  // std::tie(drop_seed, drop_offset) = flash::unpack(philox_args);
  // todo using different way if (arg.captured_)
  // topsatenPhiloxState_t philox_state;
  // philox_state.seed.val = torch_generator->current_seed();
  // philox_state.offset.val = torch_generator->get_offset();
  // }

  topsatenGenerator_t generator{drop_seed, drop_offset};
  // fixme p tensor
  topsatenTensor p_tensor;

  topsatenScalar_t max_seqlen_q_scalar;
  max_seqlen_q_scalar.dtype = TOPSATEN_DATA_I32;
  max_seqlen_q_scalar.ival = max_seqlen_q;

  topsatenScalar_t max_seqlen_k_scalar;
  max_seqlen_k_scalar.dtype = TOPSATEN_DATA_I32;
  max_seqlen_k_scalar.ival = max_seqlen_k;

  topsatenScalar_t p_dropout_scalar;
  p_dropout_scalar.dtype = TOPSATEN_DATA_FP32;
  p_dropout_scalar.fval = p_dropout;

  topsatenScalar_t softmax_scale_scalar;
  softmax_scale_scalar.dtype = TOPSATEN_DATA_FP32;
  softmax_scale_scalar.fval = softmax_scale;

  topsatenScalar_t window_size_left_scalar;
  window_size_left_scalar.dtype = TOPSATEN_DATA_I32;
  window_size_left_scalar.ival = window_size_left;

  topsatenScalar_t window_size_right_scalar;
  window_size_right_scalar.dtype = TOPSATEN_DATA_I32;
  window_size_right_scalar.ival = window_size_right;

  topsatenScalar_t softcap_scalar;
  softcap_scalar.dtype = TOPSATEN_DATA_FP32;
  softcap_scalar.fval = softcap;

  std::tuple<topsatenTensor, topsatenTensor, topsatenTensor, topsatenTensor>
      outputs(out_tensor, softmax_lse_tensor, p_tensor, rng_state_tensor);

  ////////////////////////////////////////////////////////////////////////////////////////
  // out.fill_(std::numeric_limits<float>::infinity());
  if (max_seqlen_k > 0)
  {
    auto stream = torch_gcu::getCurrentGCUStream();
    // LOG(WARNING) << "debug print before topsop " << out[0][0][0];
    // fixme data type convert
    TOPSATEN_CHECK(topsfa::topsfaFlashAttnVarlenFwd(
        outputs, q_tensor, k_tensor, v_tensor, cu_seqlens_q_tensor,
        cu_seqlens_k_tensor, seqused_k_tensor, leftpad_k_tensor,
        block_table_tensor, alibi_slopes_tensor, max_seqlen_q_scalar,
        max_seqlen_k_scalar, p_dropout_scalar, softmax_scale_scalar,
        zero_tensors, is_causal, window_size_left_scalar,
        window_size_right_scalar, softcap_scalar, return_softmax, generator,
        s_aux_tensor, stream));

    // stream.synchronize();
    // LOG(WARNING) << "debug print after topsop " << out[0][0][0];
  }
  else
  {
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output
    // to 0.
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
  }

  return {out, softmax_lse, p, rng_state};
}

std::vector<at::Tensor> mha_varlen_fwd_fp8KV(
    at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    c10::optional<at::Tensor> &out_, const at::Tensor &cu_seqlens_q,
    const at::Tensor &cu_seqlens_k, c10::optional<at::Tensor> &seqused_k,
    c10::optional<const at::Tensor> &leftpad_k_,
    c10::optional<at::Tensor> &block_table_,
    c10::optional<at::Tensor> &alibi_slopes_,
    c10::optional<at::Tensor> &q_descale_,
    c10::optional<at::Tensor> &k_descale_,
    c10::optional<at::Tensor> &v_descale_, c10::optional<at::Tensor> &s_aux_,
    const c10::optional<at::Tensor> &scheduler_metadata_,
    int max_seqlen_q, const int max_seqlen_k, const float p_dropout,
    const float softmax_scale, const bool zero_tensors, bool is_causal,
    int window_size_left, int window_size_right, const float softcap,
    const bool return_softmax, c10::optional<at::Generator> gen_)
{
  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention FP8 KV: query must be fp16 or bf16");
  TORCH_CHECK(k.dtype() == v.dtype(),
              "FlashAttention FP8 KV: key and value must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32,
              "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32,
              "cu_seqlens_k must have dtype int32");

  at::Tensor block_table;
  const bool paged_KV = block_table_.has_value();
  if (paged_KV)
  {
    block_table = block_table_.value();
    TORCH_CHECK(block_table.dtype() == torch::kInt32,
                "block_table must have dtype torch.int32");
    TORCH_CHECK(block_table.stride(-1) == 1,
                "block_table must have contiguous last dimension");
  }

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");

  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);

  const auto sizes = q.sizes();
  LOG(WARNING) << "q sizes: " << sizes;

  const int batch_size = cu_seqlens_q.numel() - 1;
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  int num_heads = sizes[1];
  const int head_size = sizes[2];
  const int num_heads_k = paged_KV ? k.size(2) : k.size(1);
  const int head_size_v = v.size(-1);

  const topsatenDataType_t kv_data_type = get_gcu_dtype(k);

  if (softcap > 0.f)
  {
    TORCH_CHECK(p_dropout == 0.f,
                "Softcapping does not support dropout for now");
  }

  const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
  const int num_blocks = !paged_KV ? 0 : k.size(0);
  const int page_block_size = !paged_KV ? 1 : k.size(1);
  TORCH_CHECK(!paged_KV || page_block_size % 16 == 0,
              "Paged KV cache block size must be divisible by 16");

  if (window_size_left >= max_seqlen_k)
  {
    window_size_left = -1;
  }
  if (window_size_right >= max_seqlen_k)
  {
    window_size_right = -1;
  }

  if (max_seqlen_q == 1 && !alibi_slopes_.has_value())
  {
    is_causal = false;
  }
  if (is_causal)
  {
    window_size_right = 0;
  }

  const int total_q = q.size(0);
  const int total_k = k.size(0);

  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(
      head_size <= 256,
      "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(head_size % 8 == 0,
              "query, key, value, and out_ must have a head_size that is a "
              "multiple of 8");
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");

  CHECK_SHAPE(q, total_q, num_heads, head_size);
  if (!paged_KV)
  {
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
  }
  else
  {
    CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size);
    CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size_v);
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
  }

  if (seqused_k.has_value())
  {
    auto seqused_k_ = seqused_k.value();
    TORCH_CHECK(seqused_k_.dtype() == torch::kInt32,
                "seqused_k must have dtype int32");
    TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
    CHECK_SHAPE(seqused_k_, batch_size);
  }

  auto opts = q.options().dtype(at::kFloat).device(at::kPrivateUse1);
  at::Tensor out;
  if (out_.has_value())
  {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype,
                "Output must have the same dtype as query");
    TORCH_CHECK(out.stride(-1) == 1,
                "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, sizes[0], sizes[1], head_size_v);
  }
  else
  {
    out = torch::empty({sizes[0], sizes[1], head_size_v}, opts.dtype(q_dtype));
  }

  LOG(WARNING) << "out sizes: " << out.sizes();

  auto softmax_lse = torch::empty({num_heads, total_q}, opts);
  at::Tensor p;
  if (return_softmax)
  {
    TORCH_CHECK(p_dropout > 0.0f,
                "return_softmax is only supported when p_dropout > 0.0");
    p = torch::empty({batch_size, num_heads, max_seqlen_q, max_seqlen_k}, opts);
  }
  else
  {
    p = torch::empty({0}, opts);
  }

  at::Tensor leftpad_k;
  if (leftpad_k_.has_value())
  {
    leftpad_k = leftpad_k_.value();
    TORCH_CHECK(leftpad_k.dtype() == torch::kInt32,
                "leftpad_k must have dtype int32");
    CHECK_CONTIGUOUS(leftpad_k);
    CHECK_SHAPE(leftpad_k, batch_size);
  }

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(at::kPrivateUse1);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));

  auto q_num_heads = num_heads;
  auto kv_num_heads = num_heads_k;

  topsatenDataType_t q_data_type = get_gcu_dtype(q);

  int batch = batch_size;
  int ele_cu_seqlens = batch + 1;
  int ele_seqused_k = batch;
  int max_block_per_request = max_num_blocks_per_seq;

  int64_t dims_rng_state[] = {2};
  int64_t strides_rng_state[] = {1};

  int64_t dims_softmax_lse[] = {q_num_heads, total_q};
  int64_t strides_softmax_lse[] = {total_q, 1};

  LOG(WARNING) << "This is an informational log message begin.";
  LOG(WARNING) << "q_num_heads is " << q_num_heads;
  LOG(WARNING) << "kv_num_heads is " << kv_num_heads;
  LOG(WARNING) << "batch_size is " << batch_size;
  LOG(WARNING) << "ele_cu_seqlens is " << ele_cu_seqlens;
  LOG(WARNING) << "ele_seqused_k is " << ele_seqused_k;
  LOG(WARNING) << "max_block_per_request is " << max_block_per_request;
  LOG(WARNING) << "total_q is " << total_q;
  LOG(WARNING) << "total_k is " << total_k;
  LOG(WARNING) << "head_size is " << head_size;

  topsatenSize_t tensor_dims;
  topsatenSize_t tensor_strides;

  TORCH_CHECK(rng_state.numel() > 0, "rng_state is null tensor");
  tensor_dims.data = dims_rng_state;
  tensor_dims.len = 1;
  tensor_strides.data = strides_rng_state;
  tensor_strides.len = 1;
  topsatenTensor rng_state_tensor(tensor_dims, tensor_strides,
                                  topsatenDataType_t::TOPSATEN_DATA_U64,
                                  rng_state.data_ptr());
  if (softmax_lse.numel() <= 0)
  {
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
    return {out, softmax_lse, p, rng_state};
  }

  tensor_dims.data = dims_softmax_lse;
  tensor_dims.len = 2;
  tensor_strides.data = strides_softmax_lse;
  tensor_strides.len = 2;
  topsatenTensor softmax_lse_tensor(tensor_dims, tensor_strides,
                                    topsatenDataType_t::TOPSATEN_DATA_FP32,
                                    softmax_lse.data_ptr());

  TORCH_CHECK(out.numel() > 0, "out is null tensor");
  auto out_tensor = makeTopsatenTensor(out, q_data_type);

  TORCH_CHECK(q.numel() > 0, "q is null tensor");
  topsatenTensor q_tensor = makeTopsatenTensor(q, q_data_type);

  TORCH_CHECK(k.numel() > 0, "k is null tensor");
  topsatenTensor k_tensor = makeTopsatenTensor(k, kv_data_type);

  TORCH_CHECK(v.numel() > 0, "v is null tensor");
  topsatenTensor v_tensor = makeTopsatenTensor(v, kv_data_type);

  TORCH_CHECK(cu_seqlens_q.numel() > 0, "cu_seqlens_q is null tensor");
  auto cu_seqlens_q_tensor =
      makeTopsatenTensor(cu_seqlens_q, topsatenDataType_t::TOPSATEN_DATA_I32);

  TORCH_CHECK(cu_seqlens_k.numel() > 0, "cu_seqlens_k is null tensor");
  auto cu_seqlens_k_tensor =
      makeTopsatenTensor(cu_seqlens_k, topsatenDataType_t::TOPSATEN_DATA_I32);

  topsatenTensor block_table_tensor = makeOptionalTopsatenTensor(
      block_table_, topsatenDataType_t::TOPSATEN_DATA_I32);

  topsatenTensor seqused_k_tensor = makeOptionalTopsatenTensor(
      seqused_k, topsatenDataType_t::TOPSATEN_DATA_I32);
  topsatenTensor leftpad_k_tensor =
      leftpad_k_.has_value()
          ? makeTopsatenTensor(leftpad_k, topsatenDataType_t::TOPSATEN_DATA_I32)
          : topsatenTensor();
  topsatenTensor alibi_slopes_tensor = makeOptionalTopsatenTensor(
      alibi_slopes_, topsatenDataType_t::TOPSATEN_DATA_FP32);

  if (q_descale_.has_value())
  {
    TORCH_CHECK(q_descale_.value().dtype() == torch::kFloat32,
                "q_descale must be float32");
  }
  if (k_descale_.has_value())
  {
    TORCH_CHECK(k_descale_.value().dtype() == torch::kFloat32,
                "k_descale must be float32");
  }
  if (v_descale_.has_value())
  {
    TORCH_CHECK(v_descale_.value().dtype() == torch::kFloat32,
                "v_descale must be float32");
  }

  topsatenTensor q_descale_tensor = makeOptionalTopsatenTensor(
      q_descale_, topsatenDataType_t::TOPSATEN_DATA_FP32);
  topsatenTensor k_descale_tensor = makeOptionalTopsatenTensor(
      k_descale_, topsatenDataType_t::TOPSATEN_DATA_FP32);
  topsatenTensor v_descale_tensor = makeOptionalTopsatenTensor(
      v_descale_, topsatenDataType_t::TOPSATEN_DATA_FP32);

  topsatenTensor s_aux_tensor = makeOptionalTopsatenTensor(s_aux_, q_data_type);
  // topsfaFlashAttnVarlenFwdFp8KV(..., const topsatenTensor& sche_metadata, ...):
  // pass empty topsatenTensor when Python did not supply scheduler_metadata.
  topsatenTensor scheduler_metadata_tensor = makeOptionalTopsatenTensor(
      scheduler_metadata_, topsatenDataType_t::TOPSATEN_DATA_I32);

  auto torch_generator =
      at::get_generator_or_default<torch_gcu::GCUGeneratorImpl>(
          gen_, torch_gcu::getDefaultGCUGenerator(q.device().index()));

  bool is_dropout = p_dropout > 0.0;
  uint64_t drop_seed = is_dropout > 0.0 ? torch_generator->current_seed() : 0;
  uint64_t drop_offset = is_dropout > 0.0 ? torch_generator->get_offset() : 0;

  topsatenGenerator_t generator{drop_seed, drop_offset};
  topsatenTensor p_tensor;

  topsatenScalar_t max_seqlen_q_scalar;
  max_seqlen_q_scalar.dtype = TOPSATEN_DATA_I32;
  max_seqlen_q_scalar.ival = max_seqlen_q;

  topsatenScalar_t max_seqlen_k_scalar;
  max_seqlen_k_scalar.dtype = TOPSATEN_DATA_I32;
  max_seqlen_k_scalar.ival = max_seqlen_k;

  topsatenScalar_t p_dropout_scalar;
  p_dropout_scalar.dtype = TOPSATEN_DATA_FP32;
  p_dropout_scalar.fval = p_dropout;

  topsatenScalar_t softmax_scale_scalar;
  softmax_scale_scalar.dtype = TOPSATEN_DATA_FP32;
  softmax_scale_scalar.fval = softmax_scale;

  topsatenScalar_t window_size_left_scalar;
  window_size_left_scalar.dtype = TOPSATEN_DATA_I32;
  window_size_left_scalar.ival = window_size_left;

  topsatenScalar_t window_size_right_scalar;
  window_size_right_scalar.dtype = TOPSATEN_DATA_I32;
  window_size_right_scalar.ival = window_size_right;

  topsatenScalar_t softcap_scalar;
  softcap_scalar.dtype = TOPSATEN_DATA_FP32;
  softcap_scalar.fval = softcap;

  std::tuple<topsatenTensor, topsatenTensor, topsatenTensor, topsatenTensor>
      outputs(out_tensor, softmax_lse_tensor, p_tensor, rng_state_tensor);

  if (max_seqlen_k > 0)
  {
    auto stream = torch_gcu::getCurrentGCUStream();
    // LOG(WARNING) << "debug print before topsop " << out[0][0][0];
    TOPSATEN_CHECK(topsfa::topsfaFlashAttnVarlenFwdFp8KV(
        outputs, q_tensor, k_tensor, v_tensor, cu_seqlens_q_tensor,
        cu_seqlens_k_tensor, seqused_k_tensor, leftpad_k_tensor,
        block_table_tensor, alibi_slopes_tensor, q_descale_tensor,
        k_descale_tensor, v_descale_tensor, max_seqlen_q_scalar,
        max_seqlen_k_scalar, p_dropout_scalar, softmax_scale_scalar,
        zero_tensors, is_causal, window_size_left_scalar,
        window_size_right_scalar, softcap_scalar, return_softmax, generator,
        s_aux_tensor, scheduler_metadata_tensor, stream));
    // stream.synchronize();
    // LOG(WARNING) << "debug print after topsop " << out[0][0][0];
  }
  else
  {
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
  }

  return {out, softmax_lse, p, rng_state};
}

std::vector<at::Tensor> mha_varlen_bwd(
    const at::Tensor &dout, // total_q x num_heads, x head_size
    const at::Tensor
        &q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor
        &k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor
        &v,                        // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &out,         // total_q x num_heads x head_size
    const at::Tensor &softmax_lse, // h x total_q, softmax logsumexp
    c10::optional<at::Tensor>
        &dq_,                                 // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    c10::optional<at::Tensor> &dk_,           // total_k x num_heads_k x head_size,
                                              // total_k := \sum_{i=0}^{b} s_i
    c10::optional<at::Tensor> &dv_,           // total_k x num_heads_k x head_size,
                                              // total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &cu_seqlens_q,           // b+1
    const at::Tensor &cu_seqlens_k,           // b+1
    c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
    const int max_seqlen_q,
    const int max_seqlen_k, // max sequence length to choose the kernel
    const float p_dropout,  // probability to drop
    const float softmax_scale, const bool zero_tensors, const bool is_causal,
    int window_size_left, int window_size_right, const float softcap,
    const bool deterministic, c10::optional<at::Generator> gen_,
    c10::optional<at::Tensor> &rng_state)
{
#ifdef FLASHATTENTION_DISABLE_BACKWARD
  TORCH_CHECK(false, "This flash attention build does not support backward.");
#endif

  if (is_causal)
  {
    window_size_right = 0;
  }

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention GCU only support fp16 and bf16 data type");

  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
  TORCH_CHECK(dout.dtype() == q_dtype,
              "query and dout must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32,
              "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32,
              "cu_seqlens_k must have dtype int32");

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(out.stride(-1) == 1,
              "out tensor must have contiguous last dimension");
  TORCH_CHECK(dout.stride(-1) == 1,
              "dout tensor must have contiguous last dimension");
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);

  const auto sizes = q.sizes();

  const int total_q = sizes[0];
  const int batch_size = cu_seqlens_q.numel() - 1;
  const int num_heads = sizes[1];
  const int head_size_og = dout.size(2);
  const int head_size = sizes[2];
  const int total_k = k.size(0);
  const int num_heads_k = k.size(1);
  const int head_size_v = v.size(-1);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(head_size_v % 8 == 0, "head_size_v should be a multiple of 8");
  TORCH_CHECK(
      head_size <= 256,
      "FlashAttention backward only supports head dimension at most 256");

  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");
  if (softcap > 0.f)
  {
    TORCH_CHECK(p_dropout == 0.f,
                "Softcapping does not support dropout for now");
  }

  auto round_multiple = [](int x, int m)
  { return (x + m - 1) / m * m; };
  const int head_size_rounded =
      head_size <= 192 ? round_multiple(head_size, 32) : 256;
  // const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
  // const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

  TORCH_CHECK(head_size_v == round_multiple(head_size_og, 8),
              "head_size_v must be head_size_og rounded to a multiple of 8");

  if (window_size_left >= max_seqlen_k)
  {
    window_size_left = -1;
  }
  if (window_size_right >= max_seqlen_k)
  {
    window_size_right = -1;
  }

  CHECK_SHAPE(q, total_q, num_heads, head_size);
  CHECK_SHAPE(k, total_k, num_heads_k, head_size);
  CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
  CHECK_SHAPE(out, total_q, num_heads, head_size_v);
  CHECK_SHAPE(dout, total_q, num_heads, head_size_og);
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

  at::Tensor dq, dk, dv;
  if (dq_.has_value())
  {
    dq = dq_.value();
    TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
    TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
    CHECK_SHAPE(dq, total_q, num_heads, head_size);
  }
  else
  {
    dq = torch::empty_like(q);
  }
  if (dk_.has_value())
  {
    dk = dk_.value();
    TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
    TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
  }
  else
  {
    dk = torch::empty_like(k);
  }
  if (dv_.has_value())
  {
    dv = dv_.value();
    TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
    TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
    CHECK_SHAPE(dv, total_k, num_heads_k, head_size_v);
  }
  else
  {
    dv = torch::empty_like(v);
  }

  at::Tensor dout_padded;
  if (head_size_og % 8 != 0)
  {
    dout_padded = torch::nn::functional::pad(
        dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
  }
  else
  {
    dout_padded = dout;
  }

  // bool loop = max_seqlen_k > blocksize_c;
  // TODO: change later, for now set to true for simplicity
  // bool loop = true;

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  // at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  auto opts = q.options();
  auto softmax_d = torch::empty({num_heads, total_q + 128 * batch_size},
                                opts.dtype(at::kFloat));
  at::Tensor dq_accum;
  /*
  if (loop) {
      // We don't want to allocate dq_accum of size (batch, seqlen_q_rounded,
  num_heads, head_size_rounded)
      // because that would be too large if there is a very long sequence and
  the rest of the sequences are short.
      // Instead, we allocate dq_accum of size (total_q + 128 * batch,
  num_heads, head_size_rounded).
      // Note that 128 is the max block size on the seqlen_q dimension.
      // For dQ, the i-th sequence is stored in indices from cu_seqlens[i] + 128
  * i to
      // cu_seqlens[i + 1] * 128 * i - 1. This ensures that the i-th sequence
  and (i + 1)-th sequence will
      // be at least 128 apart. It's ok for us to do atomicAdds up to 128 rows
  beyond what we're normally
      // allowed to do. So we won't have to do any bound checking, and
  performance should stay the same.
      // Same holds for softmax_d, since LSE is stored in unpadded format.
      if (!deterministic) {
          dq_accum = torch::empty({total_q + 128 * batch_size, num_heads,
  head_size_rounded}, opts.dtype(at::kFloat)); } else { const int nsplits =
  (dprops->multiProcessorCount + batch_size * num_heads - 1) / (batch_size *
  num_heads); dq_accum = torch::zeros({nsplits, total_q + 128 * batch_size,
  num_heads, head_size_rounded}, opts.dtype(at::kFloat));
      }
  }
  */
  dq_accum =
      torch::empty({total_q + 128 * batch_size, num_heads, head_size_rounded},
                   opts.dtype(at::kFloat));

  // at::Tensor dk_expanded, dv_expanded;
  // if (num_heads_k != num_heads)
  // { // MQA / GQA
  //   dk_expanded = torch::empty({total_k, num_heads, head_size}, opts);
  //   dv_expanded = torch::empty({total_k, num_heads, head_size}, opts);
  // }
  // else
  // {
  //   dk_expanded = dk;
  //   dv_expanded = dv;
  // }

  if (zero_tensors)
  {
    dq.zero_();
    dk.zero_();
    dv.zero_();
    softmax_d.zero_();
  }

  // ---------------------start gcu tensor build--------------------------------
  auto q_num_heads = num_heads;
  auto kv_num_heads = num_heads_k;

  LOG(WARNING) << "This is an informational log message begin.";
  LOG(WARNING) << "q_num_heads is " << q_num_heads;
  LOG(WARNING) << "kv_num_heads is " << kv_num_heads;
  LOG(WARNING) << "total_q is " << total_q;
  LOG(WARNING) << "total_k is " << total_k;

  topsatenDataType_t q_data_type = q_dtype == torch::kBFloat16 ? topsatenDataType_t::TOPSATEN_DATA_BF16
                                                               : topsatenDataType_t::TOPSATEN_DATA_FP16;

  TORCH_CHECK(q.numel() > 0, "q is null tensor");
  topsatenTensor q_tensor = makeTopsatenTensor(q, q_data_type);

  TORCH_CHECK(k.numel() > 0, "k is null tensor");
  topsatenTensor k_tensor = makeTopsatenTensor(k, q_data_type);

  TORCH_CHECK(v.numel() > 0, "v is null tensor");
  topsatenTensor v_tensor = makeTopsatenTensor(v, q_data_type);

  TORCH_CHECK(out.numel() > 0, "out is null tensor");
  auto out_tensor = makeTopsatenTensor(out, q_data_type);

  auto dout_tensor = makeTopsatenTensor(dout, q_data_type);
  auto dq_tensor = makeTopsatenTensor(dq, q_data_type);
  auto dk_tensor = makeTopsatenTensor(dk, q_data_type);
  auto dv_tensor = makeTopsatenTensor(dv, q_data_type);

  // construct softmax_lse tensor params
  topsatenSize_t softmax_lse_tensor_dims, softmax_lse_tensor_strides;
  int64_t dims_softmax_lse[] = {q_num_heads, total_q};
  int64_t strides_softmax_lse[] = {total_q, 1};
  topsatenDataType_t softmax_lse_data_type =
      topsatenDataType_t::TOPSATEN_DATA_FP32;
  softmax_lse_tensor_dims.data = dims_softmax_lse;
  softmax_lse_tensor_dims.len = 2;
  softmax_lse_tensor_strides.data = strides_softmax_lse;
  softmax_lse_tensor_strides.len = 2;

  topsatenTensor softmax_lse_tensor(
      softmax_lse_tensor_dims, softmax_lse_tensor_strides,
      softmax_lse_data_type, softmax_lse.data_ptr());

  TORCH_CHECK(cu_seqlens_q.numel() > 0, "cu_seqlens_q is null tensor");
  auto cu_seqlens_q_tensor =
      makeTopsatenTensor(cu_seqlens_q, topsatenDataType_t::TOPSATEN_DATA_I32);

  TORCH_CHECK(cu_seqlens_k.numel() > 0, "cu_seqlens_k is null tensor");
  auto cu_seqlens_k_tensor =
      makeTopsatenTensor(cu_seqlens_k, topsatenDataType_t::TOPSATEN_DATA_I32);

  // result tensor contruct
  // fix me need concern about data type
  int64_t dims_softmax_d[2] = {num_heads, total_q + 128 * batch_size};
  int64_t stride_softmax_d[2] = {total_q + 128 * batch_size, 1};
  topsatenSize_t softmax_d_tensor_dims, softmax_d_tensor_strides;
  softmax_d_tensor_dims.data = dims_softmax_d;
  softmax_d_tensor_dims.len = 2;
  softmax_d_tensor_strides.data = stride_softmax_d;
  softmax_d_tensor_strides.len = 2;
  topsatenTensor softmax_d_tensor(
      softmax_d_tensor_dims, softmax_d_tensor_strides,
      topsatenDataType_t::TOPSATEN_DATA_FP32, softmax_d.data_ptr());

  std::tuple<topsatenTensor, topsatenTensor, topsatenTensor, topsatenTensor>
      result(dq_tensor, dk_tensor, dv_tensor, softmax_d_tensor);

  topsatenTensor alibi_slopes_tensor = makeOptionalTopsatenTensor(
      alibi_slopes_, topsatenDataType_t::TOPSATEN_DATA_FP32);

  // fixme do need exact max q k value

  topsatenScalar_t max_seqlen_q_scalar;
  max_seqlen_q_scalar.dtype = TOPSATEN_DATA_I32;
  max_seqlen_q_scalar.ival = max_seqlen_q;

  topsatenScalar_t max_seqlen_k_scalar;
  max_seqlen_k_scalar.dtype = TOPSATEN_DATA_I32;
  max_seqlen_k_scalar.ival = max_seqlen_k;

  topsatenScalar_t p_dropout_scalar;
  p_dropout_scalar.dtype = TOPSATEN_DATA_FP32;
  p_dropout_scalar.fval = p_dropout;

  topsatenScalar_t softmax_scale_scalar;
  softmax_scale_scalar.dtype = TOPSATEN_DATA_FP32;
  softmax_scale_scalar.fval = softmax_scale;

  topsatenScalar_t window_size_left_scalar;
  window_size_left_scalar.dtype = TOPSATEN_DATA_I32;
  window_size_left_scalar.ival = window_size_left;

  topsatenScalar_t window_size_right_scalar;
  window_size_right_scalar.dtype = TOPSATEN_DATA_I32;
  window_size_right_scalar.ival = window_size_right;

  topsatenScalar_t softcap_scalar;
  softcap_scalar.dtype = TOPSATEN_DATA_FP32;
  softcap_scalar.fval = softcap;

  // fixme is the same generator with the PhiloxState
  auto torch_generator =
      at::get_generator_or_default<torch_gcu::GCUGeneratorImpl>(
          gen_, torch_gcu::getDefaultGCUGenerator(q.device().index()));

  bool is_dropout = p_dropout > 0.0;
  uint64_t drop_seed = is_dropout > 0.0 ? torch_generator->current_seed() : 0;
  uint64_t drop_offset = is_dropout > 0.0 ? torch_generator->get_offset() : 0;

  topsatenGenerator_t generator{drop_seed, drop_offset};
  topsatenTensor rng_state_tensor = makeOptionalTopsatenTensor(
      rng_state, topsatenDataType_t::TOPSATEN_DATA_U64);
  // ---------------------end gcu tensor build--------------------------------

  if (max_seqlen_q > 0)
  {
    auto stream = torch_gcu::getCurrentGCUStream();
    TOPSATEN_CHECK(topsfa::topsfaFlashAttnVarlenBwd(
        result, dout_tensor, q_tensor, k_tensor, v_tensor, out_tensor,
        softmax_lse_tensor,
        dq_tensor, dk_tensor, dv_tensor, cu_seqlens_q_tensor, cu_seqlens_k_tensor,
        alibi_slopes_tensor, max_seqlen_q_scalar, max_seqlen_k_scalar,
        p_dropout_scalar, softmax_scale_scalar, zero_tensors, is_causal,
        window_size_left_scalar, window_size_right_scalar, softcap_scalar,
        deterministic,
        generator, rng_state_tensor, stream));
  }
  else
  {
    // If seqlen_q == 0, then we have an empty tensor. We need to set the output
    // to 0.
    dk.zero_();
    dv.zero_();
    softmax_d.zero_();
  }

  // For MQA/GQA we need to sum dK and dV across the groups
  // if (num_heads_k != num_heads)
  // {
  //   at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads
  //   / num_heads_k, head_size}), {2}); at::sum_out(dv,
  //   at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k,
  //   head_size}), {2});
  // }
  if (head_size_og % 8 != 0)
  {
    dq = dq.index(
        {"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    dk = dk.index(
        {"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    dv = dv.index(
        {"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
  }

  return {dq, dk, dv, softmax_d};
}

std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,                                     // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,                          // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,                          // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                c10::optional<const at::Tensor> &k_,               // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &v_,               // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &seqlens_k_,       // batch_size
                c10::optional<const at::Tensor> &rotary_cos_,      // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &rotary_sin_,      // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                c10::optional<const at::Tensor> &leftpad_k_,       // batch_size
                c10::optional<at::Tensor> &block_table_,           // batch_size x max_num_blocks_per_seq
                c10::optional<at::Tensor> &alibi_slopes_,          // num_heads or batch_size x num_heads
                c10::optional<at::Tensor> &out_,                   // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved, // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits)
{

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");

  TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  at::Tensor block_table;
  const bool paged_KV = block_table_.has_value();
  if (paged_KV)
  {
    TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
    block_table = block_table_.value();
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
  }

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  const int head_size_og = sizes[3];

  const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
  const int num_blocks = !paged_KV ? 0 : kcache.size(0);
  const int page_block_size = !paged_KV ? 1 : kcache.size(1);
  TORCH_CHECK(!paged_KV || page_block_size % 16 == 0, "Paged KV cache block size must be divisible by 16");
  const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_blocks_per_seq * page_block_size;
  const int num_heads_k = kcache.size(2);
  const int batch_size_c = !paged_KV ? kcache.size(0) : batch_size;
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1 && !alibi_slopes_.has_value())
  {
    is_causal = false;
  }
  if (is_causal)
  {
    window_size_right = 0;
  }

  // TODO(GCU): support seqlenq_ngroups_swapped in future
  // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
  // H/t Daniel Haziza
  // const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
  // if (seqlenq_ngroups_swapped)
  // {
  //   const int ngroups = num_heads / num_heads_k;
  //   q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
  //   seqlen_q = ngroups;
  //   num_heads = num_heads_k;
  // }

  if (window_size_left >= seqlen_k)
  {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_k)
  {
    window_size_right = -1;
  }

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
  if (!paged_KV)
  {
    CHECK_SHAPE(kcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
    CHECK_SHAPE(vcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
  }
  else
  {
    CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_og);
    CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_og);
    CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
  }

  at::Tensor out;
  if (out_.has_value())
  {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
    TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
  }
  else
  {
    out = torch::empty_like(q);
  }

  auto opts = q.options().dtype(at::kFloat).device(at::kPrivateUse1);

  auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts);

  at::Tensor k, v;
  if (k_.has_value())
  {
    TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
    TORCH_CHECK(seqlens_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");
    TORCH_CHECK(seqlen_q <= seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
    k = k_.value();
    v = v_.value();
    TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as query");
    TORCH_CHECK(v.dtype() == q_dtype, "Value must have the same dtype as query");

    TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
    int seqlen_knew = k.size(1);
    CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
    CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
  }

  if (seqlens_k_.has_value())
  {
    auto seqlens_k = seqlens_k_.value();
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");

    CHECK_CONTIGUOUS(seqlens_k);
    CHECK_SHAPE(seqlens_k, batch_size);
  }

  if (leftpad_k_.has_value())
  {
    TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
    auto leftpad_k = leftpad_k_.value();
    TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");

    CHECK_CONTIGUOUS(leftpad_k);
    CHECK_SHAPE(leftpad_k, batch_size);
  }

  if (rotary_cos_.has_value())
  {
    TORCH_CHECK(k_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
    auto rotary_cos = rotary_cos_.value();

    auto rotary_dim = rotary_cos.size(1) * 2;
    TORCH_CHECK(rotary_dim <= head_size_og, "rotary_dim must be <= headdim");
    TORCH_CHECK(rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
    const int seqlen_ro = rotary_cos.size(0);
    TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
    CHECK_SHAPE(rotary_cos, seqlen_ro, rotary_dim / 2);
    CHECK_CONTIGUOUS(rotary_cos);
    TORCH_CHECK(rotary_cos.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");

    TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
    auto rotary_sin = rotary_sin_.value();

    CHECK_SHAPE(rotary_sin, seqlen_ro, rotary_dim / 2);
    CHECK_CONTIGUOUS(rotary_sin);
    TORCH_CHECK(rotary_sin.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");
  }

  if (cache_batch_idx_.has_value())
  {
    auto cache_batch_idx = cache_batch_idx_.value();

    CHECK_CONTIGUOUS(cache_batch_idx);
    TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32, "cache_batch_idx must have dtype int32");
  }

  // // Keep references to these tensors to extend their lifetime
  // at::Tensor softmax_lse_accum, out_accum;
  // std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
  //     params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
  //     head_size_rounded, /*dropout*/ 0.f, num_splits, dprops, opts);

  // if (paged_KV)
  // {
  //   params.block_table = block_table.data_ptr<int>();
  //   params.block_table_batch_stride = block_table.stride(0);
  // }
  // params.page_block_size = page_block_size;

  // set_params_alibi(params, alibi_slopes_, batch_size, num_heads);
  topsatenDataType_t q_data_type = q_dtype == torch::kBFloat16 ? topsatenDataType_t::TOPSATEN_DATA_BF16
                                                               : topsatenDataType_t::TOPSATEN_DATA_FP16;
  // construct out tensor params
  auto out_tensor = makeTopsatenTensor(out, q_data_type);

  // construct softmax_lse tensor params
  topsatenSize_t softmax_lse_tensor_dims, softmax_lse_tensor_strides;
  int64_t dims_softmax_lse[3] = {batch_size, num_heads, seqlen_q};
  int64_t stride_softmax_lse[3] = {num_heads * seqlen_q, seqlen_q, 1};
  topsatenDataType_t softmax_lse_data_type =
      topsatenDataType_t::TOPSATEN_DATA_FP32;
  softmax_lse_tensor_dims.data = dims_softmax_lse;
  softmax_lse_tensor_dims.len = 3;
  softmax_lse_tensor_strides.data = stride_softmax_lse;
  softmax_lse_tensor_strides.len = 3;

  topsatenTensor softmax_lse_tensor(
      softmax_lse_tensor_dims, softmax_lse_tensor_strides,
      softmax_lse_data_type, softmax_lse.data_ptr());

  std::tuple<topsatenTensor, topsatenTensor> outputs(
      out_tensor, softmax_lse_tensor);

  TORCH_CHECK(q.numel() > 0, "q is null tensor");
  topsatenTensor q_tensor = makeTopsatenTensor(q, q_data_type);

  TORCH_CHECK(kcache.numel() > 0, "kcache is null tensor");
  topsatenTensor kcache_tensor = makeTopsatenTensor(kcache, q_data_type);

  TORCH_CHECK(vcache.numel() > 0, "vcache is null tensor");
  topsatenTensor vcache_tensor = makeTopsatenTensor(vcache, q_data_type);

  topsatenTensor k_tensor = makeOptionalConstTopsatenTensor(k_, q_data_type);

  topsatenTensor v_tensor = makeOptionalConstTopsatenTensor(v_, q_data_type);

  topsatenTensor seqlens_k_tensor = makeOptionalConstTopsatenTensor(
      seqlens_k_, topsatenDataType_t::TOPSATEN_DATA_I32);

  topsatenTensor rotary_cos_tensor = makeOptionalConstTopsatenTensor(
      rotary_cos_, q_data_type);

  topsatenTensor rotary_sin_tensor = makeOptionalConstTopsatenTensor(
      rotary_sin_, q_data_type);

  topsatenTensor cache_batch_idx_tensor = makeOptionalConstTopsatenTensor(
      cache_batch_idx_, topsatenDataType_t::TOPSATEN_DATA_I32);

  topsatenTensor leftpad_k_tensor = makeOptionalConstTopsatenTensor(
      leftpad_k_, topsatenDataType_t::TOPSATEN_DATA_I32);

  topsatenTensor block_table_tensor = makeOptionalTopsatenTensor(
      block_table_, topsatenDataType_t::TOPSATEN_DATA_I32);

  topsatenTensor alibi_slopes_tensor = makeOptionalTopsatenTensor(
      alibi_slopes_, topsatenDataType_t::TOPSATEN_DATA_FP32);

  topsatenScalar_t softmax_scale_scalar;
  softmax_scale_scalar.dtype = TOPSATEN_DATA_FP32;
  softmax_scale_scalar.fval = softmax_scale;

  topsatenScalar_t window_size_left_scalar;
  window_size_left_scalar.dtype = TOPSATEN_DATA_I32;
  window_size_left_scalar.ival = window_size_left;

  topsatenScalar_t window_size_right_scalar;
  window_size_right_scalar.dtype = TOPSATEN_DATA_I32;
  window_size_right_scalar.ival = window_size_right;

  topsatenScalar_t softcap_scalar;
  softcap_scalar.dtype = TOPSATEN_DATA_FP32;
  softcap_scalar.fval = softcap;

  topsatenScalar_t num_splits_scalar;
  num_splits_scalar.dtype = TOPSATEN_DATA_I32;
  num_splits_scalar.ival = num_splits;

  auto stream = torch_gcu::getCurrentGCUStream();
  TOPSATEN_CHECK(topsfa::topsfaFlashAttnFwdKvcache(
      outputs, q_tensor, kcache_tensor, vcache_tensor, k_tensor, v_tensor,
      seqlens_k_tensor, rotary_cos_tensor, rotary_sin_tensor,
      cache_batch_idx_tensor, leftpad_k_tensor,
      block_table_tensor, alibi_slopes_tensor, softmax_scale_scalar, is_causal,
      window_size_left_scalar, window_size_right_scalar, softcap_scalar,
      is_rotary_interleaved, num_splits_scalar, stream));

  // if (seqlenq_ngroups_swapped)
  // {
  //   out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
  //   softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
  // }
  return {out, softmax_lse};
}

// Only applicable to the case where seqused_k (i.e. cache_seqlens) is available
at::Tensor
mha_fwd_get_scheduler_metadata(
    int batch_size,
    int max_seqlen_q,
    int max_seqlen_k,
    int num_heads,
    int num_heads_k,
    int headdim,
    int headdim_v,
    at::ScalarType qkv_dtype,
    const at::Tensor &seqused_k,                        // b
    const c10::optional<at::Tensor> &cu_seqlens_q_,     // b+1
    const c10::optional<at::Tensor> &cu_seqlens_k_,     // b+1
    const c10::optional<at::Tensor> &cu_seqlens_k_new_, // b+1
    const c10::optional<at::Tensor> &seqused_q_,        // b
    const c10::optional<at::Tensor> &leftpad_k_,        // b
    const c10::optional<int64_t> page_size,
    const int max_seqlen_k_new, // 0 means we're not appending new KV
    const bool is_causal,
    const int window_size_left,
    const int window_size_right,
    const bool has_softcap,
    const int num_splits,
    const c10::optional<bool> pack_gqa_,
    const int sm_margin)
{
  int window_size_left_mut = window_size_left;
  int window_size_right_mut = window_size_right;
  bool is_causal_mut = is_causal;

  // auto check_i32_1d_tensor = [](const at::Tensor &t, const char *name,
  //                               const int64_t expected_numel) {
  //   TORCH_CHECK(t.defined(), name, " must be defined");
  //   TORCH_CHECK(t.device().is_privateuseone(), name, " must be on GCU");
  //   TORCH_CHECK(t.dtype() == torch::kInt32, name, " must have dtype int32");
  //   TORCH_CHECK(t.dim() == 1, name, " must be 1D");
  //   TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  //   TORCH_CHECK(t.numel() == expected_numel, name, " must have numel == ",
  //               expected_numel, ", but got ", t.numel());
  // };

  // // Defensive checks to avoid passing invalid tensors into topsfa and causing
  // // hard crashes.
  // check_i32_1d_tensor(seqused_k, "seqused_k", batch_size);
  // if (cu_seqlens_q_.has_value())
  // {
  //   check_i32_1d_tensor(cu_seqlens_q_.value(), "cu_seqlens_q", batch_size + 1);
  // }
  // if (cu_seqlens_k_.has_value())
  // {
  //   check_i32_1d_tensor(cu_seqlens_k_.value(), "cu_seqlens_k", batch_size + 1);
  // }
  // if (cu_seqlens_k_new_.has_value())
  // {
  //   check_i32_1d_tensor(cu_seqlens_k_new_.value(), "cu_seqlens_k_new",
  //                       batch_size + 1);
  // }
  // if (seqused_q_.has_value())
  // {
  //   check_i32_1d_tensor(seqused_q_.value(), "seqused_q", batch_size);
  // }
  // if (leftpad_k_.has_value())
  // {
  //   check_i32_1d_tensor(leftpad_k_.value(), "leftpad_k", batch_size);
  // }
  // if (page_size.has_value())
  // {
  //   TORCH_CHECK(page_size.value() > 0, "page_size must be > 0");
  // }
  // TORCH_CHECK(max_seqlen_q >= 0, "max_seqlen_q must be >= 0");
  // TORCH_CHECK(max_seqlen_k >= 0, "max_seqlen_k must be >= 0");
  // TORCH_CHECK(max_seqlen_k_new >= 0, "max_seqlen_k_new must be >= 0");
  // TORCH_CHECK(num_splits >= 0, "num_splits must be >= 0");

  TORCH_CHECK(qkv_dtype == at::ScalarType::Half || qkv_dtype == at::ScalarType::BFloat16 || qkv_dtype == at::ScalarType::Float8_e4m3fn,
              "FlashAttention only supports fp16, bf16, and fp8_e4m3 data type");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  if (window_size_left_mut >= max_seqlen_k - 1)
  {
    window_size_left_mut = -1;
  }
  if (window_size_right_mut >= max_seqlen_q - 1)
  {
    window_size_right_mut = -1;
  }
  // causal=true is the same as causal=false in this case
  if (max_seqlen_q == 1 && window_size_left_mut == -1 && window_size_right_mut == -1)
  {
    // Special case of hdim 128 where we want causal to have kBlockN=128, better for pagedKV and TMA
    if ((headdim <= 64 || headdim > 128) || !page_size.has_value())
    {
      is_causal_mut = false;
    }
  }
  if (is_causal_mut)
  {
    window_size_right_mut = 0;
  }

  if (window_size_left_mut < 0 && window_size_right_mut >= 0)
  {
    window_size_left_mut = max_seqlen_k - 1;
  }
  if (window_size_left_mut >= 0 && window_size_right_mut < 0)
  {
    window_size_right_mut = max_seqlen_q - 1;
  }
  topsatenDataType_t qkv_dtype_t;
  switch (qkv_dtype)
  {
  case at::ScalarType::Half:
    qkv_dtype_t = topsatenDataType_t::TOPSATEN_DATA_FP16;
    break;
  case at::ScalarType::BFloat16:
    qkv_dtype_t = topsatenDataType_t::TOPSATEN_DATA_BF16;
    break;
  case at::ScalarType::Float8_e4m3fn:
    qkv_dtype_t = topsatenDataType_t::TOPSATEN_DATA_FP8E4M3;
    break;
  default:
    TORCH_CHECK(false, "Unsupported qkv dtype for scheduler metadata");
  }

  auto opts = seqused_k.options().dtype(torch::kInt32);
  // (1024+(batch_size+1)*16) uses the currently required GCU buffer size (effective int32
  // count is 3 + 3 * batch_size; keep large buffer for existing consumers).
  // hack for vllm
  const int sche_meta_size = (qkv_dtype != at::ScalarType::Float8_e4m3fn) ? (1 + 4 * 4) : (1024 + (batch_size + 1) * 16);
  at::Tensor sche_metadata = torch::empty({sche_meta_size}, opts);

  if (qkv_dtype != at::ScalarType::Float8_e4m3fn)
  {
    sche_metadata.zero_();
    return sche_metadata;
  }

  topsatenTensor sche_metadata_tensor =
      makeTopsatenTensor(sche_metadata, topsatenDataType_t::TOPSATEN_DATA_I32);

  // topsaten_fa.h: all tensor args are const topsatenTensor&; unset ATen
  // optionals use empty topsatenTensor (same as makeOptionalConstTopsatenTensor
  // elsewhere in this file).
  topsatenTensor seqused_k_tensor =
      makeTopsatenTensor(seqused_k, topsatenDataType_t::TOPSATEN_DATA_I32);
  topsatenTensor cu_seqlens_q_tensor = makeOptionalConstTopsatenTensor(
      cu_seqlens_q_, topsatenDataType_t::TOPSATEN_DATA_I32);
  topsatenTensor cu_seqlens_k_tensor = makeOptionalConstTopsatenTensor(
      cu_seqlens_k_, topsatenDataType_t::TOPSATEN_DATA_I32);
  topsatenTensor cu_seqlens_k_new_tensor = makeOptionalConstTopsatenTensor(
      cu_seqlens_k_new_, topsatenDataType_t::TOPSATEN_DATA_I32);
  topsatenTensor seqused_q_tensor = makeOptionalConstTopsatenTensor(
      seqused_q_, topsatenDataType_t::TOPSATEN_DATA_I32);
  topsatenTensor leftpad_k_tensor = makeOptionalConstTopsatenTensor(
      leftpad_k_, topsatenDataType_t::TOPSATEN_DATA_I32);

  const int page_size_int =
      page_size.has_value() ? static_cast<int>(page_size.value()) : 0;
  const bool pack_gqa_bool = pack_gqa_.has_value() ? pack_gqa_.value() : false;

  auto stream = torch_gcu::getCurrentGCUStream();
  TOPSATEN_CHECK(topsfa::topsfaFlashAttnVarlenFwdGetScheMetadata(
      sche_metadata_tensor, batch_size, max_seqlen_q, max_seqlen_k, num_heads,
      num_heads_k, headdim, headdim_v, qkv_dtype_t, seqused_k_tensor,
      cu_seqlens_q_tensor, cu_seqlens_k_tensor, cu_seqlens_k_new_tensor,
      seqused_q_tensor, leftpad_k_tensor, page_size_int, max_seqlen_k_new,
      is_causal_mut, window_size_left_mut, window_size_right_mut, has_softcap,
      num_splits, pack_gqa_bool, sm_margin, stream));
  return sche_metadata;
}

PYBIND11_MODULE(flash_attn_gcu, m)
{
  m.doc() = "FlashAttention";
  m.def("fwd", &mha_fwd, "Forward pass");
  m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
  m.def("varlen_fwd_fp8kv", &mha_varlen_fwd_fp8KV, "Forward pass (variable length) with fp8 KV");
  m.def("bwd", &mha_bwd, "Backward pass");
  m.def("varlen_bwd", &mha_varlen_bwd, "Backward pass (variable length)");
  m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
  m.def("get_scheduler_metadata", &mha_fwd_get_scheduler_metadata, "Get scheduler metadata for varlen forward pass");
}
