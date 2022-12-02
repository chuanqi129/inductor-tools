import torch._inductor.overrides

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

# torch version: 1.14.0a0+gitb9e3838
# torch cuda version: None
# torch git version: b9e3838070b53d2481cc352598451a0f5ef88149


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_tensor_constant0', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant1', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant2', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant3', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant4', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant5', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant6', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant7', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant8', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant9', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant10', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant11', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant12', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant13', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant14', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant15', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant16', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant17', torch.randn([], dtype=torch.float32))
        self.register_buffer('_tensor_constant18', torch.randn([], dtype=torch.float32))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1):
        ne = torch.ops.aten.ne.Scalar(arg286_1, 1)
        unsqueeze = torch.ops.aten.unsqueeze.default(ne, -2);  ne = None
        ne_1 = torch.ops.aten.ne.Scalar(arg287_1, 1)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(ne_1, -2);  ne_1 = None
        ones = torch.ops.aten.ones.default([1, 21, 21], device = device(type='cpu'), pin_memory = False)
        alias = torch.ops.aten.alias.default(ones);  ones = None
        triu = torch.ops.aten.triu.default(alias, 1);  alias = None
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        sub = torch.ops.aten.sub.Tensor(lift_fresh_copy, triu);  lift_fresh_copy = triu = None
        _to_copy = torch.ops.aten._to_copy.default(sub, dtype = torch.bool);  sub = None
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(unsqueeze_1, _to_copy);  unsqueeze_1 = _to_copy = None
        embedding = torch.ops.aten.embedding.default(arg0_1, arg286_1, 1);  arg0_1 = arg286_1 = None
        slice_1 = torch.ops.aten.slice.Tensor(arg284_1, 0, 0, 9223372036854775807);  arg284_1 = None
        slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 22);  slice_1 = None
        clone = torch.ops.aten.clone.default(slice_2);  slice_2 = None
        alias_1 = torch.ops.aten.alias.default(clone);  clone = None
        alias_2 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
        add = torch.ops.aten.add.Tensor(embedding, alias_2);  embedding = alias_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        sqrt = torch.ops.aten.sqrt.default(add_1);  add_1 = None
        reciprocal = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        sub_1 = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub_1, reciprocal);  sub_1 = reciprocal = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        _mkl_linear = torch.ops.mkl._mkl_linear.default(add_2, arg4_1, arg3_1, None, 704);  arg4_1 = arg3_1 = None
        view = torch.ops.aten.view.default(_mkl_linear, [32, 22, 8, 64]);  _mkl_linear = None
        _mkl_linear_1 = torch.ops.mkl._mkl_linear.default(add_2, arg6_1, arg5_1, None, 704);  arg6_1 = arg5_1 = None
        view_1 = torch.ops.aten.view.default(_mkl_linear_1, [32, 22, 8, 64]);  _mkl_linear_1 = None
        _mkl_linear_2 = torch.ops.mkl._mkl_linear.default(add_2, arg8_1, arg7_1, None, 704);  arg8_1 = arg7_1 = None
        view_2 = torch.ops.aten.view.default(_mkl_linear_2, [32, 22, 8, 64]);  _mkl_linear_2 = None
        permute = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        permute_1 = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        permute_2 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        div = torch.ops.aten.div.Tensor(permute, 8.0);  permute = None
        permute_3 = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2]);  permute_1 = None
        expand = torch.ops.aten.expand.default(div, [32, 8, 22, 64]);  div = None
        clone_1 = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        _unsafe_view = torch.ops.aten._unsafe_view.default(clone_1, [256, 22, 64]);  clone_1 = None
        expand_1 = torch.ops.aten.expand.default(permute_3, [32, 8, 64, 22]);  permute_3 = None
        clone_2 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        _unsafe_view_1 = torch.ops.aten._unsafe_view.default(clone_2, [256, 64, 22]);  clone_2 = None
        bmm = torch.ops.aten.bmm.default(_unsafe_view, _unsafe_view_1);  _unsafe_view = _unsafe_view_1 = None
        _unsafe_view_2 = torch.ops.aten._unsafe_view.default(bmm, [32, 8, 22, 22]);  bmm = None
        eq = torch.ops.aten.eq.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None
        _tensor_constant1 = self._tensor_constant1
        lift_fresh_copy_1 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        where = torch.ops.aten.where.self(eq, lift_fresh_copy_1, _unsafe_view_2);  eq = lift_fresh_copy_1 = _unsafe_view_2 = None
        amax = torch.ops.aten.amax.default(where, [-1], True)
        sub_2 = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        expand_2 = torch.ops.aten.expand.default(div_1, [32, 8, 22, 22]);  div_1 = None
        view_3 = torch.ops.aten.view.default(expand_2, [256, 22, 22]);  expand_2 = None
        expand_3 = torch.ops.aten.expand.default(permute_2, [32, 8, 22, 64]);  permute_2 = None
        clone_3 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        _unsafe_view_3 = torch.ops.aten._unsafe_view.default(clone_3, [256, 22, 64]);  clone_3 = None
        bmm_1 = torch.ops.aten.bmm.default(view_3, _unsafe_view_3);  view_3 = _unsafe_view_3 = None
        _unsafe_view_4 = torch.ops.aten._unsafe_view.default(bmm_1, [32, 8, 22, 64]);  bmm_1 = None
        permute_4 = torch.ops.aten.permute.default(_unsafe_view_4, [0, 2, 1, 3]);  _unsafe_view_4 = None
        clone_4 = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_4 = torch.ops.aten.view.default(clone_4, [32, 22, -1]);  clone_4 = None
        _mkl_linear_3 = torch.ops.mkl._mkl_linear.default(view_4, arg10_1, arg9_1, None, 704);  view_4 = arg10_1 = arg9_1 = None
        add_ = torch.ops.aten.add_.Tensor(_mkl_linear_3, add_2);  _mkl_linear_3 = add_2 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_3 = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
        sqrt_1 = torch.ops.aten.sqrt.default(add_3);  add_3 = None
        reciprocal_1 = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_, getitem_3);  add_ = getitem_3 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_3, reciprocal_1);  sub_3 = reciprocal_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg11_1);  mul_2 = arg11_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_3, arg12_1);  mul_3 = arg12_1 = None
        _mkl_linear_4 = torch.ops.mkl._mkl_linear.default(add_4, arg15_1, arg13_1, arg14_1, 704);  arg15_1 = arg13_1 = arg14_1 = None
        relu = torch.ops.aten.relu.default(_mkl_linear_4);  _mkl_linear_4 = None
        _mkl_linear_5 = torch.ops.mkl._mkl_linear.default(relu, arg18_1, arg16_1, arg17_1, 704);  relu = arg18_1 = arg16_1 = arg17_1 = None
        add__1 = torch.ops.aten.add_.Tensor(_mkl_linear_5, add_4);  _mkl_linear_5 = add_4 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add__1, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
        sqrt_2 = torch.ops.aten.sqrt.default(add_5);  add_5 = None
        reciprocal_2 = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        sub_4 = torch.ops.aten.sub.Tensor(add__1, getitem_5);  add__1 = getitem_5 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_4, reciprocal_2);  sub_4 = reciprocal_2 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, arg19_1);  mul_4 = arg19_1 = None
        add_6 = torch.ops.aten.add.Tensor(mul_5, arg20_1);  mul_5 = arg20_1 = None
        _mkl_linear_6 = torch.ops.mkl._mkl_linear.default(add_6, arg22_1, arg21_1, None, 704);  arg22_1 = arg21_1 = None
        view_5 = torch.ops.aten.view.default(_mkl_linear_6, [32, 22, 8, 64]);  _mkl_linear_6 = None
        _mkl_linear_7 = torch.ops.mkl._mkl_linear.default(add_6, arg24_1, arg23_1, None, 704);  arg24_1 = arg23_1 = None
        view_6 = torch.ops.aten.view.default(_mkl_linear_7, [32, 22, 8, 64]);  _mkl_linear_7 = None
        _mkl_linear_8 = torch.ops.mkl._mkl_linear.default(add_6, arg26_1, arg25_1, None, 704);  arg26_1 = arg25_1 = None
        view_7 = torch.ops.aten.view.default(_mkl_linear_8, [32, 22, 8, 64]);  _mkl_linear_8 = None
        permute_5 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        permute_6 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_7 = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        div_2 = torch.ops.aten.div.Tensor(permute_5, 8.0);  permute_5 = None
        permute_8 = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2]);  permute_6 = None
        expand_4 = torch.ops.aten.expand.default(div_2, [32, 8, 22, 64]);  div_2 = None
        clone_5 = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        _unsafe_view_5 = torch.ops.aten._unsafe_view.default(clone_5, [256, 22, 64]);  clone_5 = None
        expand_5 = torch.ops.aten.expand.default(permute_8, [32, 8, 64, 22]);  permute_8 = None
        clone_6 = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        _unsafe_view_6 = torch.ops.aten._unsafe_view.default(clone_6, [256, 64, 22]);  clone_6 = None
        bmm_2 = torch.ops.aten.bmm.default(_unsafe_view_5, _unsafe_view_6);  _unsafe_view_5 = _unsafe_view_6 = None
        _unsafe_view_7 = torch.ops.aten._unsafe_view.default(bmm_2, [32, 8, 22, 22]);  bmm_2 = None
        eq_1 = torch.ops.aten.eq.Scalar(unsqueeze_3, 0);  unsqueeze_3 = None
        _tensor_constant2 = self._tensor_constant2
        lift_fresh_copy_2 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        where_1 = torch.ops.aten.where.self(eq_1, lift_fresh_copy_2, _unsafe_view_7);  eq_1 = lift_fresh_copy_2 = _unsafe_view_7 = None
        amax_1 = torch.ops.aten.amax.default(where_1, [-1], True)
        sub_5 = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        expand_6 = torch.ops.aten.expand.default(div_3, [32, 8, 22, 22]);  div_3 = None
        view_8 = torch.ops.aten.view.default(expand_6, [256, 22, 22]);  expand_6 = None
        expand_7 = torch.ops.aten.expand.default(permute_7, [32, 8, 22, 64]);  permute_7 = None
        clone_7 = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        _unsafe_view_8 = torch.ops.aten._unsafe_view.default(clone_7, [256, 22, 64]);  clone_7 = None
        bmm_3 = torch.ops.aten.bmm.default(view_8, _unsafe_view_8);  view_8 = _unsafe_view_8 = None
        _unsafe_view_9 = torch.ops.aten._unsafe_view.default(bmm_3, [32, 8, 22, 64]);  bmm_3 = None
        permute_9 = torch.ops.aten.permute.default(_unsafe_view_9, [0, 2, 1, 3]);  _unsafe_view_9 = None
        clone_8 = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        view_9 = torch.ops.aten.view.default(clone_8, [32, 22, -1]);  clone_8 = None
        _mkl_linear_9 = torch.ops.mkl._mkl_linear.default(view_9, arg28_1, arg27_1, None, 704);  view_9 = arg28_1 = arg27_1 = None
        add__2 = torch.ops.aten.add_.Tensor(_mkl_linear_9, add_6);  _mkl_linear_9 = add_6 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add__2, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_7 = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
        sqrt_3 = torch.ops.aten.sqrt.default(add_7);  add_7 = None
        reciprocal_3 = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        sub_6 = torch.ops.aten.sub.Tensor(add__2, getitem_7);  add__2 = getitem_7 = None
        mul_6 = torch.ops.aten.mul.Tensor(sub_6, reciprocal_3);  sub_6 = reciprocal_3 = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, arg29_1);  mul_6 = arg29_1 = None
        add_8 = torch.ops.aten.add.Tensor(mul_7, arg30_1);  mul_7 = arg30_1 = None
        _mkl_linear_10 = torch.ops.mkl._mkl_linear.default(add_8, arg33_1, arg31_1, arg32_1, 704);  arg33_1 = arg31_1 = arg32_1 = None
        relu_1 = torch.ops.aten.relu.default(_mkl_linear_10);  _mkl_linear_10 = None
        _mkl_linear_11 = torch.ops.mkl._mkl_linear.default(relu_1, arg36_1, arg34_1, arg35_1, 704);  relu_1 = arg36_1 = arg34_1 = arg35_1 = None
        add__3 = torch.ops.aten.add_.Tensor(_mkl_linear_11, add_8);  _mkl_linear_11 = add_8 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add__3, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
        sqrt_4 = torch.ops.aten.sqrt.default(add_9);  add_9 = None
        reciprocal_4 = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        sub_7 = torch.ops.aten.sub.Tensor(add__3, getitem_9);  add__3 = getitem_9 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_7, reciprocal_4);  sub_7 = reciprocal_4 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, arg37_1);  mul_8 = arg37_1 = None
        add_10 = torch.ops.aten.add.Tensor(mul_9, arg38_1);  mul_9 = arg38_1 = None
        _mkl_linear_12 = torch.ops.mkl._mkl_linear.default(add_10, arg40_1, arg39_1, None, 704);  arg40_1 = arg39_1 = None
        view_10 = torch.ops.aten.view.default(_mkl_linear_12, [32, 22, 8, 64]);  _mkl_linear_12 = None
        _mkl_linear_13 = torch.ops.mkl._mkl_linear.default(add_10, arg42_1, arg41_1, None, 704);  arg42_1 = arg41_1 = None
        view_11 = torch.ops.aten.view.default(_mkl_linear_13, [32, 22, 8, 64]);  _mkl_linear_13 = None
        _mkl_linear_14 = torch.ops.mkl._mkl_linear.default(add_10, arg44_1, arg43_1, None, 704);  arg44_1 = arg43_1 = None
        view_12 = torch.ops.aten.view.default(_mkl_linear_14, [32, 22, 8, 64]);  _mkl_linear_14 = None
        permute_10 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        permute_11 = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        permute_12 = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        div_4 = torch.ops.aten.div.Tensor(permute_10, 8.0);  permute_10 = None
        permute_13 = torch.ops.aten.permute.default(permute_11, [0, 1, 3, 2]);  permute_11 = None
        expand_8 = torch.ops.aten.expand.default(div_4, [32, 8, 22, 64]);  div_4 = None
        clone_9 = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        _unsafe_view_10 = torch.ops.aten._unsafe_view.default(clone_9, [256, 22, 64]);  clone_9 = None
        expand_9 = torch.ops.aten.expand.default(permute_13, [32, 8, 64, 22]);  permute_13 = None
        clone_10 = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        _unsafe_view_11 = torch.ops.aten._unsafe_view.default(clone_10, [256, 64, 22]);  clone_10 = None
        bmm_4 = torch.ops.aten.bmm.default(_unsafe_view_10, _unsafe_view_11);  _unsafe_view_10 = _unsafe_view_11 = None
        _unsafe_view_12 = torch.ops.aten._unsafe_view.default(bmm_4, [32, 8, 22, 22]);  bmm_4 = None
        eq_2 = torch.ops.aten.eq.Scalar(unsqueeze_4, 0);  unsqueeze_4 = None
        _tensor_constant3 = self._tensor_constant3
        lift_fresh_copy_3 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
        where_2 = torch.ops.aten.where.self(eq_2, lift_fresh_copy_3, _unsafe_view_12);  eq_2 = lift_fresh_copy_3 = _unsafe_view_12 = None
        amax_2 = torch.ops.aten.amax.default(where_2, [-1], True)
        sub_8 = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        expand_10 = torch.ops.aten.expand.default(div_5, [32, 8, 22, 22]);  div_5 = None
        view_13 = torch.ops.aten.view.default(expand_10, [256, 22, 22]);  expand_10 = None
        expand_11 = torch.ops.aten.expand.default(permute_12, [32, 8, 22, 64]);  permute_12 = None
        clone_11 = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        _unsafe_view_13 = torch.ops.aten._unsafe_view.default(clone_11, [256, 22, 64]);  clone_11 = None
        bmm_5 = torch.ops.aten.bmm.default(view_13, _unsafe_view_13);  view_13 = _unsafe_view_13 = None
        _unsafe_view_14 = torch.ops.aten._unsafe_view.default(bmm_5, [32, 8, 22, 64]);  bmm_5 = None
        permute_14 = torch.ops.aten.permute.default(_unsafe_view_14, [0, 2, 1, 3]);  _unsafe_view_14 = None
        clone_12 = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_14 = torch.ops.aten.view.default(clone_12, [32, 22, -1]);  clone_12 = None
        _mkl_linear_15 = torch.ops.mkl._mkl_linear.default(view_14, arg46_1, arg45_1, None, 704);  view_14 = arg46_1 = arg45_1 = None
        add__4 = torch.ops.aten.add_.Tensor(_mkl_linear_15, add_10);  _mkl_linear_15 = add_10 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add__4, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_11 = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
        sqrt_5 = torch.ops.aten.sqrt.default(add_11);  add_11 = None
        reciprocal_5 = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        sub_9 = torch.ops.aten.sub.Tensor(add__4, getitem_11);  add__4 = getitem_11 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_9, reciprocal_5);  sub_9 = reciprocal_5 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg47_1);  mul_10 = arg47_1 = None
        add_12 = torch.ops.aten.add.Tensor(mul_11, arg48_1);  mul_11 = arg48_1 = None
        _mkl_linear_16 = torch.ops.mkl._mkl_linear.default(add_12, arg51_1, arg49_1, arg50_1, 704);  arg51_1 = arg49_1 = arg50_1 = None
        relu_2 = torch.ops.aten.relu.default(_mkl_linear_16);  _mkl_linear_16 = None
        _mkl_linear_17 = torch.ops.mkl._mkl_linear.default(relu_2, arg54_1, arg52_1, arg53_1, 704);  relu_2 = arg54_1 = arg52_1 = arg53_1 = None
        add__5 = torch.ops.aten.add_.Tensor(_mkl_linear_17, add_12);  _mkl_linear_17 = add_12 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add__5, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
        sqrt_6 = torch.ops.aten.sqrt.default(add_13);  add_13 = None
        reciprocal_6 = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        sub_10 = torch.ops.aten.sub.Tensor(add__5, getitem_13);  add__5 = getitem_13 = None
        mul_12 = torch.ops.aten.mul.Tensor(sub_10, reciprocal_6);  sub_10 = reciprocal_6 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, arg55_1);  mul_12 = arg55_1 = None
        add_14 = torch.ops.aten.add.Tensor(mul_13, arg56_1);  mul_13 = arg56_1 = None
        _mkl_linear_18 = torch.ops.mkl._mkl_linear.default(add_14, arg58_1, arg57_1, None, 704);  arg58_1 = arg57_1 = None
        view_15 = torch.ops.aten.view.default(_mkl_linear_18, [32, 22, 8, 64]);  _mkl_linear_18 = None
        _mkl_linear_19 = torch.ops.mkl._mkl_linear.default(add_14, arg60_1, arg59_1, None, 704);  arg60_1 = arg59_1 = None
        view_16 = torch.ops.aten.view.default(_mkl_linear_19, [32, 22, 8, 64]);  _mkl_linear_19 = None
        _mkl_linear_20 = torch.ops.mkl._mkl_linear.default(add_14, arg62_1, arg61_1, None, 704);  arg62_1 = arg61_1 = None
        view_17 = torch.ops.aten.view.default(_mkl_linear_20, [32, 22, 8, 64]);  _mkl_linear_20 = None
        permute_15 = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        permute_16 = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        permute_17 = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        div_6 = torch.ops.aten.div.Tensor(permute_15, 8.0);  permute_15 = None
        permute_18 = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2]);  permute_16 = None
        expand_12 = torch.ops.aten.expand.default(div_6, [32, 8, 22, 64]);  div_6 = None
        clone_13 = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        _unsafe_view_15 = torch.ops.aten._unsafe_view.default(clone_13, [256, 22, 64]);  clone_13 = None
        expand_13 = torch.ops.aten.expand.default(permute_18, [32, 8, 64, 22]);  permute_18 = None
        clone_14 = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        _unsafe_view_16 = torch.ops.aten._unsafe_view.default(clone_14, [256, 64, 22]);  clone_14 = None
        bmm_6 = torch.ops.aten.bmm.default(_unsafe_view_15, _unsafe_view_16);  _unsafe_view_15 = _unsafe_view_16 = None
        _unsafe_view_17 = torch.ops.aten._unsafe_view.default(bmm_6, [32, 8, 22, 22]);  bmm_6 = None
        eq_3 = torch.ops.aten.eq.Scalar(unsqueeze_5, 0);  unsqueeze_5 = None
        _tensor_constant4 = self._tensor_constant4
        lift_fresh_copy_4 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
        where_3 = torch.ops.aten.where.self(eq_3, lift_fresh_copy_4, _unsafe_view_17);  eq_3 = lift_fresh_copy_4 = _unsafe_view_17 = None
        amax_3 = torch.ops.aten.amax.default(where_3, [-1], True)
        sub_11 = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        expand_14 = torch.ops.aten.expand.default(div_7, [32, 8, 22, 22]);  div_7 = None
        view_18 = torch.ops.aten.view.default(expand_14, [256, 22, 22]);  expand_14 = None
        expand_15 = torch.ops.aten.expand.default(permute_17, [32, 8, 22, 64]);  permute_17 = None
        clone_15 = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        _unsafe_view_18 = torch.ops.aten._unsafe_view.default(clone_15, [256, 22, 64]);  clone_15 = None
        bmm_7 = torch.ops.aten.bmm.default(view_18, _unsafe_view_18);  view_18 = _unsafe_view_18 = None
        _unsafe_view_19 = torch.ops.aten._unsafe_view.default(bmm_7, [32, 8, 22, 64]);  bmm_7 = None
        permute_19 = torch.ops.aten.permute.default(_unsafe_view_19, [0, 2, 1, 3]);  _unsafe_view_19 = None
        clone_16 = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
        view_19 = torch.ops.aten.view.default(clone_16, [32, 22, -1]);  clone_16 = None
        _mkl_linear_21 = torch.ops.mkl._mkl_linear.default(view_19, arg64_1, arg63_1, None, 704);  view_19 = arg64_1 = arg63_1 = None
        add__6 = torch.ops.aten.add_.Tensor(_mkl_linear_21, add_14);  _mkl_linear_21 = add_14 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add__6, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_15 = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        sqrt_7 = torch.ops.aten.sqrt.default(add_15);  add_15 = None
        reciprocal_7 = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        sub_12 = torch.ops.aten.sub.Tensor(add__6, getitem_15);  add__6 = getitem_15 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_12, reciprocal_7);  sub_12 = reciprocal_7 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, arg65_1);  mul_14 = arg65_1 = None
        add_16 = torch.ops.aten.add.Tensor(mul_15, arg66_1);  mul_15 = arg66_1 = None
        _mkl_linear_22 = torch.ops.mkl._mkl_linear.default(add_16, arg69_1, arg67_1, arg68_1, 704);  arg69_1 = arg67_1 = arg68_1 = None
        relu_3 = torch.ops.aten.relu.default(_mkl_linear_22);  _mkl_linear_22 = None
        _mkl_linear_23 = torch.ops.mkl._mkl_linear.default(relu_3, arg72_1, arg70_1, arg71_1, 704);  relu_3 = arg72_1 = arg70_1 = arg71_1 = None
        add__7 = torch.ops.aten.add_.Tensor(_mkl_linear_23, add_16);  _mkl_linear_23 = add_16 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add__7, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_17 = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        sqrt_8 = torch.ops.aten.sqrt.default(add_17);  add_17 = None
        reciprocal_8 = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        sub_13 = torch.ops.aten.sub.Tensor(add__7, getitem_17);  add__7 = getitem_17 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_13, reciprocal_8);  sub_13 = reciprocal_8 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, arg73_1);  mul_16 = arg73_1 = None
        add_18 = torch.ops.aten.add.Tensor(mul_17, arg74_1);  mul_17 = arg74_1 = None
        _mkl_linear_24 = torch.ops.mkl._mkl_linear.default(add_18, arg76_1, arg75_1, None, 704);  arg76_1 = arg75_1 = None
        view_20 = torch.ops.aten.view.default(_mkl_linear_24, [32, 22, 8, 64]);  _mkl_linear_24 = None
        _mkl_linear_25 = torch.ops.mkl._mkl_linear.default(add_18, arg78_1, arg77_1, None, 704);  arg78_1 = arg77_1 = None
        view_21 = torch.ops.aten.view.default(_mkl_linear_25, [32, 22, 8, 64]);  _mkl_linear_25 = None
        _mkl_linear_26 = torch.ops.mkl._mkl_linear.default(add_18, arg80_1, arg79_1, None, 704);  arg80_1 = arg79_1 = None
        view_22 = torch.ops.aten.view.default(_mkl_linear_26, [32, 22, 8, 64]);  _mkl_linear_26 = None
        permute_20 = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        permute_21 = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        permute_22 = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        div_8 = torch.ops.aten.div.Tensor(permute_20, 8.0);  permute_20 = None
        permute_23 = torch.ops.aten.permute.default(permute_21, [0, 1, 3, 2]);  permute_21 = None
        expand_16 = torch.ops.aten.expand.default(div_8, [32, 8, 22, 64]);  div_8 = None
        clone_17 = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        _unsafe_view_20 = torch.ops.aten._unsafe_view.default(clone_17, [256, 22, 64]);  clone_17 = None
        expand_17 = torch.ops.aten.expand.default(permute_23, [32, 8, 64, 22]);  permute_23 = None
        clone_18 = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        _unsafe_view_21 = torch.ops.aten._unsafe_view.default(clone_18, [256, 64, 22]);  clone_18 = None
        bmm_8 = torch.ops.aten.bmm.default(_unsafe_view_20, _unsafe_view_21);  _unsafe_view_20 = _unsafe_view_21 = None
        _unsafe_view_22 = torch.ops.aten._unsafe_view.default(bmm_8, [32, 8, 22, 22]);  bmm_8 = None
        eq_4 = torch.ops.aten.eq.Scalar(unsqueeze_6, 0);  unsqueeze_6 = None
        _tensor_constant5 = self._tensor_constant5
        lift_fresh_copy_5 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
        where_4 = torch.ops.aten.where.self(eq_4, lift_fresh_copy_5, _unsafe_view_22);  eq_4 = lift_fresh_copy_5 = _unsafe_view_22 = None
        amax_4 = torch.ops.aten.amax.default(where_4, [-1], True)
        sub_14 = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_9 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        expand_18 = torch.ops.aten.expand.default(div_9, [32, 8, 22, 22]);  div_9 = None
        view_23 = torch.ops.aten.view.default(expand_18, [256, 22, 22]);  expand_18 = None
        expand_19 = torch.ops.aten.expand.default(permute_22, [32, 8, 22, 64]);  permute_22 = None
        clone_19 = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        _unsafe_view_23 = torch.ops.aten._unsafe_view.default(clone_19, [256, 22, 64]);  clone_19 = None
        bmm_9 = torch.ops.aten.bmm.default(view_23, _unsafe_view_23);  view_23 = _unsafe_view_23 = None
        _unsafe_view_24 = torch.ops.aten._unsafe_view.default(bmm_9, [32, 8, 22, 64]);  bmm_9 = None
        permute_24 = torch.ops.aten.permute.default(_unsafe_view_24, [0, 2, 1, 3]);  _unsafe_view_24 = None
        clone_20 = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_24 = torch.ops.aten.view.default(clone_20, [32, 22, -1]);  clone_20 = None
        _mkl_linear_27 = torch.ops.mkl._mkl_linear.default(view_24, arg82_1, arg81_1, None, 704);  view_24 = arg82_1 = arg81_1 = None
        add__8 = torch.ops.aten.add_.Tensor(_mkl_linear_27, add_18);  _mkl_linear_27 = add_18 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add__8, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_19 = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
        sqrt_9 = torch.ops.aten.sqrt.default(add_19);  add_19 = None
        reciprocal_9 = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        sub_15 = torch.ops.aten.sub.Tensor(add__8, getitem_19);  add__8 = getitem_19 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_15, reciprocal_9);  sub_15 = reciprocal_9 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, arg83_1);  mul_18 = arg83_1 = None
        add_20 = torch.ops.aten.add.Tensor(mul_19, arg84_1);  mul_19 = arg84_1 = None
        _mkl_linear_28 = torch.ops.mkl._mkl_linear.default(add_20, arg87_1, arg85_1, arg86_1, 704);  arg87_1 = arg85_1 = arg86_1 = None
        relu_4 = torch.ops.aten.relu.default(_mkl_linear_28);  _mkl_linear_28 = None
        _mkl_linear_29 = torch.ops.mkl._mkl_linear.default(relu_4, arg90_1, arg88_1, arg89_1, 704);  relu_4 = arg90_1 = arg88_1 = arg89_1 = None
        add__9 = torch.ops.aten.add_.Tensor(_mkl_linear_29, add_20);  _mkl_linear_29 = add_20 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add__9, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
        sqrt_10 = torch.ops.aten.sqrt.default(add_21);  add_21 = None
        reciprocal_10 = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        sub_16 = torch.ops.aten.sub.Tensor(add__9, getitem_21);  add__9 = getitem_21 = None
        mul_20 = torch.ops.aten.mul.Tensor(sub_16, reciprocal_10);  sub_16 = reciprocal_10 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, arg91_1);  mul_20 = arg91_1 = None
        add_22 = torch.ops.aten.add.Tensor(mul_21, arg92_1);  mul_21 = arg92_1 = None
        _mkl_linear_30 = torch.ops.mkl._mkl_linear.default(add_22, arg94_1, arg93_1, None, 704);  arg94_1 = arg93_1 = None
        view_25 = torch.ops.aten.view.default(_mkl_linear_30, [32, 22, 8, 64]);  _mkl_linear_30 = None
        _mkl_linear_31 = torch.ops.mkl._mkl_linear.default(add_22, arg96_1, arg95_1, None, 704);  arg96_1 = arg95_1 = None
        view_26 = torch.ops.aten.view.default(_mkl_linear_31, [32, 22, 8, 64]);  _mkl_linear_31 = None
        _mkl_linear_32 = torch.ops.mkl._mkl_linear.default(add_22, arg98_1, arg97_1, None, 704);  arg98_1 = arg97_1 = None
        view_27 = torch.ops.aten.view.default(_mkl_linear_32, [32, 22, 8, 64]);  _mkl_linear_32 = None
        permute_25 = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        permute_26 = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        permute_27 = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        div_10 = torch.ops.aten.div.Tensor(permute_25, 8.0);  permute_25 = None
        permute_28 = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        expand_20 = torch.ops.aten.expand.default(div_10, [32, 8, 22, 64]);  div_10 = None
        clone_21 = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        _unsafe_view_25 = torch.ops.aten._unsafe_view.default(clone_21, [256, 22, 64]);  clone_21 = None
        expand_21 = torch.ops.aten.expand.default(permute_28, [32, 8, 64, 22]);  permute_28 = None
        clone_22 = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        _unsafe_view_26 = torch.ops.aten._unsafe_view.default(clone_22, [256, 64, 22]);  clone_22 = None
        bmm_10 = torch.ops.aten.bmm.default(_unsafe_view_25, _unsafe_view_26);  _unsafe_view_25 = _unsafe_view_26 = None
        _unsafe_view_27 = torch.ops.aten._unsafe_view.default(bmm_10, [32, 8, 22, 22]);  bmm_10 = None
        eq_5 = torch.ops.aten.eq.Scalar(unsqueeze_7, 0);  unsqueeze_7 = None
        _tensor_constant6 = self._tensor_constant6
        lift_fresh_copy_6 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
        where_5 = torch.ops.aten.where.self(eq_5, lift_fresh_copy_6, _unsafe_view_27);  eq_5 = lift_fresh_copy_6 = _unsafe_view_27 = None
        amax_5 = torch.ops.aten.amax.default(where_5, [-1], True)
        sub_17 = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_11 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        expand_22 = torch.ops.aten.expand.default(div_11, [32, 8, 22, 22]);  div_11 = None
        view_28 = torch.ops.aten.view.default(expand_22, [256, 22, 22]);  expand_22 = None
        expand_23 = torch.ops.aten.expand.default(permute_27, [32, 8, 22, 64]);  permute_27 = None
        clone_23 = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        _unsafe_view_28 = torch.ops.aten._unsafe_view.default(clone_23, [256, 22, 64]);  clone_23 = None
        bmm_11 = torch.ops.aten.bmm.default(view_28, _unsafe_view_28);  view_28 = _unsafe_view_28 = None
        _unsafe_view_29 = torch.ops.aten._unsafe_view.default(bmm_11, [32, 8, 22, 64]);  bmm_11 = None
        permute_29 = torch.ops.aten.permute.default(_unsafe_view_29, [0, 2, 1, 3]);  _unsafe_view_29 = None
        clone_24 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_29 = torch.ops.aten.view.default(clone_24, [32, 22, -1]);  clone_24 = None
        _mkl_linear_33 = torch.ops.mkl._mkl_linear.default(view_29, arg100_1, arg99_1, None, 704);  view_29 = arg100_1 = arg99_1 = None
        add__10 = torch.ops.aten.add_.Tensor(_mkl_linear_33, add_22);  _mkl_linear_33 = add_22 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add__10, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_23 = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
        sqrt_11 = torch.ops.aten.sqrt.default(add_23);  add_23 = None
        reciprocal_11 = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        sub_18 = torch.ops.aten.sub.Tensor(add__10, getitem_23);  add__10 = getitem_23 = None
        mul_22 = torch.ops.aten.mul.Tensor(sub_18, reciprocal_11);  sub_18 = reciprocal_11 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, arg101_1);  mul_22 = arg101_1 = None
        add_24 = torch.ops.aten.add.Tensor(mul_23, arg102_1);  mul_23 = arg102_1 = None
        _mkl_linear_34 = torch.ops.mkl._mkl_linear.default(add_24, arg105_1, arg103_1, arg104_1, 704);  arg105_1 = arg103_1 = arg104_1 = None
        relu_5 = torch.ops.aten.relu.default(_mkl_linear_34);  _mkl_linear_34 = None
        _mkl_linear_35 = torch.ops.mkl._mkl_linear.default(relu_5, arg108_1, arg106_1, arg107_1, 704);  relu_5 = arg108_1 = arg106_1 = arg107_1 = None
        add__11 = torch.ops.aten.add_.Tensor(_mkl_linear_35, add_24);  _mkl_linear_35 = add_24 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add__11, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_25 = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
        sqrt_12 = torch.ops.aten.sqrt.default(add_25);  add_25 = None
        reciprocal_12 = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        sub_19 = torch.ops.aten.sub.Tensor(add__11, getitem_25);  add__11 = getitem_25 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_19, reciprocal_12);  sub_19 = reciprocal_12 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, arg109_1);  mul_24 = arg109_1 = None
        add_26 = torch.ops.aten.add.Tensor(mul_25, arg110_1);  mul_25 = arg110_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg111_1, arg287_1, 1);  arg111_1 = arg287_1 = None
        slice_3 = torch.ops.aten.slice.Tensor(arg285_1, 0, 0, 9223372036854775807);  arg285_1 = None
        slice_4 = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 21);  slice_3 = None
        clone_25 = torch.ops.aten.clone.default(slice_4);  slice_4 = None
        alias_3 = torch.ops.aten.alias.default(clone_25);  clone_25 = None
        alias_4 = torch.ops.aten.alias.default(alias_3);  alias_3 = None
        add_27 = torch.ops.aten.add.Tensor(embedding_1, alias_4);  embedding_1 = alias_4 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_28 = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
        sqrt_13 = torch.ops.aten.sqrt.default(add_28);  add_28 = None
        reciprocal_13 = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_27, getitem_27);  add_27 = getitem_27 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_20, reciprocal_13);  sub_20 = reciprocal_13 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, arg112_1);  mul_26 = arg112_1 = None
        add_29 = torch.ops.aten.add.Tensor(mul_27, arg113_1);  mul_27 = arg113_1 = None
        _mkl_linear_36 = torch.ops.mkl._mkl_linear.default(add_29, arg115_1, arg114_1, None, 672);  arg115_1 = arg114_1 = None
        view_30 = torch.ops.aten.view.default(_mkl_linear_36, [32, 21, 8, 64]);  _mkl_linear_36 = None
        _mkl_linear_37 = torch.ops.mkl._mkl_linear.default(add_29, arg117_1, arg116_1, None, 672);  arg117_1 = arg116_1 = None
        view_31 = torch.ops.aten.view.default(_mkl_linear_37, [32, 21, 8, 64]);  _mkl_linear_37 = None
        _mkl_linear_38 = torch.ops.mkl._mkl_linear.default(add_29, arg119_1, arg118_1, None, 672);  arg119_1 = arg118_1 = None
        view_32 = torch.ops.aten.view.default(_mkl_linear_38, [32, 21, 8, 64]);  _mkl_linear_38 = None
        permute_30 = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        permute_31 = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
        permute_32 = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(bitwise_and, 1)
        div_12 = torch.ops.aten.div.Tensor(permute_30, 8.0);  permute_30 = None
        permute_33 = torch.ops.aten.permute.default(permute_31, [0, 1, 3, 2]);  permute_31 = None
        expand_24 = torch.ops.aten.expand.default(div_12, [32, 8, 21, 64]);  div_12 = None
        clone_26 = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        _unsafe_view_30 = torch.ops.aten._unsafe_view.default(clone_26, [256, 21, 64]);  clone_26 = None
        expand_25 = torch.ops.aten.expand.default(permute_33, [32, 8, 64, 21]);  permute_33 = None
        clone_27 = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        _unsafe_view_31 = torch.ops.aten._unsafe_view.default(clone_27, [256, 64, 21]);  clone_27 = None
        bmm_12 = torch.ops.aten.bmm.default(_unsafe_view_30, _unsafe_view_31);  _unsafe_view_30 = _unsafe_view_31 = None
        _unsafe_view_32 = torch.ops.aten._unsafe_view.default(bmm_12, [32, 8, 21, 21]);  bmm_12 = None
        eq_6 = torch.ops.aten.eq.Scalar(unsqueeze_8, 0);  unsqueeze_8 = None
        _tensor_constant7 = self._tensor_constant7
        lift_fresh_copy_7 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant7);  _tensor_constant7 = None
        where_6 = torch.ops.aten.where.self(eq_6, lift_fresh_copy_7, _unsafe_view_32);  eq_6 = lift_fresh_copy_7 = _unsafe_view_32 = None
        clone_28 = torch.ops.aten.clone.default(where_6, memory_format = torch.contiguous_format);  where_6 = None
        amax_6 = torch.ops.aten.amax.default(clone_28, [-1], True)
        sub_21 = torch.ops.aten.sub.Tensor(clone_28, amax_6);  clone_28 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_13 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        expand_26 = torch.ops.aten.expand.default(div_13, [32, 8, 21, 21]);  div_13 = None
        view_33 = torch.ops.aten.view.default(expand_26, [256, 21, 21]);  expand_26 = None
        expand_27 = torch.ops.aten.expand.default(permute_32, [32, 8, 21, 64]);  permute_32 = None
        clone_29 = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        _unsafe_view_33 = torch.ops.aten._unsafe_view.default(clone_29, [256, 21, 64]);  clone_29 = None
        bmm_13 = torch.ops.aten.bmm.default(view_33, _unsafe_view_33);  view_33 = _unsafe_view_33 = None
        _unsafe_view_34 = torch.ops.aten._unsafe_view.default(bmm_13, [32, 8, 21, 64]);  bmm_13 = None
        permute_34 = torch.ops.aten.permute.default(_unsafe_view_34, [0, 2, 1, 3]);  _unsafe_view_34 = None
        clone_30 = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_34 = torch.ops.aten.view.default(clone_30, [32, 21, -1]);  clone_30 = None
        _mkl_linear_39 = torch.ops.mkl._mkl_linear.default(view_34, arg121_1, arg120_1, None, 672);  view_34 = arg121_1 = arg120_1 = None
        add__12 = torch.ops.aten.add_.Tensor(_mkl_linear_39, add_29);  _mkl_linear_39 = add_29 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add__12, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
        sqrt_14 = torch.ops.aten.sqrt.default(add_30);  add_30 = None
        reciprocal_14 = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        sub_22 = torch.ops.aten.sub.Tensor(add__12, getitem_29);  add__12 = getitem_29 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_22, reciprocal_14);  sub_22 = reciprocal_14 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, arg122_1);  mul_28 = arg122_1 = None
        add_31 = torch.ops.aten.add.Tensor(mul_29, arg123_1);  mul_29 = arg123_1 = None
        _mkl_linear_40 = torch.ops.mkl._mkl_linear.default(add_31, arg125_1, arg124_1, None, 672);  arg125_1 = arg124_1 = None
        view_35 = torch.ops.aten.view.default(_mkl_linear_40, [32, 21, 8, 64]);  _mkl_linear_40 = None
        _mkl_linear_41 = torch.ops.mkl._mkl_linear.default(add_26, arg127_1, arg126_1, None, 704);  arg127_1 = arg126_1 = None
        view_36 = torch.ops.aten.view.default(_mkl_linear_41, [32, 22, 8, 64]);  _mkl_linear_41 = None
        _mkl_linear_42 = torch.ops.mkl._mkl_linear.default(add_26, arg129_1, arg128_1, None, 704);  arg129_1 = arg128_1 = None
        view_37 = torch.ops.aten.view.default(_mkl_linear_42, [32, 22, 8, 64]);  _mkl_linear_42 = None
        permute_35 = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
        permute_36 = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        permute_37 = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        div_14 = torch.ops.aten.div.Tensor(permute_35, 8.0);  permute_35 = None
        permute_38 = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        expand_28 = torch.ops.aten.expand.default(div_14, [32, 8, 21, 64]);  div_14 = None
        clone_31 = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        _unsafe_view_35 = torch.ops.aten._unsafe_view.default(clone_31, [256, 21, 64]);  clone_31 = None
        expand_29 = torch.ops.aten.expand.default(permute_38, [32, 8, 64, 22]);  permute_38 = None
        clone_32 = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        _unsafe_view_36 = torch.ops.aten._unsafe_view.default(clone_32, [256, 64, 22]);  clone_32 = None
        bmm_14 = torch.ops.aten.bmm.default(_unsafe_view_35, _unsafe_view_36);  _unsafe_view_35 = _unsafe_view_36 = None
        _unsafe_view_37 = torch.ops.aten._unsafe_view.default(bmm_14, [32, 8, 21, 22]);  bmm_14 = None
        eq_7 = torch.ops.aten.eq.Scalar(unsqueeze_9, 0);  unsqueeze_9 = None
        _tensor_constant8 = self._tensor_constant8
        lift_fresh_copy_8 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
        where_7 = torch.ops.aten.where.self(eq_7, lift_fresh_copy_8, _unsafe_view_37);  eq_7 = lift_fresh_copy_8 = _unsafe_view_37 = None
        amax_7 = torch.ops.aten.amax.default(where_7, [-1], True)
        sub_23 = torch.ops.aten.sub.Tensor(where_7, amax_7);  where_7 = amax_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_15 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        expand_30 = torch.ops.aten.expand.default(div_15, [32, 8, 21, 22]);  div_15 = None
        view_38 = torch.ops.aten.view.default(expand_30, [256, 21, 22]);  expand_30 = None
        expand_31 = torch.ops.aten.expand.default(permute_37, [32, 8, 22, 64]);  permute_37 = None
        clone_33 = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        _unsafe_view_38 = torch.ops.aten._unsafe_view.default(clone_33, [256, 22, 64]);  clone_33 = None
        bmm_15 = torch.ops.aten.bmm.default(view_38, _unsafe_view_38);  view_38 = _unsafe_view_38 = None
        _unsafe_view_39 = torch.ops.aten._unsafe_view.default(bmm_15, [32, 8, 21, 64]);  bmm_15 = None
        permute_39 = torch.ops.aten.permute.default(_unsafe_view_39, [0, 2, 1, 3]);  _unsafe_view_39 = None
        clone_34 = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        view_39 = torch.ops.aten.view.default(clone_34, [32, 21, -1]);  clone_34 = None
        _mkl_linear_43 = torch.ops.mkl._mkl_linear.default(view_39, arg131_1, arg130_1, None, 672);  view_39 = arg131_1 = arg130_1 = None
        add__13 = torch.ops.aten.add_.Tensor(_mkl_linear_43, add_31);  _mkl_linear_43 = add_31 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add__13, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        sqrt_15 = torch.ops.aten.sqrt.default(add_32);  add_32 = None
        reciprocal_15 = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        sub_24 = torch.ops.aten.sub.Tensor(add__13, getitem_31);  add__13 = getitem_31 = None
        mul_30 = torch.ops.aten.mul.Tensor(sub_24, reciprocal_15);  sub_24 = reciprocal_15 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, arg132_1);  mul_30 = arg132_1 = None
        add_33 = torch.ops.aten.add.Tensor(mul_31, arg133_1);  mul_31 = arg133_1 = None
        _mkl_linear_44 = torch.ops.mkl._mkl_linear.default(add_33, arg136_1, arg134_1, arg135_1, 672);  arg136_1 = arg134_1 = arg135_1 = None
        relu_6 = torch.ops.aten.relu.default(_mkl_linear_44);  _mkl_linear_44 = None
        _mkl_linear_45 = torch.ops.mkl._mkl_linear.default(relu_6, arg139_1, arg137_1, arg138_1, 672);  relu_6 = arg139_1 = arg137_1 = arg138_1 = None
        add__14 = torch.ops.aten.add_.Tensor(_mkl_linear_45, add_33);  _mkl_linear_45 = add_33 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add__14, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        sqrt_16 = torch.ops.aten.sqrt.default(add_34);  add_34 = None
        reciprocal_16 = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        sub_25 = torch.ops.aten.sub.Tensor(add__14, getitem_33);  add__14 = getitem_33 = None
        mul_32 = torch.ops.aten.mul.Tensor(sub_25, reciprocal_16);  sub_25 = reciprocal_16 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, arg140_1);  mul_32 = arg140_1 = None
        add_35 = torch.ops.aten.add.Tensor(mul_33, arg141_1);  mul_33 = arg141_1 = None
        _mkl_linear_46 = torch.ops.mkl._mkl_linear.default(add_35, arg143_1, arg142_1, None, 672);  arg143_1 = arg142_1 = None
        view_40 = torch.ops.aten.view.default(_mkl_linear_46, [32, 21, 8, 64]);  _mkl_linear_46 = None
        _mkl_linear_47 = torch.ops.mkl._mkl_linear.default(add_35, arg145_1, arg144_1, None, 672);  arg145_1 = arg144_1 = None
        view_41 = torch.ops.aten.view.default(_mkl_linear_47, [32, 21, 8, 64]);  _mkl_linear_47 = None
        _mkl_linear_48 = torch.ops.mkl._mkl_linear.default(add_35, arg147_1, arg146_1, None, 672);  arg147_1 = arg146_1 = None
        view_42 = torch.ops.aten.view.default(_mkl_linear_48, [32, 21, 8, 64]);  _mkl_linear_48 = None
        permute_40 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        permute_41 = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        permute_42 = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(bitwise_and, 1)
        div_16 = torch.ops.aten.div.Tensor(permute_40, 8.0);  permute_40 = None
        permute_43 = torch.ops.aten.permute.default(permute_41, [0, 1, 3, 2]);  permute_41 = None
        expand_32 = torch.ops.aten.expand.default(div_16, [32, 8, 21, 64]);  div_16 = None
        clone_35 = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        _unsafe_view_40 = torch.ops.aten._unsafe_view.default(clone_35, [256, 21, 64]);  clone_35 = None
        expand_33 = torch.ops.aten.expand.default(permute_43, [32, 8, 64, 21]);  permute_43 = None
        clone_36 = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        _unsafe_view_41 = torch.ops.aten._unsafe_view.default(clone_36, [256, 64, 21]);  clone_36 = None
        bmm_16 = torch.ops.aten.bmm.default(_unsafe_view_40, _unsafe_view_41);  _unsafe_view_40 = _unsafe_view_41 = None
        _unsafe_view_42 = torch.ops.aten._unsafe_view.default(bmm_16, [32, 8, 21, 21]);  bmm_16 = None
        eq_8 = torch.ops.aten.eq.Scalar(unsqueeze_10, 0);  unsqueeze_10 = None
        _tensor_constant9 = self._tensor_constant9
        lift_fresh_copy_9 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant9);  _tensor_constant9 = None
        where_8 = torch.ops.aten.where.self(eq_8, lift_fresh_copy_9, _unsafe_view_42);  eq_8 = lift_fresh_copy_9 = _unsafe_view_42 = None
        clone_37 = torch.ops.aten.clone.default(where_8, memory_format = torch.contiguous_format);  where_8 = None
        amax_8 = torch.ops.aten.amax.default(clone_37, [-1], True)
        sub_26 = torch.ops.aten.sub.Tensor(clone_37, amax_8);  clone_37 = amax_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_17 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        expand_34 = torch.ops.aten.expand.default(div_17, [32, 8, 21, 21]);  div_17 = None
        view_43 = torch.ops.aten.view.default(expand_34, [256, 21, 21]);  expand_34 = None
        expand_35 = torch.ops.aten.expand.default(permute_42, [32, 8, 21, 64]);  permute_42 = None
        clone_38 = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        _unsafe_view_43 = torch.ops.aten._unsafe_view.default(clone_38, [256, 21, 64]);  clone_38 = None
        bmm_17 = torch.ops.aten.bmm.default(view_43, _unsafe_view_43);  view_43 = _unsafe_view_43 = None
        _unsafe_view_44 = torch.ops.aten._unsafe_view.default(bmm_17, [32, 8, 21, 64]);  bmm_17 = None
        permute_44 = torch.ops.aten.permute.default(_unsafe_view_44, [0, 2, 1, 3]);  _unsafe_view_44 = None
        clone_39 = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_44 = torch.ops.aten.view.default(clone_39, [32, 21, -1]);  clone_39 = None
        _mkl_linear_49 = torch.ops.mkl._mkl_linear.default(view_44, arg149_1, arg148_1, None, 672);  view_44 = arg149_1 = arg148_1 = None
        add__15 = torch.ops.aten.add_.Tensor(_mkl_linear_49, add_35);  _mkl_linear_49 = add_35 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add__15, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
        sqrt_17 = torch.ops.aten.sqrt.default(add_36);  add_36 = None
        reciprocal_17 = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        sub_27 = torch.ops.aten.sub.Tensor(add__15, getitem_35);  add__15 = getitem_35 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_27, reciprocal_17);  sub_27 = reciprocal_17 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, arg150_1);  mul_34 = arg150_1 = None
        add_37 = torch.ops.aten.add.Tensor(mul_35, arg151_1);  mul_35 = arg151_1 = None
        _mkl_linear_50 = torch.ops.mkl._mkl_linear.default(add_37, arg153_1, arg152_1, None, 672);  arg153_1 = arg152_1 = None
        view_45 = torch.ops.aten.view.default(_mkl_linear_50, [32, 21, 8, 64]);  _mkl_linear_50 = None
        _mkl_linear_51 = torch.ops.mkl._mkl_linear.default(add_26, arg155_1, arg154_1, None, 704);  arg155_1 = arg154_1 = None
        view_46 = torch.ops.aten.view.default(_mkl_linear_51, [32, 22, 8, 64]);  _mkl_linear_51 = None
        _mkl_linear_52 = torch.ops.mkl._mkl_linear.default(add_26, arg157_1, arg156_1, None, 704);  arg157_1 = arg156_1 = None
        view_47 = torch.ops.aten.view.default(_mkl_linear_52, [32, 22, 8, 64]);  _mkl_linear_52 = None
        permute_45 = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        permute_46 = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        permute_47 = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        div_18 = torch.ops.aten.div.Tensor(permute_45, 8.0);  permute_45 = None
        permute_48 = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2]);  permute_46 = None
        expand_36 = torch.ops.aten.expand.default(div_18, [32, 8, 21, 64]);  div_18 = None
        clone_40 = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        _unsafe_view_45 = torch.ops.aten._unsafe_view.default(clone_40, [256, 21, 64]);  clone_40 = None
        expand_37 = torch.ops.aten.expand.default(permute_48, [32, 8, 64, 22]);  permute_48 = None
        clone_41 = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        _unsafe_view_46 = torch.ops.aten._unsafe_view.default(clone_41, [256, 64, 22]);  clone_41 = None
        bmm_18 = torch.ops.aten.bmm.default(_unsafe_view_45, _unsafe_view_46);  _unsafe_view_45 = _unsafe_view_46 = None
        _unsafe_view_47 = torch.ops.aten._unsafe_view.default(bmm_18, [32, 8, 21, 22]);  bmm_18 = None
        eq_9 = torch.ops.aten.eq.Scalar(unsqueeze_11, 0);  unsqueeze_11 = None
        _tensor_constant10 = self._tensor_constant10
        lift_fresh_copy_10 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
        where_9 = torch.ops.aten.where.self(eq_9, lift_fresh_copy_10, _unsafe_view_47);  eq_9 = lift_fresh_copy_10 = _unsafe_view_47 = None
        amax_9 = torch.ops.aten.amax.default(where_9, [-1], True)
        sub_28 = torch.ops.aten.sub.Tensor(where_9, amax_9);  where_9 = amax_9 = None
        exp_9 = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_19 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        expand_38 = torch.ops.aten.expand.default(div_19, [32, 8, 21, 22]);  div_19 = None
        view_48 = torch.ops.aten.view.default(expand_38, [256, 21, 22]);  expand_38 = None
        expand_39 = torch.ops.aten.expand.default(permute_47, [32, 8, 22, 64]);  permute_47 = None
        clone_42 = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        _unsafe_view_48 = torch.ops.aten._unsafe_view.default(clone_42, [256, 22, 64]);  clone_42 = None
        bmm_19 = torch.ops.aten.bmm.default(view_48, _unsafe_view_48);  view_48 = _unsafe_view_48 = None
        _unsafe_view_49 = torch.ops.aten._unsafe_view.default(bmm_19, [32, 8, 21, 64]);  bmm_19 = None
        permute_49 = torch.ops.aten.permute.default(_unsafe_view_49, [0, 2, 1, 3]);  _unsafe_view_49 = None
        clone_43 = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_49 = torch.ops.aten.view.default(clone_43, [32, 21, -1]);  clone_43 = None
        _mkl_linear_53 = torch.ops.mkl._mkl_linear.default(view_49, arg159_1, arg158_1, None, 672);  view_49 = arg159_1 = arg158_1 = None
        add__16 = torch.ops.aten.add_.Tensor(_mkl_linear_53, add_37);  _mkl_linear_53 = add_37 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add__16, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_38 = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
        sqrt_18 = torch.ops.aten.sqrt.default(add_38);  add_38 = None
        reciprocal_18 = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        sub_29 = torch.ops.aten.sub.Tensor(add__16, getitem_37);  add__16 = getitem_37 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_29, reciprocal_18);  sub_29 = reciprocal_18 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, arg160_1);  mul_36 = arg160_1 = None
        add_39 = torch.ops.aten.add.Tensor(mul_37, arg161_1);  mul_37 = arg161_1 = None
        _mkl_linear_54 = torch.ops.mkl._mkl_linear.default(add_39, arg164_1, arg162_1, arg163_1, 672);  arg164_1 = arg162_1 = arg163_1 = None
        relu_7 = torch.ops.aten.relu.default(_mkl_linear_54);  _mkl_linear_54 = None
        _mkl_linear_55 = torch.ops.mkl._mkl_linear.default(relu_7, arg167_1, arg165_1, arg166_1, 672);  relu_7 = arg167_1 = arg165_1 = arg166_1 = None
        add__17 = torch.ops.aten.add_.Tensor(_mkl_linear_55, add_39);  _mkl_linear_55 = add_39 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add__17, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_40 = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
        sqrt_19 = torch.ops.aten.sqrt.default(add_40);  add_40 = None
        reciprocal_19 = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        sub_30 = torch.ops.aten.sub.Tensor(add__17, getitem_39);  add__17 = getitem_39 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_30, reciprocal_19);  sub_30 = reciprocal_19 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, arg168_1);  mul_38 = arg168_1 = None
        add_41 = torch.ops.aten.add.Tensor(mul_39, arg169_1);  mul_39 = arg169_1 = None
        _mkl_linear_56 = torch.ops.mkl._mkl_linear.default(add_41, arg171_1, arg170_1, None, 672);  arg171_1 = arg170_1 = None
        view_50 = torch.ops.aten.view.default(_mkl_linear_56, [32, 21, 8, 64]);  _mkl_linear_56 = None
        _mkl_linear_57 = torch.ops.mkl._mkl_linear.default(add_41, arg173_1, arg172_1, None, 672);  arg173_1 = arg172_1 = None
        view_51 = torch.ops.aten.view.default(_mkl_linear_57, [32, 21, 8, 64]);  _mkl_linear_57 = None
        _mkl_linear_58 = torch.ops.mkl._mkl_linear.default(add_41, arg175_1, arg174_1, None, 672);  arg175_1 = arg174_1 = None
        view_52 = torch.ops.aten.view.default(_mkl_linear_58, [32, 21, 8, 64]);  _mkl_linear_58 = None
        permute_50 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        permute_51 = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        permute_52 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(bitwise_and, 1)
        div_20 = torch.ops.aten.div.Tensor(permute_50, 8.0);  permute_50 = None
        permute_53 = torch.ops.aten.permute.default(permute_51, [0, 1, 3, 2]);  permute_51 = None
        expand_40 = torch.ops.aten.expand.default(div_20, [32, 8, 21, 64]);  div_20 = None
        clone_44 = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        _unsafe_view_50 = torch.ops.aten._unsafe_view.default(clone_44, [256, 21, 64]);  clone_44 = None
        expand_41 = torch.ops.aten.expand.default(permute_53, [32, 8, 64, 21]);  permute_53 = None
        clone_45 = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        _unsafe_view_51 = torch.ops.aten._unsafe_view.default(clone_45, [256, 64, 21]);  clone_45 = None
        bmm_20 = torch.ops.aten.bmm.default(_unsafe_view_50, _unsafe_view_51);  _unsafe_view_50 = _unsafe_view_51 = None
        _unsafe_view_52 = torch.ops.aten._unsafe_view.default(bmm_20, [32, 8, 21, 21]);  bmm_20 = None
        eq_10 = torch.ops.aten.eq.Scalar(unsqueeze_12, 0);  unsqueeze_12 = None
        _tensor_constant11 = self._tensor_constant11
        lift_fresh_copy_11 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant11);  _tensor_constant11 = None
        where_10 = torch.ops.aten.where.self(eq_10, lift_fresh_copy_11, _unsafe_view_52);  eq_10 = lift_fresh_copy_11 = _unsafe_view_52 = None
        clone_46 = torch.ops.aten.clone.default(where_10, memory_format = torch.contiguous_format);  where_10 = None
        amax_10 = torch.ops.aten.amax.default(clone_46, [-1], True)
        sub_31 = torch.ops.aten.sub.Tensor(clone_46, amax_10);  clone_46 = amax_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_21 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        expand_42 = torch.ops.aten.expand.default(div_21, [32, 8, 21, 21]);  div_21 = None
        view_53 = torch.ops.aten.view.default(expand_42, [256, 21, 21]);  expand_42 = None
        expand_43 = torch.ops.aten.expand.default(permute_52, [32, 8, 21, 64]);  permute_52 = None
        clone_47 = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        _unsafe_view_53 = torch.ops.aten._unsafe_view.default(clone_47, [256, 21, 64]);  clone_47 = None
        bmm_21 = torch.ops.aten.bmm.default(view_53, _unsafe_view_53);  view_53 = _unsafe_view_53 = None
        _unsafe_view_54 = torch.ops.aten._unsafe_view.default(bmm_21, [32, 8, 21, 64]);  bmm_21 = None
        permute_54 = torch.ops.aten.permute.default(_unsafe_view_54, [0, 2, 1, 3]);  _unsafe_view_54 = None
        clone_48 = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
        view_54 = torch.ops.aten.view.default(clone_48, [32, 21, -1]);  clone_48 = None
        _mkl_linear_59 = torch.ops.mkl._mkl_linear.default(view_54, arg177_1, arg176_1, None, 672);  view_54 = arg177_1 = arg176_1 = None
        add__18 = torch.ops.aten.add_.Tensor(_mkl_linear_59, add_41);  _mkl_linear_59 = add_41 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add__18, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
        sqrt_20 = torch.ops.aten.sqrt.default(add_42);  add_42 = None
        reciprocal_20 = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
        sub_32 = torch.ops.aten.sub.Tensor(add__18, getitem_41);  add__18 = getitem_41 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_32, reciprocal_20);  sub_32 = reciprocal_20 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, arg178_1);  mul_40 = arg178_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_41, arg179_1);  mul_41 = arg179_1 = None
        _mkl_linear_60 = torch.ops.mkl._mkl_linear.default(add_43, arg181_1, arg180_1, None, 672);  arg181_1 = arg180_1 = None
        view_55 = torch.ops.aten.view.default(_mkl_linear_60, [32, 21, 8, 64]);  _mkl_linear_60 = None
        _mkl_linear_61 = torch.ops.mkl._mkl_linear.default(add_26, arg183_1, arg182_1, None, 704);  arg183_1 = arg182_1 = None
        view_56 = torch.ops.aten.view.default(_mkl_linear_61, [32, 22, 8, 64]);  _mkl_linear_61 = None
        _mkl_linear_62 = torch.ops.mkl._mkl_linear.default(add_26, arg185_1, arg184_1, None, 704);  arg185_1 = arg184_1 = None
        view_57 = torch.ops.aten.view.default(_mkl_linear_62, [32, 22, 8, 64]);  _mkl_linear_62 = None
        permute_55 = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
        permute_56 = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        permute_57 = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        div_22 = torch.ops.aten.div.Tensor(permute_55, 8.0);  permute_55 = None
        permute_58 = torch.ops.aten.permute.default(permute_56, [0, 1, 3, 2]);  permute_56 = None
        expand_44 = torch.ops.aten.expand.default(div_22, [32, 8, 21, 64]);  div_22 = None
        clone_49 = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        _unsafe_view_55 = torch.ops.aten._unsafe_view.default(clone_49, [256, 21, 64]);  clone_49 = None
        expand_45 = torch.ops.aten.expand.default(permute_58, [32, 8, 64, 22]);  permute_58 = None
        clone_50 = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        _unsafe_view_56 = torch.ops.aten._unsafe_view.default(clone_50, [256, 64, 22]);  clone_50 = None
        bmm_22 = torch.ops.aten.bmm.default(_unsafe_view_55, _unsafe_view_56);  _unsafe_view_55 = _unsafe_view_56 = None
        _unsafe_view_57 = torch.ops.aten._unsafe_view.default(bmm_22, [32, 8, 21, 22]);  bmm_22 = None
        eq_11 = torch.ops.aten.eq.Scalar(unsqueeze_13, 0);  unsqueeze_13 = None
        _tensor_constant12 = self._tensor_constant12
        lift_fresh_copy_12 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = None
        where_11 = torch.ops.aten.where.self(eq_11, lift_fresh_copy_12, _unsafe_view_57);  eq_11 = lift_fresh_copy_12 = _unsafe_view_57 = None
        amax_11 = torch.ops.aten.amax.default(where_11, [-1], True)
        sub_33 = torch.ops.aten.sub.Tensor(where_11, amax_11);  where_11 = amax_11 = None
        exp_11 = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_23 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        expand_46 = torch.ops.aten.expand.default(div_23, [32, 8, 21, 22]);  div_23 = None
        view_58 = torch.ops.aten.view.default(expand_46, [256, 21, 22]);  expand_46 = None
        expand_47 = torch.ops.aten.expand.default(permute_57, [32, 8, 22, 64]);  permute_57 = None
        clone_51 = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        _unsafe_view_58 = torch.ops.aten._unsafe_view.default(clone_51, [256, 22, 64]);  clone_51 = None
        bmm_23 = torch.ops.aten.bmm.default(view_58, _unsafe_view_58);  view_58 = _unsafe_view_58 = None
        _unsafe_view_59 = torch.ops.aten._unsafe_view.default(bmm_23, [32, 8, 21, 64]);  bmm_23 = None
        permute_59 = torch.ops.aten.permute.default(_unsafe_view_59, [0, 2, 1, 3]);  _unsafe_view_59 = None
        clone_52 = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_59 = torch.ops.aten.view.default(clone_52, [32, 21, -1]);  clone_52 = None
        _mkl_linear_63 = torch.ops.mkl._mkl_linear.default(view_59, arg187_1, arg186_1, None, 672);  view_59 = arg187_1 = arg186_1 = None
        add__19 = torch.ops.aten.add_.Tensor(_mkl_linear_63, add_43);  _mkl_linear_63 = add_43 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add__19, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_44 = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
        sqrt_21 = torch.ops.aten.sqrt.default(add_44);  add_44 = None
        reciprocal_21 = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
        sub_34 = torch.ops.aten.sub.Tensor(add__19, getitem_43);  add__19 = getitem_43 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_34, reciprocal_21);  sub_34 = reciprocal_21 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, arg188_1);  mul_42 = arg188_1 = None
        add_45 = torch.ops.aten.add.Tensor(mul_43, arg189_1);  mul_43 = arg189_1 = None
        _mkl_linear_64 = torch.ops.mkl._mkl_linear.default(add_45, arg192_1, arg190_1, arg191_1, 672);  arg192_1 = arg190_1 = arg191_1 = None
        relu_8 = torch.ops.aten.relu.default(_mkl_linear_64);  _mkl_linear_64 = None
        _mkl_linear_65 = torch.ops.mkl._mkl_linear.default(relu_8, arg195_1, arg193_1, arg194_1, 672);  relu_8 = arg195_1 = arg193_1 = arg194_1 = None
        add__20 = torch.ops.aten.add_.Tensor(_mkl_linear_65, add_45);  _mkl_linear_65 = add_45 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add__20, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
        sqrt_22 = torch.ops.aten.sqrt.default(add_46);  add_46 = None
        reciprocal_22 = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
        sub_35 = torch.ops.aten.sub.Tensor(add__20, getitem_45);  add__20 = getitem_45 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_35, reciprocal_22);  sub_35 = reciprocal_22 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, arg196_1);  mul_44 = arg196_1 = None
        add_47 = torch.ops.aten.add.Tensor(mul_45, arg197_1);  mul_45 = arg197_1 = None
        _mkl_linear_66 = torch.ops.mkl._mkl_linear.default(add_47, arg199_1, arg198_1, None, 672);  arg199_1 = arg198_1 = None
        view_60 = torch.ops.aten.view.default(_mkl_linear_66, [32, 21, 8, 64]);  _mkl_linear_66 = None
        _mkl_linear_67 = torch.ops.mkl._mkl_linear.default(add_47, arg201_1, arg200_1, None, 672);  arg201_1 = arg200_1 = None
        view_61 = torch.ops.aten.view.default(_mkl_linear_67, [32, 21, 8, 64]);  _mkl_linear_67 = None
        _mkl_linear_68 = torch.ops.mkl._mkl_linear.default(add_47, arg203_1, arg202_1, None, 672);  arg203_1 = arg202_1 = None
        view_62 = torch.ops.aten.view.default(_mkl_linear_68, [32, 21, 8, 64]);  _mkl_linear_68 = None
        permute_60 = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        permute_61 = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
        permute_62 = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(bitwise_and, 1)
        div_24 = torch.ops.aten.div.Tensor(permute_60, 8.0);  permute_60 = None
        permute_63 = torch.ops.aten.permute.default(permute_61, [0, 1, 3, 2]);  permute_61 = None
        expand_48 = torch.ops.aten.expand.default(div_24, [32, 8, 21, 64]);  div_24 = None
        clone_53 = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        _unsafe_view_60 = torch.ops.aten._unsafe_view.default(clone_53, [256, 21, 64]);  clone_53 = None
        expand_49 = torch.ops.aten.expand.default(permute_63, [32, 8, 64, 21]);  permute_63 = None
        clone_54 = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        _unsafe_view_61 = torch.ops.aten._unsafe_view.default(clone_54, [256, 64, 21]);  clone_54 = None
        bmm_24 = torch.ops.aten.bmm.default(_unsafe_view_60, _unsafe_view_61);  _unsafe_view_60 = _unsafe_view_61 = None
        _unsafe_view_62 = torch.ops.aten._unsafe_view.default(bmm_24, [32, 8, 21, 21]);  bmm_24 = None
        eq_12 = torch.ops.aten.eq.Scalar(unsqueeze_14, 0);  unsqueeze_14 = None
        _tensor_constant13 = self._tensor_constant13
        lift_fresh_copy_13 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant13);  _tensor_constant13 = None
        where_12 = torch.ops.aten.where.self(eq_12, lift_fresh_copy_13, _unsafe_view_62);  eq_12 = lift_fresh_copy_13 = _unsafe_view_62 = None
        clone_55 = torch.ops.aten.clone.default(where_12, memory_format = torch.contiguous_format);  where_12 = None
        amax_12 = torch.ops.aten.amax.default(clone_55, [-1], True)
        sub_36 = torch.ops.aten.sub.Tensor(clone_55, amax_12);  clone_55 = amax_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_25 = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        expand_50 = torch.ops.aten.expand.default(div_25, [32, 8, 21, 21]);  div_25 = None
        view_63 = torch.ops.aten.view.default(expand_50, [256, 21, 21]);  expand_50 = None
        expand_51 = torch.ops.aten.expand.default(permute_62, [32, 8, 21, 64]);  permute_62 = None
        clone_56 = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        _unsafe_view_63 = torch.ops.aten._unsafe_view.default(clone_56, [256, 21, 64]);  clone_56 = None
        bmm_25 = torch.ops.aten.bmm.default(view_63, _unsafe_view_63);  view_63 = _unsafe_view_63 = None
        _unsafe_view_64 = torch.ops.aten._unsafe_view.default(bmm_25, [32, 8, 21, 64]);  bmm_25 = None
        permute_64 = torch.ops.aten.permute.default(_unsafe_view_64, [0, 2, 1, 3]);  _unsafe_view_64 = None
        clone_57 = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        view_64 = torch.ops.aten.view.default(clone_57, [32, 21, -1]);  clone_57 = None
        _mkl_linear_69 = torch.ops.mkl._mkl_linear.default(view_64, arg205_1, arg204_1, None, 672);  view_64 = arg205_1 = arg204_1 = None
        add__21 = torch.ops.aten.add_.Tensor(_mkl_linear_69, add_47);  _mkl_linear_69 = add_47 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add__21, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_48 = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
        sqrt_23 = torch.ops.aten.sqrt.default(add_48);  add_48 = None
        reciprocal_23 = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        sub_37 = torch.ops.aten.sub.Tensor(add__21, getitem_47);  add__21 = getitem_47 = None
        mul_46 = torch.ops.aten.mul.Tensor(sub_37, reciprocal_23);  sub_37 = reciprocal_23 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_46, arg206_1);  mul_46 = arg206_1 = None
        add_49 = torch.ops.aten.add.Tensor(mul_47, arg207_1);  mul_47 = arg207_1 = None
        _mkl_linear_70 = torch.ops.mkl._mkl_linear.default(add_49, arg209_1, arg208_1, None, 672);  arg209_1 = arg208_1 = None
        view_65 = torch.ops.aten.view.default(_mkl_linear_70, [32, 21, 8, 64]);  _mkl_linear_70 = None
        _mkl_linear_71 = torch.ops.mkl._mkl_linear.default(add_26, arg211_1, arg210_1, None, 704);  arg211_1 = arg210_1 = None
        view_66 = torch.ops.aten.view.default(_mkl_linear_71, [32, 22, 8, 64]);  _mkl_linear_71 = None
        _mkl_linear_72 = torch.ops.mkl._mkl_linear.default(add_26, arg213_1, arg212_1, None, 704);  arg213_1 = arg212_1 = None
        view_67 = torch.ops.aten.view.default(_mkl_linear_72, [32, 22, 8, 64]);  _mkl_linear_72 = None
        permute_65 = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        permute_66 = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        permute_67 = torch.ops.aten.permute.default(view_67, [0, 2, 1, 3]);  view_67 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        div_26 = torch.ops.aten.div.Tensor(permute_65, 8.0);  permute_65 = None
        permute_68 = torch.ops.aten.permute.default(permute_66, [0, 1, 3, 2]);  permute_66 = None
        expand_52 = torch.ops.aten.expand.default(div_26, [32, 8, 21, 64]);  div_26 = None
        clone_58 = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        _unsafe_view_65 = torch.ops.aten._unsafe_view.default(clone_58, [256, 21, 64]);  clone_58 = None
        expand_53 = torch.ops.aten.expand.default(permute_68, [32, 8, 64, 22]);  permute_68 = None
        clone_59 = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        _unsafe_view_66 = torch.ops.aten._unsafe_view.default(clone_59, [256, 64, 22]);  clone_59 = None
        bmm_26 = torch.ops.aten.bmm.default(_unsafe_view_65, _unsafe_view_66);  _unsafe_view_65 = _unsafe_view_66 = None
        _unsafe_view_67 = torch.ops.aten._unsafe_view.default(bmm_26, [32, 8, 21, 22]);  bmm_26 = None
        eq_13 = torch.ops.aten.eq.Scalar(unsqueeze_15, 0);  unsqueeze_15 = None
        _tensor_constant14 = self._tensor_constant14
        lift_fresh_copy_14 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant14);  _tensor_constant14 = None
        where_13 = torch.ops.aten.where.self(eq_13, lift_fresh_copy_14, _unsafe_view_67);  eq_13 = lift_fresh_copy_14 = _unsafe_view_67 = None
        amax_13 = torch.ops.aten.amax.default(where_13, [-1], True)
        sub_38 = torch.ops.aten.sub.Tensor(where_13, amax_13);  where_13 = amax_13 = None
        exp_13 = torch.ops.aten.exp.default(sub_38);  sub_38 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_27 = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        expand_54 = torch.ops.aten.expand.default(div_27, [32, 8, 21, 22]);  div_27 = None
        view_68 = torch.ops.aten.view.default(expand_54, [256, 21, 22]);  expand_54 = None
        expand_55 = torch.ops.aten.expand.default(permute_67, [32, 8, 22, 64]);  permute_67 = None
        clone_60 = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        _unsafe_view_68 = torch.ops.aten._unsafe_view.default(clone_60, [256, 22, 64]);  clone_60 = None
        bmm_27 = torch.ops.aten.bmm.default(view_68, _unsafe_view_68);  view_68 = _unsafe_view_68 = None
        _unsafe_view_69 = torch.ops.aten._unsafe_view.default(bmm_27, [32, 8, 21, 64]);  bmm_27 = None
        permute_69 = torch.ops.aten.permute.default(_unsafe_view_69, [0, 2, 1, 3]);  _unsafe_view_69 = None
        clone_61 = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
        view_69 = torch.ops.aten.view.default(clone_61, [32, 21, -1]);  clone_61 = None
        _mkl_linear_73 = torch.ops.mkl._mkl_linear.default(view_69, arg215_1, arg214_1, None, 672);  view_69 = arg215_1 = arg214_1 = None
        add__22 = torch.ops.aten.add_.Tensor(_mkl_linear_73, add_49);  _mkl_linear_73 = add_49 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add__22, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
        sqrt_24 = torch.ops.aten.sqrt.default(add_50);  add_50 = None
        reciprocal_24 = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        sub_39 = torch.ops.aten.sub.Tensor(add__22, getitem_49);  add__22 = getitem_49 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_39, reciprocal_24);  sub_39 = reciprocal_24 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, arg216_1);  mul_48 = arg216_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_49, arg217_1);  mul_49 = arg217_1 = None
        _mkl_linear_74 = torch.ops.mkl._mkl_linear.default(add_51, arg220_1, arg218_1, arg219_1, 672);  arg220_1 = arg218_1 = arg219_1 = None
        relu_9 = torch.ops.aten.relu.default(_mkl_linear_74);  _mkl_linear_74 = None
        _mkl_linear_75 = torch.ops.mkl._mkl_linear.default(relu_9, arg223_1, arg221_1, arg222_1, 672);  relu_9 = arg223_1 = arg221_1 = arg222_1 = None
        add__23 = torch.ops.aten.add_.Tensor(_mkl_linear_75, add_51);  _mkl_linear_75 = add_51 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add__23, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_25[0]
        getitem_51 = var_mean_25[1];  var_mean_25 = None
        add_52 = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
        sqrt_25 = torch.ops.aten.sqrt.default(add_52);  add_52 = None
        reciprocal_25 = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        sub_40 = torch.ops.aten.sub.Tensor(add__23, getitem_51);  add__23 = getitem_51 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_40, reciprocal_25);  sub_40 = reciprocal_25 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, arg224_1);  mul_50 = arg224_1 = None
        add_53 = torch.ops.aten.add.Tensor(mul_51, arg225_1);  mul_51 = arg225_1 = None
        _mkl_linear_76 = torch.ops.mkl._mkl_linear.default(add_53, arg227_1, arg226_1, None, 672);  arg227_1 = arg226_1 = None
        view_70 = torch.ops.aten.view.default(_mkl_linear_76, [32, 21, 8, 64]);  _mkl_linear_76 = None
        _mkl_linear_77 = torch.ops.mkl._mkl_linear.default(add_53, arg229_1, arg228_1, None, 672);  arg229_1 = arg228_1 = None
        view_71 = torch.ops.aten.view.default(_mkl_linear_77, [32, 21, 8, 64]);  _mkl_linear_77 = None
        _mkl_linear_78 = torch.ops.mkl._mkl_linear.default(add_53, arg231_1, arg230_1, None, 672);  arg231_1 = arg230_1 = None
        view_72 = torch.ops.aten.view.default(_mkl_linear_78, [32, 21, 8, 64]);  _mkl_linear_78 = None
        permute_70 = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        permute_71 = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        permute_72 = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(bitwise_and, 1)
        div_28 = torch.ops.aten.div.Tensor(permute_70, 8.0);  permute_70 = None
        permute_73 = torch.ops.aten.permute.default(permute_71, [0, 1, 3, 2]);  permute_71 = None
        expand_56 = torch.ops.aten.expand.default(div_28, [32, 8, 21, 64]);  div_28 = None
        clone_62 = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        _unsafe_view_70 = torch.ops.aten._unsafe_view.default(clone_62, [256, 21, 64]);  clone_62 = None
        expand_57 = torch.ops.aten.expand.default(permute_73, [32, 8, 64, 21]);  permute_73 = None
        clone_63 = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        _unsafe_view_71 = torch.ops.aten._unsafe_view.default(clone_63, [256, 64, 21]);  clone_63 = None
        bmm_28 = torch.ops.aten.bmm.default(_unsafe_view_70, _unsafe_view_71);  _unsafe_view_70 = _unsafe_view_71 = None
        _unsafe_view_72 = torch.ops.aten._unsafe_view.default(bmm_28, [32, 8, 21, 21]);  bmm_28 = None
        eq_14 = torch.ops.aten.eq.Scalar(unsqueeze_16, 0);  unsqueeze_16 = None
        _tensor_constant15 = self._tensor_constant15
        lift_fresh_copy_15 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant15);  _tensor_constant15 = None
        where_14 = torch.ops.aten.where.self(eq_14, lift_fresh_copy_15, _unsafe_view_72);  eq_14 = lift_fresh_copy_15 = _unsafe_view_72 = None
        clone_64 = torch.ops.aten.clone.default(where_14, memory_format = torch.contiguous_format);  where_14 = None
        amax_14 = torch.ops.aten.amax.default(clone_64, [-1], True)
        sub_41 = torch.ops.aten.sub.Tensor(clone_64, amax_14);  clone_64 = amax_14 = None
        exp_14 = torch.ops.aten.exp.default(sub_41);  sub_41 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_29 = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        expand_58 = torch.ops.aten.expand.default(div_29, [32, 8, 21, 21]);  div_29 = None
        view_73 = torch.ops.aten.view.default(expand_58, [256, 21, 21]);  expand_58 = None
        expand_59 = torch.ops.aten.expand.default(permute_72, [32, 8, 21, 64]);  permute_72 = None
        clone_65 = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        _unsafe_view_73 = torch.ops.aten._unsafe_view.default(clone_65, [256, 21, 64]);  clone_65 = None
        bmm_29 = torch.ops.aten.bmm.default(view_73, _unsafe_view_73);  view_73 = _unsafe_view_73 = None
        _unsafe_view_74 = torch.ops.aten._unsafe_view.default(bmm_29, [32, 8, 21, 64]);  bmm_29 = None
        permute_74 = torch.ops.aten.permute.default(_unsafe_view_74, [0, 2, 1, 3]);  _unsafe_view_74 = None
        clone_66 = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
        view_74 = torch.ops.aten.view.default(clone_66, [32, 21, -1]);  clone_66 = None
        _mkl_linear_79 = torch.ops.mkl._mkl_linear.default(view_74, arg233_1, arg232_1, None, 672);  view_74 = arg233_1 = arg232_1 = None
        add__24 = torch.ops.aten.add_.Tensor(_mkl_linear_79, add_53);  _mkl_linear_79 = add_53 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add__24, [2], correction = 0, keepdim = True)
        getitem_52 = var_mean_26[0]
        getitem_53 = var_mean_26[1];  var_mean_26 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
        sqrt_26 = torch.ops.aten.sqrt.default(add_54);  add_54 = None
        reciprocal_26 = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
        sub_42 = torch.ops.aten.sub.Tensor(add__24, getitem_53);  add__24 = getitem_53 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_42, reciprocal_26);  sub_42 = reciprocal_26 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, arg234_1);  mul_52 = arg234_1 = None
        add_55 = torch.ops.aten.add.Tensor(mul_53, arg235_1);  mul_53 = arg235_1 = None
        _mkl_linear_80 = torch.ops.mkl._mkl_linear.default(add_55, arg237_1, arg236_1, None, 672);  arg237_1 = arg236_1 = None
        view_75 = torch.ops.aten.view.default(_mkl_linear_80, [32, 21, 8, 64]);  _mkl_linear_80 = None
        _mkl_linear_81 = torch.ops.mkl._mkl_linear.default(add_26, arg239_1, arg238_1, None, 704);  arg239_1 = arg238_1 = None
        view_76 = torch.ops.aten.view.default(_mkl_linear_81, [32, 22, 8, 64]);  _mkl_linear_81 = None
        _mkl_linear_82 = torch.ops.mkl._mkl_linear.default(add_26, arg241_1, arg240_1, None, 704);  arg241_1 = arg240_1 = None
        view_77 = torch.ops.aten.view.default(_mkl_linear_82, [32, 22, 8, 64]);  _mkl_linear_82 = None
        permute_75 = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        permute_76 = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        permute_77 = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        div_30 = torch.ops.aten.div.Tensor(permute_75, 8.0);  permute_75 = None
        permute_78 = torch.ops.aten.permute.default(permute_76, [0, 1, 3, 2]);  permute_76 = None
        expand_60 = torch.ops.aten.expand.default(div_30, [32, 8, 21, 64]);  div_30 = None
        clone_67 = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        _unsafe_view_75 = torch.ops.aten._unsafe_view.default(clone_67, [256, 21, 64]);  clone_67 = None
        expand_61 = torch.ops.aten.expand.default(permute_78, [32, 8, 64, 22]);  permute_78 = None
        clone_68 = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        _unsafe_view_76 = torch.ops.aten._unsafe_view.default(clone_68, [256, 64, 22]);  clone_68 = None
        bmm_30 = torch.ops.aten.bmm.default(_unsafe_view_75, _unsafe_view_76);  _unsafe_view_75 = _unsafe_view_76 = None
        _unsafe_view_77 = torch.ops.aten._unsafe_view.default(bmm_30, [32, 8, 21, 22]);  bmm_30 = None
        eq_15 = torch.ops.aten.eq.Scalar(unsqueeze_17, 0);  unsqueeze_17 = None
        _tensor_constant16 = self._tensor_constant16
        lift_fresh_copy_16 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant16);  _tensor_constant16 = None
        where_15 = torch.ops.aten.where.self(eq_15, lift_fresh_copy_16, _unsafe_view_77);  eq_15 = lift_fresh_copy_16 = _unsafe_view_77 = None
        amax_15 = torch.ops.aten.amax.default(where_15, [-1], True)
        sub_43 = torch.ops.aten.sub.Tensor(where_15, amax_15);  where_15 = amax_15 = None
        exp_15 = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_31 = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        expand_62 = torch.ops.aten.expand.default(div_31, [32, 8, 21, 22]);  div_31 = None
        view_78 = torch.ops.aten.view.default(expand_62, [256, 21, 22]);  expand_62 = None
        expand_63 = torch.ops.aten.expand.default(permute_77, [32, 8, 22, 64]);  permute_77 = None
        clone_69 = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        _unsafe_view_78 = torch.ops.aten._unsafe_view.default(clone_69, [256, 22, 64]);  clone_69 = None
        bmm_31 = torch.ops.aten.bmm.default(view_78, _unsafe_view_78);  view_78 = _unsafe_view_78 = None
        _unsafe_view_79 = torch.ops.aten._unsafe_view.default(bmm_31, [32, 8, 21, 64]);  bmm_31 = None
        permute_79 = torch.ops.aten.permute.default(_unsafe_view_79, [0, 2, 1, 3]);  _unsafe_view_79 = None
        clone_70 = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
        view_79 = torch.ops.aten.view.default(clone_70, [32, 21, -1]);  clone_70 = None
        _mkl_linear_83 = torch.ops.mkl._mkl_linear.default(view_79, arg243_1, arg242_1, None, 672);  view_79 = arg243_1 = arg242_1 = None
        add__25 = torch.ops.aten.add_.Tensor(_mkl_linear_83, add_55);  _mkl_linear_83 = add_55 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add__25, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_27[0]
        getitem_55 = var_mean_27[1];  var_mean_27 = None
        add_56 = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
        sqrt_27 = torch.ops.aten.sqrt.default(add_56);  add_56 = None
        reciprocal_27 = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        sub_44 = torch.ops.aten.sub.Tensor(add__25, getitem_55);  add__25 = getitem_55 = None
        mul_54 = torch.ops.aten.mul.Tensor(sub_44, reciprocal_27);  sub_44 = reciprocal_27 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, arg244_1);  mul_54 = arg244_1 = None
        add_57 = torch.ops.aten.add.Tensor(mul_55, arg245_1);  mul_55 = arg245_1 = None
        _mkl_linear_84 = torch.ops.mkl._mkl_linear.default(add_57, arg248_1, arg246_1, arg247_1, 672);  arg248_1 = arg246_1 = arg247_1 = None
        relu_10 = torch.ops.aten.relu.default(_mkl_linear_84);  _mkl_linear_84 = None
        _mkl_linear_85 = torch.ops.mkl._mkl_linear.default(relu_10, arg251_1, arg249_1, arg250_1, 672);  relu_10 = arg251_1 = arg249_1 = arg250_1 = None
        add__26 = torch.ops.aten.add_.Tensor(_mkl_linear_85, add_57);  _mkl_linear_85 = add_57 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add__26, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_28[0]
        getitem_57 = var_mean_28[1];  var_mean_28 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
        sqrt_28 = torch.ops.aten.sqrt.default(add_58);  add_58 = None
        reciprocal_28 = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        sub_45 = torch.ops.aten.sub.Tensor(add__26, getitem_57);  add__26 = getitem_57 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_45, reciprocal_28);  sub_45 = reciprocal_28 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, arg252_1);  mul_56 = arg252_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_57, arg253_1);  mul_57 = arg253_1 = None
        _mkl_linear_86 = torch.ops.mkl._mkl_linear.default(add_59, arg255_1, arg254_1, None, 672);  arg255_1 = arg254_1 = None
        view_80 = torch.ops.aten.view.default(_mkl_linear_86, [32, 21, 8, 64]);  _mkl_linear_86 = None
        _mkl_linear_87 = torch.ops.mkl._mkl_linear.default(add_59, arg257_1, arg256_1, None, 672);  arg257_1 = arg256_1 = None
        view_81 = torch.ops.aten.view.default(_mkl_linear_87, [32, 21, 8, 64]);  _mkl_linear_87 = None
        _mkl_linear_88 = torch.ops.mkl._mkl_linear.default(add_59, arg259_1, arg258_1, None, 672);  arg259_1 = arg258_1 = None
        view_82 = torch.ops.aten.view.default(_mkl_linear_88, [32, 21, 8, 64]);  _mkl_linear_88 = None
        permute_80 = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        permute_81 = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
        permute_82 = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(bitwise_and, 1);  bitwise_and = None
        div_32 = torch.ops.aten.div.Tensor(permute_80, 8.0);  permute_80 = None
        permute_83 = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2]);  permute_81 = None
        expand_64 = torch.ops.aten.expand.default(div_32, [32, 8, 21, 64]);  div_32 = None
        clone_71 = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        _unsafe_view_80 = torch.ops.aten._unsafe_view.default(clone_71, [256, 21, 64]);  clone_71 = None
        expand_65 = torch.ops.aten.expand.default(permute_83, [32, 8, 64, 21]);  permute_83 = None
        clone_72 = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        _unsafe_view_81 = torch.ops.aten._unsafe_view.default(clone_72, [256, 64, 21]);  clone_72 = None
        bmm_32 = torch.ops.aten.bmm.default(_unsafe_view_80, _unsafe_view_81);  _unsafe_view_80 = _unsafe_view_81 = None
        _unsafe_view_82 = torch.ops.aten._unsafe_view.default(bmm_32, [32, 8, 21, 21]);  bmm_32 = None
        eq_16 = torch.ops.aten.eq.Scalar(unsqueeze_18, 0);  unsqueeze_18 = None
        _tensor_constant17 = self._tensor_constant17
        lift_fresh_copy_17 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant17);  _tensor_constant17 = None
        where_16 = torch.ops.aten.where.self(eq_16, lift_fresh_copy_17, _unsafe_view_82);  eq_16 = lift_fresh_copy_17 = _unsafe_view_82 = None
        clone_73 = torch.ops.aten.clone.default(where_16, memory_format = torch.contiguous_format);  where_16 = None
        amax_16 = torch.ops.aten.amax.default(clone_73, [-1], True)
        sub_46 = torch.ops.aten.sub.Tensor(clone_73, amax_16);  clone_73 = amax_16 = None
        exp_16 = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_33 = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        expand_66 = torch.ops.aten.expand.default(div_33, [32, 8, 21, 21]);  div_33 = None
        view_83 = torch.ops.aten.view.default(expand_66, [256, 21, 21]);  expand_66 = None
        expand_67 = torch.ops.aten.expand.default(permute_82, [32, 8, 21, 64]);  permute_82 = None
        clone_74 = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        _unsafe_view_83 = torch.ops.aten._unsafe_view.default(clone_74, [256, 21, 64]);  clone_74 = None
        bmm_33 = torch.ops.aten.bmm.default(view_83, _unsafe_view_83);  view_83 = _unsafe_view_83 = None
        _unsafe_view_84 = torch.ops.aten._unsafe_view.default(bmm_33, [32, 8, 21, 64]);  bmm_33 = None
        permute_84 = torch.ops.aten.permute.default(_unsafe_view_84, [0, 2, 1, 3]);  _unsafe_view_84 = None
        clone_75 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_84 = torch.ops.aten.view.default(clone_75, [32, 21, -1]);  clone_75 = None
        _mkl_linear_89 = torch.ops.mkl._mkl_linear.default(view_84, arg261_1, arg260_1, None, 672);  view_84 = arg261_1 = arg260_1 = None
        add__27 = torch.ops.aten.add_.Tensor(_mkl_linear_89, add_59);  _mkl_linear_89 = add_59 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add__27, [2], correction = 0, keepdim = True)
        getitem_58 = var_mean_29[0]
        getitem_59 = var_mean_29[1];  var_mean_29 = None
        add_60 = torch.ops.aten.add.Tensor(getitem_58, 1e-06);  getitem_58 = None
        sqrt_29 = torch.ops.aten.sqrt.default(add_60);  add_60 = None
        reciprocal_29 = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        sub_47 = torch.ops.aten.sub.Tensor(add__27, getitem_59);  add__27 = getitem_59 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_47, reciprocal_29);  sub_47 = reciprocal_29 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, arg262_1);  mul_58 = arg262_1 = None
        add_61 = torch.ops.aten.add.Tensor(mul_59, arg263_1);  mul_59 = arg263_1 = None
        _mkl_linear_90 = torch.ops.mkl._mkl_linear.default(add_61, arg265_1, arg264_1, None, 672);  arg265_1 = arg264_1 = None
        view_85 = torch.ops.aten.view.default(_mkl_linear_90, [32, 21, 8, 64]);  _mkl_linear_90 = None
        _mkl_linear_91 = torch.ops.mkl._mkl_linear.default(add_26, arg267_1, arg266_1, None, 704);  arg267_1 = arg266_1 = None
        view_86 = torch.ops.aten.view.default(_mkl_linear_91, [32, 22, 8, 64]);  _mkl_linear_91 = None
        _mkl_linear_92 = torch.ops.mkl._mkl_linear.default(add_26, arg269_1, arg268_1, None, 704);  add_26 = arg269_1 = arg268_1 = None
        view_87 = torch.ops.aten.view.default(_mkl_linear_92, [32, 22, 8, 64]);  _mkl_linear_92 = None
        permute_85 = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        permute_86 = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        permute_87 = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        div_34 = torch.ops.aten.div.Tensor(permute_85, 8.0);  permute_85 = None
        permute_88 = torch.ops.aten.permute.default(permute_86, [0, 1, 3, 2]);  permute_86 = None
        expand_68 = torch.ops.aten.expand.default(div_34, [32, 8, 21, 64]);  div_34 = None
        clone_76 = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        _unsafe_view_85 = torch.ops.aten._unsafe_view.default(clone_76, [256, 21, 64]);  clone_76 = None
        expand_69 = torch.ops.aten.expand.default(permute_88, [32, 8, 64, 22]);  permute_88 = None
        clone_77 = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        _unsafe_view_86 = torch.ops.aten._unsafe_view.default(clone_77, [256, 64, 22]);  clone_77 = None
        bmm_34 = torch.ops.aten.bmm.default(_unsafe_view_85, _unsafe_view_86);  _unsafe_view_85 = _unsafe_view_86 = None
        _unsafe_view_87 = torch.ops.aten._unsafe_view.default(bmm_34, [32, 8, 21, 22]);  bmm_34 = None
        eq_17 = torch.ops.aten.eq.Scalar(unsqueeze_19, 0);  unsqueeze_19 = None
        _tensor_constant18 = self._tensor_constant18
        lift_fresh_copy_18 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant18);  _tensor_constant18 = None
        where_17 = torch.ops.aten.where.self(eq_17, lift_fresh_copy_18, _unsafe_view_87);  eq_17 = lift_fresh_copy_18 = _unsafe_view_87 = None
        amax_17 = torch.ops.aten.amax.default(where_17, [-1], True)
        sub_48 = torch.ops.aten.sub.Tensor(where_17, amax_17);  where_17 = amax_17 = None
        exp_17 = torch.ops.aten.exp.default(sub_48);  sub_48 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_35 = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        expand_70 = torch.ops.aten.expand.default(div_35, [32, 8, 21, 22]);  div_35 = None
        view_88 = torch.ops.aten.view.default(expand_70, [256, 21, 22]);  expand_70 = None
        expand_71 = torch.ops.aten.expand.default(permute_87, [32, 8, 22, 64]);  permute_87 = None
        clone_78 = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        _unsafe_view_88 = torch.ops.aten._unsafe_view.default(clone_78, [256, 22, 64]);  clone_78 = None
        bmm_35 = torch.ops.aten.bmm.default(view_88, _unsafe_view_88);  view_88 = _unsafe_view_88 = None
        _unsafe_view_89 = torch.ops.aten._unsafe_view.default(bmm_35, [32, 8, 21, 64]);  bmm_35 = None
        permute_89 = torch.ops.aten.permute.default(_unsafe_view_89, [0, 2, 1, 3]);  _unsafe_view_89 = None
        clone_79 = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        view_89 = torch.ops.aten.view.default(clone_79, [32, 21, -1]);  clone_79 = None
        _mkl_linear_93 = torch.ops.mkl._mkl_linear.default(view_89, arg271_1, arg270_1, None, 672);  view_89 = arg271_1 = arg270_1 = None
        add__28 = torch.ops.aten.add_.Tensor(_mkl_linear_93, add_61);  _mkl_linear_93 = add_61 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add__28, [2], correction = 0, keepdim = True)
        getitem_60 = var_mean_30[0]
        getitem_61 = var_mean_30[1];  var_mean_30 = None
        add_62 = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
        sqrt_30 = torch.ops.aten.sqrt.default(add_62);  add_62 = None
        reciprocal_30 = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        sub_49 = torch.ops.aten.sub.Tensor(add__28, getitem_61);  add__28 = getitem_61 = None
        mul_60 = torch.ops.aten.mul.Tensor(sub_49, reciprocal_30);  sub_49 = reciprocal_30 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, arg272_1);  mul_60 = arg272_1 = None
        add_63 = torch.ops.aten.add.Tensor(mul_61, arg273_1);  mul_61 = arg273_1 = None
        _mkl_linear_94 = torch.ops.mkl._mkl_linear.default(add_63, arg276_1, arg274_1, arg275_1, 672);  arg276_1 = arg274_1 = arg275_1 = None
        relu_11 = torch.ops.aten.relu.default(_mkl_linear_94);  _mkl_linear_94 = None
        _mkl_linear_95 = torch.ops.mkl._mkl_linear.default(relu_11, arg279_1, arg277_1, arg278_1, 672);  relu_11 = arg279_1 = arg277_1 = arg278_1 = None
        add__29 = torch.ops.aten.add_.Tensor(_mkl_linear_95, add_63);  _mkl_linear_95 = add_63 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add__29, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_31[0]
        getitem_63 = var_mean_31[1];  var_mean_31 = None
        add_64 = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
        sqrt_31 = torch.ops.aten.sqrt.default(add_64);  add_64 = None
        reciprocal_31 = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        sub_50 = torch.ops.aten.sub.Tensor(add__29, getitem_63);  add__29 = getitem_63 = None
        mul_62 = torch.ops.aten.mul.Tensor(sub_50, reciprocal_31);  sub_50 = reciprocal_31 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_62, arg280_1);  mul_62 = arg280_1 = None
        add_65 = torch.ops.aten.add.Tensor(mul_63, arg281_1);  mul_63 = arg281_1 = None
        _mkl_linear_96 = torch.ops.mkl._mkl_linear.default(add_65, arg283_1, arg282_1, None, 672);  add_65 = arg283_1 = arg282_1 = None
        mul_64 = torch.ops.aten.mul.Tensor(_mkl_linear_96, 1.0);  _mkl_linear_96 = None
        view_90 = torch.ops.aten.view.default(mul_64, [-1, 9521]);  mul_64 = None
        return (view_90,)
        
args = [((9521, 512), (512, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512), (512, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((3293409, 1), (1, 0), torch.float32, 'cpu'), ((512, 2048), (2048, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((4079841, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512), (512, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((3293409, 1), (1, 0), torch.float32, 'cpu'), ((512, 2048), (2048, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((4079841, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512), (512, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((3293409, 1), (1, 0), torch.float32, 'cpu'), ((512, 2048), (2048, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((4079841, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512), (512, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((3293409, 1), (1, 0), torch.float32, 'cpu'), ((512, 2048), (2048, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((4079841, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512), (512, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((3293409, 1), (1, 0), torch.float32, 'cpu'), ((512, 2048), (2048, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((4079841, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512), (512, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((3293409, 1), (1, 0), torch.float32, 'cpu'), ((512, 2048), (2048, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((4079841, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((9521, 512), (512, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512), (512, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((3293409, 1), (1, 0), torch.float32, 'cpu'), ((512, 2048), (2048, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((4079841, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512), (512, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((3293409, 1), (1, 0), torch.float32, 'cpu'), ((512, 2048), (2048, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((4079841, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512), (512, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((3293409, 1), (1, 0), torch.float32, 'cpu'), ((512, 2048), (2048, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((4079841, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512), (512, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((3293409, 1), (1, 0), torch.float32, 'cpu'), ((512, 2048), (2048, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((4079841, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512), (512, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((3293409, 1), (1, 0), torch.float32, 'cpu'), ((512, 2048), (2048, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((4079841, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512, 512), (512, 1), torch.float32, 'cpu'), ((2310369, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((2048, 512), (512, 1), torch.float32, 'cpu'), ((2048,), (1,), torch.float32, 'cpu'), ((3293409, 1), (1, 0), torch.float32, 'cpu'), ((512, 2048), (2048, 1), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((4079841, 1), (1, 0), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((512,), (1,), torch.float32, 'cpu'), ((9521, 512), (512, 1), torch.float32, 'cpu'), ((7880929, 1), (1, 0), torch.float32, 'cpu'), ((1, 200, 512), (102400, 512, 1), torch.float32, 'cpu'), ((1, 200, 512), (102400, 512, 1), torch.float32, 'cpu'), ((32, 22), (1, 32), torch.int64, 'cpu'), ((32, 21), (1, 32), torch.int64, 'cpu')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
compiled(args)
