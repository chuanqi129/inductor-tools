
from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const float* __restrict__ in_ptr4,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<22; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<32; i1+=1)
            {
                {
                    {
                        float tmp4 = 0;
                        float tmp5 = 0;
                        for(long i2=0; i2<512; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr0[i1 + (32*i0)];
                                auto tmp2 = in_ptr2[i2 + (512*i0)];
                                auto tmp1 = in_ptr1[i2 + (512*tmp0)];
                                auto tmp3 = tmp1 + tmp2;
                                tmp4 += tmp3;
                                tmp5 += tmp3;
                            }
                        }
                        out_ptr0[i1 + (32*i0)] = tmp4;
                        out_ptr1[i1 + (32*i0)] = tmp5;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<22; i1+=1)
            {
                {
                    {
                        float tmp9 = 0;
                        for(long i2=0; i2<512; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i1)];
                                auto tmp2 = in_ptr2[i2 + (512*i1)];
                                auto tmp4 = out_ptr0[i0 + (32*i1)];
                                auto tmp1 = in_ptr1[i2 + (512*tmp0)];
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp5 = static_cast<float>(512);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = tmp3 - tmp6;
                                auto tmp8 = tmp7 * tmp7;
                                tmp9 += tmp8;
                            }
                        }
                        out_ptr2[i0 + (32*i1)] = tmp9;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<22; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i0 + (32*i1)];
                            auto tmp2 = in_ptr2[i2 + (512*i1)];
                            auto tmp4 = out_ptr1[i0 + (32*i1)];
                            auto tmp8 = out_ptr2[i0 + (32*i1)];
                            auto tmp15 = in_ptr3[i2];
                            auto tmp17 = in_ptr4[i2];
                            auto tmp1 = in_ptr1[i2 + (512*tmp0)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = static_cast<float>(512);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = tmp3 - tmp6;
                            auto tmp9 = tmp8 / tmp5;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = tmp9 + tmp10;
                            auto tmp12 = std::sqrt(tmp11);
                            auto tmp13 = 1 / tmp12;
                            auto tmp14 = tmp7 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            out_ptr3[i2 + (512*i1) + (11264*i0)] = tmp18;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_1 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (11264*i0)];
                            out_ptr1[i2 + (22*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_2 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<176; i1+=1)
            {
                {
                    {
                        float tmp8 = -std::numeric_limits<float>::infinity();
                        for(long i2=0; i2<22; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i2)];
                                auto tmp6 = in_ptr1[i2 + (22*i1) + (3872*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<bool>(0);
                                auto tmp4 = tmp2 == tmp3;
                                auto tmp5 = static_cast<float>(-1000000000.0);
                                auto tmp7 = tmp4 ? tmp5 : tmp6;
                                tmp8 = std::max(tmp8, tmp7);
                            }
                        }
                        out_ptr0[i1 + (176*i0)] = tmp8;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<176; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i0 + (32*i2)];
                            auto tmp6 = in_ptr1[i2 + (22*i1) + (3872*i0)];
                            auto tmp8 = out_ptr0[i1 + (176*i0)];
                            auto tmp1 = static_cast<long>(1);
                            auto tmp2 = tmp0 != tmp1;
                            auto tmp3 = static_cast<bool>(0);
                            auto tmp4 = tmp2 == tmp3;
                            auto tmp5 = static_cast<float>(-1000000000.0);
                            auto tmp7 = tmp4 ? tmp5 : tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = std::exp(tmp9);
                            out_ptr1[i2 + (22*i1) + (3872*i0)] = tmp10;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5632; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<22; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (22*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5632; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (22*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<22; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (22*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (22*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        out_ptr4[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_3 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<22; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1408*(i2 / 64)) + (11264*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_4 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<22528; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=360448; i0<360448; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_5 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<90112; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=1441792; i0<1441792; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_6 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<22528; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=360448; i0<360448; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_7 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (11264*i0)];
                            out_ptr1[i2 + (22*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_8 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<176; i1+=1)
            {
                {
                    {
                        float tmp8 = -std::numeric_limits<float>::infinity();
                        for(long i2=0; i2<22; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i2)];
                                auto tmp6 = in_ptr1[i2 + (22*i1) + (3872*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<bool>(0);
                                auto tmp4 = tmp2 == tmp3;
                                auto tmp5 = static_cast<float>(-1000000000.0);
                                auto tmp7 = tmp4 ? tmp5 : tmp6;
                                tmp8 = std::max(tmp8, tmp7);
                            }
                        }
                        out_ptr0[i1 + (176*i0)] = tmp8;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<176; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i0 + (32*i2)];
                            auto tmp6 = in_ptr1[i2 + (22*i1) + (3872*i0)];
                            auto tmp8 = out_ptr0[i1 + (176*i0)];
                            auto tmp1 = static_cast<long>(1);
                            auto tmp2 = tmp0 != tmp1;
                            auto tmp3 = static_cast<bool>(0);
                            auto tmp4 = tmp2 == tmp3;
                            auto tmp5 = static_cast<float>(-1000000000.0);
                            auto tmp7 = tmp4 ? tmp5 : tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = std::exp(tmp9);
                            out_ptr1[i2 + (22*i1) + (3872*i0)] = tmp10;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5632; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<22; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (22*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5632; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (22*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<22; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (22*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (22*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        out_ptr4[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_9 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<22; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1408*(i2 / 64)) + (11264*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_10 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<22528; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=360448; i0<360448; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_11 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<90112; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=1441792; i0<1441792; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_12 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<22528; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=360448; i0<360448; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_13 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (11264*i0)];
                            out_ptr1[i2 + (22*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_14 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<176; i1+=1)
            {
                {
                    {
                        float tmp8 = -std::numeric_limits<float>::infinity();
                        for(long i2=0; i2<22; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i2)];
                                auto tmp6 = in_ptr1[i2 + (22*i1) + (3872*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<bool>(0);
                                auto tmp4 = tmp2 == tmp3;
                                auto tmp5 = static_cast<float>(-1000000000.0);
                                auto tmp7 = tmp4 ? tmp5 : tmp6;
                                tmp8 = std::max(tmp8, tmp7);
                            }
                        }
                        out_ptr0[i1 + (176*i0)] = tmp8;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<176; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i0 + (32*i2)];
                            auto tmp6 = in_ptr1[i2 + (22*i1) + (3872*i0)];
                            auto tmp8 = out_ptr0[i1 + (176*i0)];
                            auto tmp1 = static_cast<long>(1);
                            auto tmp2 = tmp0 != tmp1;
                            auto tmp3 = static_cast<bool>(0);
                            auto tmp4 = tmp2 == tmp3;
                            auto tmp5 = static_cast<float>(-1000000000.0);
                            auto tmp7 = tmp4 ? tmp5 : tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = std::exp(tmp9);
                            out_ptr1[i2 + (22*i1) + (3872*i0)] = tmp10;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5632; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<22; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (22*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5632; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (22*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<22; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (22*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (22*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        out_ptr4[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_15 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<22; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1408*(i2 / 64)) + (11264*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_16 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<22528; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=360448; i0<360448; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_17 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<90112; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=1441792; i0<1441792; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_18 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<22528; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=360448; i0<360448; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_19 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (11264*i0)];
                            out_ptr1[i2 + (22*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_20 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<176; i1+=1)
            {
                {
                    {
                        float tmp8 = -std::numeric_limits<float>::infinity();
                        for(long i2=0; i2<22; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i2)];
                                auto tmp6 = in_ptr1[i2 + (22*i1) + (3872*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<bool>(0);
                                auto tmp4 = tmp2 == tmp3;
                                auto tmp5 = static_cast<float>(-1000000000.0);
                                auto tmp7 = tmp4 ? tmp5 : tmp6;
                                tmp8 = std::max(tmp8, tmp7);
                            }
                        }
                        out_ptr0[i1 + (176*i0)] = tmp8;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<176; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i0 + (32*i2)];
                            auto tmp6 = in_ptr1[i2 + (22*i1) + (3872*i0)];
                            auto tmp8 = out_ptr0[i1 + (176*i0)];
                            auto tmp1 = static_cast<long>(1);
                            auto tmp2 = tmp0 != tmp1;
                            auto tmp3 = static_cast<bool>(0);
                            auto tmp4 = tmp2 == tmp3;
                            auto tmp5 = static_cast<float>(-1000000000.0);
                            auto tmp7 = tmp4 ? tmp5 : tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = std::exp(tmp9);
                            out_ptr1[i2 + (22*i1) + (3872*i0)] = tmp10;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5632; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<22; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (22*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5632; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (22*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<22; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (22*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (22*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        out_ptr4[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_21 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<22; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1408*(i2 / 64)) + (11264*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_22 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<22528; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=360448; i0<360448; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_23 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<90112; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=1441792; i0<1441792; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_24 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<22528; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=360448; i0<360448; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_25 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (11264*i0)];
                            out_ptr1[i2 + (22*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_26 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<176; i1+=1)
            {
                {
                    {
                        float tmp8 = -std::numeric_limits<float>::infinity();
                        for(long i2=0; i2<22; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i2)];
                                auto tmp6 = in_ptr1[i2 + (22*i1) + (3872*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<bool>(0);
                                auto tmp4 = tmp2 == tmp3;
                                auto tmp5 = static_cast<float>(-1000000000.0);
                                auto tmp7 = tmp4 ? tmp5 : tmp6;
                                tmp8 = std::max(tmp8, tmp7);
                            }
                        }
                        out_ptr0[i1 + (176*i0)] = tmp8;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<176; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i0 + (32*i2)];
                            auto tmp6 = in_ptr1[i2 + (22*i1) + (3872*i0)];
                            auto tmp8 = out_ptr0[i1 + (176*i0)];
                            auto tmp1 = static_cast<long>(1);
                            auto tmp2 = tmp0 != tmp1;
                            auto tmp3 = static_cast<bool>(0);
                            auto tmp4 = tmp2 == tmp3;
                            auto tmp5 = static_cast<float>(-1000000000.0);
                            auto tmp7 = tmp4 ? tmp5 : tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = std::exp(tmp9);
                            out_ptr1[i2 + (22*i1) + (3872*i0)] = tmp10;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5632; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<22; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (22*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5632; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (22*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<22; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (22*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (22*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        out_ptr4[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_27 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<22; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1408*(i2 / 64)) + (11264*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_28 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<22528; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=360448; i0<360448; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_29 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<90112; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=1441792; i0<1441792; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_30 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<22528; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=360448; i0<360448; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_31 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (11264*i0)];
                            out_ptr1[i2 + (22*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_32 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<176; i1+=1)
            {
                {
                    {
                        float tmp8 = -std::numeric_limits<float>::infinity();
                        for(long i2=0; i2<22; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i2)];
                                auto tmp6 = in_ptr1[i2 + (22*i1) + (3872*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<bool>(0);
                                auto tmp4 = tmp2 == tmp3;
                                auto tmp5 = static_cast<float>(-1000000000.0);
                                auto tmp7 = tmp4 ? tmp5 : tmp6;
                                tmp8 = std::max(tmp8, tmp7);
                            }
                        }
                        out_ptr0[i1 + (176*i0)] = tmp8;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<176; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i0 + (32*i2)];
                            auto tmp6 = in_ptr1[i2 + (22*i1) + (3872*i0)];
                            auto tmp8 = out_ptr0[i1 + (176*i0)];
                            auto tmp1 = static_cast<long>(1);
                            auto tmp2 = tmp0 != tmp1;
                            auto tmp3 = static_cast<bool>(0);
                            auto tmp4 = tmp2 == tmp3;
                            auto tmp5 = static_cast<float>(-1000000000.0);
                            auto tmp7 = tmp4 ? tmp5 : tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = std::exp(tmp9);
                            out_ptr1[i2 + (22*i1) + (3872*i0)] = tmp10;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5632; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<22; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (22*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5632; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (22*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<22; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (22*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (22*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        out_ptr4[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_33 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<22; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1408*(i2 / 64)) + (11264*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_34 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<22528; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=360448; i0<360448; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_35 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<90112; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=1441792; i0<1441792; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_36 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       const long* __restrict__ in_ptr4,
                       const float* __restrict__ in_ptr5,
                       const float* __restrict__ in_ptr6,
                       const float* __restrict__ in_ptr7,
                       const float* __restrict__ in_ptr8,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr5,
                       float* __restrict__ out_ptr6,
                       float* __restrict__ out_ptr7,
                       float* __restrict__ out_ptr8)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<22528; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=360448; i0<360448; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<704; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<21; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<32; i1+=1)
            {
                {
                    {
                        float tmp4 = 0;
                        float tmp5 = 0;
                        for(long i2=0; i2<512; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr4[i1 + (32*i0)];
                                auto tmp2 = in_ptr6[i2 + (512*i0)];
                                auto tmp1 = in_ptr5[i2 + (512*tmp0)];
                                auto tmp3 = tmp1 + tmp2;
                                tmp4 += tmp3;
                                tmp5 += tmp3;
                            }
                        }
                        out_ptr5[i1 + (32*i0)] = tmp4;
                        out_ptr6[i1 + (32*i0)] = tmp5;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<21; i1+=1)
            {
                {
                    {
                        float tmp9 = 0;
                        for(long i2=0; i2<512; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr4[i0 + (32*i1)];
                                auto tmp2 = in_ptr6[i2 + (512*i1)];
                                auto tmp4 = out_ptr5[i0 + (32*i1)];
                                auto tmp1 = in_ptr5[i2 + (512*tmp0)];
                                auto tmp3 = tmp1 + tmp2;
                                auto tmp5 = static_cast<float>(512);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = tmp3 - tmp6;
                                auto tmp8 = tmp7 * tmp7;
                                tmp9 += tmp8;
                            }
                        }
                        out_ptr7[i0 + (32*i1)] = tmp9;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<21; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr4[i0 + (32*i1)];
                            auto tmp2 = in_ptr6[i2 + (512*i1)];
                            auto tmp4 = out_ptr6[i0 + (32*i1)];
                            auto tmp8 = out_ptr7[i0 + (32*i1)];
                            auto tmp15 = in_ptr7[i2];
                            auto tmp17 = in_ptr8[i2];
                            auto tmp1 = in_ptr5[i2 + (512*tmp0)];
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = static_cast<float>(512);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = tmp3 - tmp6;
                            auto tmp9 = tmp8 / tmp5;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = tmp9 + tmp10;
                            auto tmp12 = std::sqrt(tmp11);
                            auto tmp13 = 1 / tmp12;
                            auto tmp14 = tmp7 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            out_ptr8[i2 + (512*i1) + (10752*i0)] = tmp18;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_37 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (10752*i0)];
                            out_ptr1[i2 + (21*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_38 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    {
                        {
                            float tmp18 = -std::numeric_limits<float>::infinity();
                            for(long i3=0; i3<21; i3+=1)
                            {
                                {
                                    auto tmp0 = in_ptr0[i0 + (32*i3)];
                                    auto tmp16 = in_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)];
                                    auto tmp1 = static_cast<long>(1);
                                    auto tmp2 = tmp0 != tmp1;
                                    auto tmp3 = static_cast<float>(1.0);
                                    auto tmp4 = static_cast<int>((-1) + i3 + ((-1)*i2));
                                    auto tmp5 = static_cast<int>(0);
                                    auto tmp6 = tmp4 >= tmp5;
                                    auto tmp7 = static_cast<float>(1);
                                    auto tmp8 = static_cast<float>(0);
                                    auto tmp9 = tmp6 ? tmp7 : tmp8;
                                    auto tmp10 = tmp3 - tmp9;
                                    auto tmp11 = static_cast<bool>(tmp10);
                                    auto tmp12 = tmp2 & tmp11;
                                    auto tmp13 = static_cast<bool>(0);
                                    auto tmp14 = tmp12 == tmp13;
                                    auto tmp15 = static_cast<float>(-1000000000.0);
                                    auto tmp17 = tmp14 ? tmp15 : tmp16;
                                    tmp18 = std::max(tmp18, tmp17);
                                }
                            }
                            out_ptr0[i2 + (21*i1) + (168*i0)] = tmp18;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    #pragma GCC ivdep
                    for(long i3=0; i3<21; i3+=1)
                    {
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i3)];
                                auto tmp16 = in_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)];
                                auto tmp18 = out_ptr0[i2 + (21*i1) + (168*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<float>(1.0);
                                auto tmp4 = static_cast<int>((-1) + i3 + ((-1)*i2));
                                auto tmp5 = static_cast<int>(0);
                                auto tmp6 = tmp4 >= tmp5;
                                auto tmp7 = static_cast<float>(1);
                                auto tmp8 = static_cast<float>(0);
                                auto tmp9 = tmp6 ? tmp7 : tmp8;
                                auto tmp10 = tmp3 - tmp9;
                                auto tmp11 = static_cast<bool>(tmp10);
                                auto tmp12 = tmp2 & tmp11;
                                auto tmp13 = static_cast<bool>(0);
                                auto tmp14 = tmp12 == tmp13;
                                auto tmp15 = static_cast<float>(-1000000000.0);
                                auto tmp17 = tmp14 ? tmp15 : tmp16;
                                auto tmp19 = tmp17 - tmp18;
                                auto tmp20 = std::exp(tmp19);
                                out_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)] = tmp20;
                            }
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (21*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<21; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (21*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (21*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (21*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<21; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (21*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (21*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        out_ptr4[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_39 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<21; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1344*(i2 / 64)) + (10752*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_40 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_41 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (11264*i0)];
                            out_ptr1[i2 + (22*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_42 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<168; i1+=1)
            {
                {
                    {
                        float tmp8 = -std::numeric_limits<float>::infinity();
                        for(long i2=0; i2<22; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i2)];
                                auto tmp6 = in_ptr1[i2 + (22*i1) + (3696*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<bool>(0);
                                auto tmp4 = tmp2 == tmp3;
                                auto tmp5 = static_cast<float>(-1000000000.0);
                                auto tmp7 = tmp4 ? tmp5 : tmp6;
                                tmp8 = std::max(tmp8, tmp7);
                            }
                        }
                        out_ptr0[i1 + (168*i0)] = tmp8;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<168; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i0 + (32*i2)];
                            auto tmp6 = in_ptr1[i2 + (22*i1) + (3696*i0)];
                            auto tmp8 = out_ptr0[i1 + (168*i0)];
                            auto tmp1 = static_cast<long>(1);
                            auto tmp2 = tmp0 != tmp1;
                            auto tmp3 = static_cast<bool>(0);
                            auto tmp4 = tmp2 == tmp3;
                            auto tmp5 = static_cast<float>(-1000000000.0);
                            auto tmp7 = tmp4 ? tmp5 : tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = std::exp(tmp9);
                            out_ptr1[i2 + (22*i1) + (3696*i0)] = tmp10;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<22; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (22*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (22*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<22; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (22*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (22*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        out_ptr4[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_43 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<21; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1344*(i2 / 64)) + (10752*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_44 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_45 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<86016; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=1376256; i0<1376256; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_46 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_47 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (10752*i0)];
                            out_ptr1[i2 + (21*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_48 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    {
                        {
                            float tmp18 = -std::numeric_limits<float>::infinity();
                            for(long i3=0; i3<21; i3+=1)
                            {
                                {
                                    auto tmp0 = in_ptr0[i0 + (32*i3)];
                                    auto tmp16 = in_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)];
                                    auto tmp1 = static_cast<long>(1);
                                    auto tmp2 = tmp0 != tmp1;
                                    auto tmp3 = static_cast<float>(1.0);
                                    auto tmp4 = static_cast<int>((-1) + i3 + ((-1)*i2));
                                    auto tmp5 = static_cast<int>(0);
                                    auto tmp6 = tmp4 >= tmp5;
                                    auto tmp7 = static_cast<float>(1);
                                    auto tmp8 = static_cast<float>(0);
                                    auto tmp9 = tmp6 ? tmp7 : tmp8;
                                    auto tmp10 = tmp3 - tmp9;
                                    auto tmp11 = static_cast<bool>(tmp10);
                                    auto tmp12 = tmp2 & tmp11;
                                    auto tmp13 = static_cast<bool>(0);
                                    auto tmp14 = tmp12 == tmp13;
                                    auto tmp15 = static_cast<float>(-1000000000.0);
                                    auto tmp17 = tmp14 ? tmp15 : tmp16;
                                    tmp18 = std::max(tmp18, tmp17);
                                }
                            }
                            out_ptr0[i2 + (21*i1) + (168*i0)] = tmp18;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    #pragma GCC ivdep
                    for(long i3=0; i3<21; i3+=1)
                    {
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i3)];
                                auto tmp16 = in_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)];
                                auto tmp18 = out_ptr0[i2 + (21*i1) + (168*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<float>(1.0);
                                auto tmp4 = static_cast<int>((-1) + i3 + ((-1)*i2));
                                auto tmp5 = static_cast<int>(0);
                                auto tmp6 = tmp4 >= tmp5;
                                auto tmp7 = static_cast<float>(1);
                                auto tmp8 = static_cast<float>(0);
                                auto tmp9 = tmp6 ? tmp7 : tmp8;
                                auto tmp10 = tmp3 - tmp9;
                                auto tmp11 = static_cast<bool>(tmp10);
                                auto tmp12 = tmp2 & tmp11;
                                auto tmp13 = static_cast<bool>(0);
                                auto tmp14 = tmp12 == tmp13;
                                auto tmp15 = static_cast<float>(-1000000000.0);
                                auto tmp17 = tmp14 ? tmp15 : tmp16;
                                auto tmp19 = tmp17 - tmp18;
                                auto tmp20 = std::exp(tmp19);
                                out_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)] = tmp20;
                            }
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (21*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<21; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (21*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (21*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (21*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<21; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (21*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (21*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        out_ptr4[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_49 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<21; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1344*(i2 / 64)) + (10752*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_50 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_51 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (11264*i0)];
                            out_ptr1[i2 + (22*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_52 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<168; i1+=1)
            {
                {
                    {
                        float tmp8 = -std::numeric_limits<float>::infinity();
                        for(long i2=0; i2<22; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i2)];
                                auto tmp6 = in_ptr1[i2 + (22*i1) + (3696*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<bool>(0);
                                auto tmp4 = tmp2 == tmp3;
                                auto tmp5 = static_cast<float>(-1000000000.0);
                                auto tmp7 = tmp4 ? tmp5 : tmp6;
                                tmp8 = std::max(tmp8, tmp7);
                            }
                        }
                        out_ptr0[i1 + (168*i0)] = tmp8;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<168; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i0 + (32*i2)];
                            auto tmp6 = in_ptr1[i2 + (22*i1) + (3696*i0)];
                            auto tmp8 = out_ptr0[i1 + (168*i0)];
                            auto tmp1 = static_cast<long>(1);
                            auto tmp2 = tmp0 != tmp1;
                            auto tmp3 = static_cast<bool>(0);
                            auto tmp4 = tmp2 == tmp3;
                            auto tmp5 = static_cast<float>(-1000000000.0);
                            auto tmp7 = tmp4 ? tmp5 : tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = std::exp(tmp9);
                            out_ptr1[i2 + (22*i1) + (3696*i0)] = tmp10;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<22; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (22*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (22*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<22; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (22*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (22*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        out_ptr4[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_53 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<21; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1344*(i2 / 64)) + (10752*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_54 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_55 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<86016; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=1376256; i0<1376256; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_56 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_57 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (10752*i0)];
                            out_ptr1[i2 + (21*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_58 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    {
                        {
                            float tmp18 = -std::numeric_limits<float>::infinity();
                            for(long i3=0; i3<21; i3+=1)
                            {
                                {
                                    auto tmp0 = in_ptr0[i0 + (32*i3)];
                                    auto tmp16 = in_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)];
                                    auto tmp1 = static_cast<long>(1);
                                    auto tmp2 = tmp0 != tmp1;
                                    auto tmp3 = static_cast<float>(1.0);
                                    auto tmp4 = static_cast<int>((-1) + i3 + ((-1)*i2));
                                    auto tmp5 = static_cast<int>(0);
                                    auto tmp6 = tmp4 >= tmp5;
                                    auto tmp7 = static_cast<float>(1);
                                    auto tmp8 = static_cast<float>(0);
                                    auto tmp9 = tmp6 ? tmp7 : tmp8;
                                    auto tmp10 = tmp3 - tmp9;
                                    auto tmp11 = static_cast<bool>(tmp10);
                                    auto tmp12 = tmp2 & tmp11;
                                    auto tmp13 = static_cast<bool>(0);
                                    auto tmp14 = tmp12 == tmp13;
                                    auto tmp15 = static_cast<float>(-1000000000.0);
                                    auto tmp17 = tmp14 ? tmp15 : tmp16;
                                    tmp18 = std::max(tmp18, tmp17);
                                }
                            }
                            out_ptr0[i2 + (21*i1) + (168*i0)] = tmp18;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    #pragma GCC ivdep
                    for(long i3=0; i3<21; i3+=1)
                    {
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i3)];
                                auto tmp16 = in_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)];
                                auto tmp18 = out_ptr0[i2 + (21*i1) + (168*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<float>(1.0);
                                auto tmp4 = static_cast<int>((-1) + i3 + ((-1)*i2));
                                auto tmp5 = static_cast<int>(0);
                                auto tmp6 = tmp4 >= tmp5;
                                auto tmp7 = static_cast<float>(1);
                                auto tmp8 = static_cast<float>(0);
                                auto tmp9 = tmp6 ? tmp7 : tmp8;
                                auto tmp10 = tmp3 - tmp9;
                                auto tmp11 = static_cast<bool>(tmp10);
                                auto tmp12 = tmp2 & tmp11;
                                auto tmp13 = static_cast<bool>(0);
                                auto tmp14 = tmp12 == tmp13;
                                auto tmp15 = static_cast<float>(-1000000000.0);
                                auto tmp17 = tmp14 ? tmp15 : tmp16;
                                auto tmp19 = tmp17 - tmp18;
                                auto tmp20 = std::exp(tmp19);
                                out_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)] = tmp20;
                            }
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (21*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<21; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (21*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (21*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (21*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<21; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (21*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (21*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        out_ptr4[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_59 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<21; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1344*(i2 / 64)) + (10752*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_60 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_61 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (11264*i0)];
                            out_ptr1[i2 + (22*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_62 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<168; i1+=1)
            {
                {
                    {
                        float tmp8 = -std::numeric_limits<float>::infinity();
                        for(long i2=0; i2<22; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i2)];
                                auto tmp6 = in_ptr1[i2 + (22*i1) + (3696*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<bool>(0);
                                auto tmp4 = tmp2 == tmp3;
                                auto tmp5 = static_cast<float>(-1000000000.0);
                                auto tmp7 = tmp4 ? tmp5 : tmp6;
                                tmp8 = std::max(tmp8, tmp7);
                            }
                        }
                        out_ptr0[i1 + (168*i0)] = tmp8;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<168; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i0 + (32*i2)];
                            auto tmp6 = in_ptr1[i2 + (22*i1) + (3696*i0)];
                            auto tmp8 = out_ptr0[i1 + (168*i0)];
                            auto tmp1 = static_cast<long>(1);
                            auto tmp2 = tmp0 != tmp1;
                            auto tmp3 = static_cast<bool>(0);
                            auto tmp4 = tmp2 == tmp3;
                            auto tmp5 = static_cast<float>(-1000000000.0);
                            auto tmp7 = tmp4 ? tmp5 : tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = std::exp(tmp9);
                            out_ptr1[i2 + (22*i1) + (3696*i0)] = tmp10;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<22; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (22*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (22*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<22; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (22*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (22*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        out_ptr4[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_63 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<21; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1344*(i2 / 64)) + (10752*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_64 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_65 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<86016; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=1376256; i0<1376256; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_66 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_67 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (10752*i0)];
                            out_ptr1[i2 + (21*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_68 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    {
                        {
                            float tmp18 = -std::numeric_limits<float>::infinity();
                            for(long i3=0; i3<21; i3+=1)
                            {
                                {
                                    auto tmp0 = in_ptr0[i0 + (32*i3)];
                                    auto tmp16 = in_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)];
                                    auto tmp1 = static_cast<long>(1);
                                    auto tmp2 = tmp0 != tmp1;
                                    auto tmp3 = static_cast<float>(1.0);
                                    auto tmp4 = static_cast<int>((-1) + i3 + ((-1)*i2));
                                    auto tmp5 = static_cast<int>(0);
                                    auto tmp6 = tmp4 >= tmp5;
                                    auto tmp7 = static_cast<float>(1);
                                    auto tmp8 = static_cast<float>(0);
                                    auto tmp9 = tmp6 ? tmp7 : tmp8;
                                    auto tmp10 = tmp3 - tmp9;
                                    auto tmp11 = static_cast<bool>(tmp10);
                                    auto tmp12 = tmp2 & tmp11;
                                    auto tmp13 = static_cast<bool>(0);
                                    auto tmp14 = tmp12 == tmp13;
                                    auto tmp15 = static_cast<float>(-1000000000.0);
                                    auto tmp17 = tmp14 ? tmp15 : tmp16;
                                    tmp18 = std::max(tmp18, tmp17);
                                }
                            }
                            out_ptr0[i2 + (21*i1) + (168*i0)] = tmp18;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    #pragma GCC ivdep
                    for(long i3=0; i3<21; i3+=1)
                    {
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i3)];
                                auto tmp16 = in_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)];
                                auto tmp18 = out_ptr0[i2 + (21*i1) + (168*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<float>(1.0);
                                auto tmp4 = static_cast<int>((-1) + i3 + ((-1)*i2));
                                auto tmp5 = static_cast<int>(0);
                                auto tmp6 = tmp4 >= tmp5;
                                auto tmp7 = static_cast<float>(1);
                                auto tmp8 = static_cast<float>(0);
                                auto tmp9 = tmp6 ? tmp7 : tmp8;
                                auto tmp10 = tmp3 - tmp9;
                                auto tmp11 = static_cast<bool>(tmp10);
                                auto tmp12 = tmp2 & tmp11;
                                auto tmp13 = static_cast<bool>(0);
                                auto tmp14 = tmp12 == tmp13;
                                auto tmp15 = static_cast<float>(-1000000000.0);
                                auto tmp17 = tmp14 ? tmp15 : tmp16;
                                auto tmp19 = tmp17 - tmp18;
                                auto tmp20 = std::exp(tmp19);
                                out_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)] = tmp20;
                            }
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (21*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<21; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (21*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (21*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (21*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<21; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (21*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (21*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        out_ptr4[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_69 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<21; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1344*(i2 / 64)) + (10752*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_70 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_71 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (11264*i0)];
                            out_ptr1[i2 + (22*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_72 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<168; i1+=1)
            {
                {
                    {
                        float tmp8 = -std::numeric_limits<float>::infinity();
                        for(long i2=0; i2<22; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i2)];
                                auto tmp6 = in_ptr1[i2 + (22*i1) + (3696*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<bool>(0);
                                auto tmp4 = tmp2 == tmp3;
                                auto tmp5 = static_cast<float>(-1000000000.0);
                                auto tmp7 = tmp4 ? tmp5 : tmp6;
                                tmp8 = std::max(tmp8, tmp7);
                            }
                        }
                        out_ptr0[i1 + (168*i0)] = tmp8;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<168; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i0 + (32*i2)];
                            auto tmp6 = in_ptr1[i2 + (22*i1) + (3696*i0)];
                            auto tmp8 = out_ptr0[i1 + (168*i0)];
                            auto tmp1 = static_cast<long>(1);
                            auto tmp2 = tmp0 != tmp1;
                            auto tmp3 = static_cast<bool>(0);
                            auto tmp4 = tmp2 == tmp3;
                            auto tmp5 = static_cast<float>(-1000000000.0);
                            auto tmp7 = tmp4 ? tmp5 : tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = std::exp(tmp9);
                            out_ptr1[i2 + (22*i1) + (3696*i0)] = tmp10;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<22; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (22*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (22*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<22; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (22*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (22*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        out_ptr4[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_73 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<21; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1344*(i2 / 64)) + (10752*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_74 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_75 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<86016; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=1376256; i0<1376256; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_76 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_77 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (10752*i0)];
                            out_ptr1[i2 + (21*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_78 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    {
                        {
                            float tmp18 = -std::numeric_limits<float>::infinity();
                            for(long i3=0; i3<21; i3+=1)
                            {
                                {
                                    auto tmp0 = in_ptr0[i0 + (32*i3)];
                                    auto tmp16 = in_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)];
                                    auto tmp1 = static_cast<long>(1);
                                    auto tmp2 = tmp0 != tmp1;
                                    auto tmp3 = static_cast<float>(1.0);
                                    auto tmp4 = static_cast<int>((-1) + i3 + ((-1)*i2));
                                    auto tmp5 = static_cast<int>(0);
                                    auto tmp6 = tmp4 >= tmp5;
                                    auto tmp7 = static_cast<float>(1);
                                    auto tmp8 = static_cast<float>(0);
                                    auto tmp9 = tmp6 ? tmp7 : tmp8;
                                    auto tmp10 = tmp3 - tmp9;
                                    auto tmp11 = static_cast<bool>(tmp10);
                                    auto tmp12 = tmp2 & tmp11;
                                    auto tmp13 = static_cast<bool>(0);
                                    auto tmp14 = tmp12 == tmp13;
                                    auto tmp15 = static_cast<float>(-1000000000.0);
                                    auto tmp17 = tmp14 ? tmp15 : tmp16;
                                    tmp18 = std::max(tmp18, tmp17);
                                }
                            }
                            out_ptr0[i2 + (21*i1) + (168*i0)] = tmp18;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    #pragma GCC ivdep
                    for(long i3=0; i3<21; i3+=1)
                    {
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i3)];
                                auto tmp16 = in_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)];
                                auto tmp18 = out_ptr0[i2 + (21*i1) + (168*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<float>(1.0);
                                auto tmp4 = static_cast<int>((-1) + i3 + ((-1)*i2));
                                auto tmp5 = static_cast<int>(0);
                                auto tmp6 = tmp4 >= tmp5;
                                auto tmp7 = static_cast<float>(1);
                                auto tmp8 = static_cast<float>(0);
                                auto tmp9 = tmp6 ? tmp7 : tmp8;
                                auto tmp10 = tmp3 - tmp9;
                                auto tmp11 = static_cast<bool>(tmp10);
                                auto tmp12 = tmp2 & tmp11;
                                auto tmp13 = static_cast<bool>(0);
                                auto tmp14 = tmp12 == tmp13;
                                auto tmp15 = static_cast<float>(-1000000000.0);
                                auto tmp17 = tmp14 ? tmp15 : tmp16;
                                auto tmp19 = tmp17 - tmp18;
                                auto tmp20 = std::exp(tmp19);
                                out_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)] = tmp20;
                            }
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (21*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<21; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (21*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (21*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (21*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<21; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (21*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (21*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        out_ptr4[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_79 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<21; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1344*(i2 / 64)) + (10752*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_80 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_81 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (11264*i0)];
                            out_ptr1[i2 + (22*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_82 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<168; i1+=1)
            {
                {
                    {
                        float tmp8 = -std::numeric_limits<float>::infinity();
                        for(long i2=0; i2<22; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i2)];
                                auto tmp6 = in_ptr1[i2 + (22*i1) + (3696*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<bool>(0);
                                auto tmp4 = tmp2 == tmp3;
                                auto tmp5 = static_cast<float>(-1000000000.0);
                                auto tmp7 = tmp4 ? tmp5 : tmp6;
                                tmp8 = std::max(tmp8, tmp7);
                            }
                        }
                        out_ptr0[i1 + (168*i0)] = tmp8;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<168; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i0 + (32*i2)];
                            auto tmp6 = in_ptr1[i2 + (22*i1) + (3696*i0)];
                            auto tmp8 = out_ptr0[i1 + (168*i0)];
                            auto tmp1 = static_cast<long>(1);
                            auto tmp2 = tmp0 != tmp1;
                            auto tmp3 = static_cast<bool>(0);
                            auto tmp4 = tmp2 == tmp3;
                            auto tmp5 = static_cast<float>(-1000000000.0);
                            auto tmp7 = tmp4 ? tmp5 : tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = std::exp(tmp9);
                            out_ptr1[i2 + (22*i1) + (3696*i0)] = tmp10;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<22; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (22*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (22*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<22; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (22*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (22*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        out_ptr4[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_83 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<21; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1344*(i2 / 64)) + (10752*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_84 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_85 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<86016; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=1376256; i0<1376256; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_86 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_87 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (10752*i0)];
                            out_ptr1[i2 + (21*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_88 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    {
                        {
                            float tmp18 = -std::numeric_limits<float>::infinity();
                            for(long i3=0; i3<21; i3+=1)
                            {
                                {
                                    auto tmp0 = in_ptr0[i0 + (32*i3)];
                                    auto tmp16 = in_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)];
                                    auto tmp1 = static_cast<long>(1);
                                    auto tmp2 = tmp0 != tmp1;
                                    auto tmp3 = static_cast<float>(1.0);
                                    auto tmp4 = static_cast<int>((-1) + i3 + ((-1)*i2));
                                    auto tmp5 = static_cast<int>(0);
                                    auto tmp6 = tmp4 >= tmp5;
                                    auto tmp7 = static_cast<float>(1);
                                    auto tmp8 = static_cast<float>(0);
                                    auto tmp9 = tmp6 ? tmp7 : tmp8;
                                    auto tmp10 = tmp3 - tmp9;
                                    auto tmp11 = static_cast<bool>(tmp10);
                                    auto tmp12 = tmp2 & tmp11;
                                    auto tmp13 = static_cast<bool>(0);
                                    auto tmp14 = tmp12 == tmp13;
                                    auto tmp15 = static_cast<float>(-1000000000.0);
                                    auto tmp17 = tmp14 ? tmp15 : tmp16;
                                    tmp18 = std::max(tmp18, tmp17);
                                }
                            }
                            out_ptr0[i2 + (21*i1) + (168*i0)] = tmp18;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    #pragma GCC ivdep
                    for(long i3=0; i3<21; i3+=1)
                    {
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i3)];
                                auto tmp16 = in_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)];
                                auto tmp18 = out_ptr0[i2 + (21*i1) + (168*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<float>(1.0);
                                auto tmp4 = static_cast<int>((-1) + i3 + ((-1)*i2));
                                auto tmp5 = static_cast<int>(0);
                                auto tmp6 = tmp4 >= tmp5;
                                auto tmp7 = static_cast<float>(1);
                                auto tmp8 = static_cast<float>(0);
                                auto tmp9 = tmp6 ? tmp7 : tmp8;
                                auto tmp10 = tmp3 - tmp9;
                                auto tmp11 = static_cast<bool>(tmp10);
                                auto tmp12 = tmp2 & tmp11;
                                auto tmp13 = static_cast<bool>(0);
                                auto tmp14 = tmp12 == tmp13;
                                auto tmp15 = static_cast<float>(-1000000000.0);
                                auto tmp17 = tmp14 ? tmp15 : tmp16;
                                auto tmp19 = tmp17 - tmp18;
                                auto tmp20 = std::exp(tmp19);
                                out_ptr1[i3 + (21*i2) + (441*i1) + (3528*i0)] = tmp20;
                            }
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (21*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<21; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (21*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (21*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (21*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<21; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (21*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (21*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        out_ptr4[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_89 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<21; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1344*(i2 / 64)) + (10752*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_90 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_91 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<21; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (16*i3) + (64*i1) + (512*i2) + (10752*i0));
                        auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(8.0));
                        auto tmp2 = tmp0 / tmp1;
                        tmp2.store(out_ptr0 + (16*i3) + (64*i2) + (1344*i1) + (10752*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr0[i3 + (64*i1) + (512*i2) + (10752*i0)];
                        auto tmp1 = static_cast<float>(8.0);
                        auto tmp2 = tmp0 / tmp1;
                        out_ptr0[i3 + (64*i2) + (1344*i1) + (10752*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<512; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr1[i1 + (512*i2) + (11264*i0)];
                            out_ptr1[i2 + (22*i1) + (11264*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_92 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const long* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr4)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr0;
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<168; i1+=1)
            {
                {
                    {
                        float tmp8 = -std::numeric_limits<float>::infinity();
                        for(long i2=0; i2<22; i2+=1)
                        {
                            {
                                auto tmp0 = in_ptr0[i0 + (32*i2)];
                                auto tmp6 = in_ptr1[i2 + (22*i1) + (3696*i0)];
                                auto tmp1 = static_cast<long>(1);
                                auto tmp2 = tmp0 != tmp1;
                                auto tmp3 = static_cast<bool>(0);
                                auto tmp4 = tmp2 == tmp3;
                                auto tmp5 = static_cast<float>(-1000000000.0);
                                auto tmp7 = tmp4 ? tmp5 : tmp6;
                                tmp8 = std::max(tmp8, tmp7);
                            }
                        }
                        out_ptr0[i1 + (168*i0)] = tmp8;
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<168; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i0 + (32*i2)];
                            auto tmp6 = in_ptr1[i2 + (22*i1) + (3696*i0)];
                            auto tmp8 = out_ptr0[i1 + (168*i0)];
                            auto tmp1 = static_cast<long>(1);
                            auto tmp2 = tmp0 != tmp1;
                            auto tmp3 = static_cast<bool>(0);
                            auto tmp4 = tmp2 == tmp3;
                            auto tmp5 = static_cast<float>(-1000000000.0);
                            auto tmp7 = tmp4 ? tmp5 : tmp6;
                            auto tmp9 = tmp7 - tmp8;
                            auto tmp10 = std::exp(tmp9);
                            out_ptr1[i2 + (22*i1) + (3696*i0)] = tmp10;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<1; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=16; i1<22; i1+=1)
                {
                    auto tmp0 = out_ptr1[i1 + (22*i0)];
                    tmp1 += tmp0;
                }
                out_ptr2[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<5376; i0+=1)
        {
            for(long i1=0; i1<1; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + (16*i1) + (22*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp2 = tmp0 / tmp1;
                tmp2.store(out_ptr3 + (16*i1) + (22*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=16; i1<22; i1+=1)
            {
                auto tmp0 = out_ptr1[i1 + (22*i0)];
                auto tmp1 = out_ptr2[i0];
                auto tmp2 = tmp0 / tmp1;
                out_ptr3[i1 + (22*i0)] = tmp2;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<8; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<22; i2+=1)
                {
                    for(long i3=0; i3<4; i3+=1)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + (16*i3) + (64*i1) + (512*i2) + (11264*i0));
                        tmp0.store(out_ptr4 + (16*i3) + (64*i2) + (1408*i1) + (11264*i0));
                    }
                    #pragma omp simd simdlen(8) 
                    for(long i3=64; i3<64; i3+=1)
                    {
                        auto tmp0 = in_ptr2[i3 + (64*i1) + (512*i2) + (11264*i0)];
                        out_ptr4[i3 + (64*i2) + (1408*i1) + (11264*i0)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_93 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<32; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<21; i1+=1)
            {
                #pragma GCC ivdep
                for(long i2=0; i2<512; i2+=1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[(64*i1) + (1344*(i2 / 64)) + (10752*i0) + (i2 % 64)];
                            out_ptr0[i2 + (512*i1) + (10752*i0)] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_94 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_95 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<86016; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=1376256; i0<1376256; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            out_ptr0[i0] = tmp1;
        }
    }
}
''')


kernel_cpp_96 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<21504; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=344064; i0<344064; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr0[i0] = tmp2;
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp1 = 0;
                auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    tmp1_vec += tmp0;
                }
                tmp1 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp1_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp1)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    tmp1 += tmp0;
                }
                out_ptr1[i0] = tmp1;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out += omp_in) initializer(omp_priv={{0}})
                float tmp6 = 0;
                auto tmp6_vec = at::vec::Vectorized<float>(tmp6);
                float tmp7 = 0;
                auto tmp7_vec = at::vec::Vectorized<float>(tmp7);
                for(long i1=0; i1<32; i1+=1)
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                    auto tmp1 = at::vec::Vectorized<float>(out_ptr1[i0]);
                    auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.pow(2);
                    tmp6_vec += tmp5;
                    tmp7_vec += tmp0;
                }
                tmp6 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp6_vec);
                tmp7 = at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;}, tmp7_vec);
                #pragma omp simd simdlen(8)  reduction(+:tmp6) reduction(+:tmp7)
                for(long i1=512; i1<512; i1+=1)
                {
                    auto tmp0 = out_ptr0[i1 + (512*i0)];
                    auto tmp1 = out_ptr1[i0];
                    auto tmp2 = static_cast<float>(512);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4 * tmp4;
                    tmp6 += tmp5;
                    tmp7 += tmp0;
                }
                out_ptr2[i0] = tmp6;
                out_ptr3[i0] = tmp7;
            }
        }
        #pragma omp for 
        for(long i0=0; i0<672; i0+=1)
        {
            for(long i1=0; i1<32; i1+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + (16*i1) + (512*i0));
                auto tmp1 = at::vec::Vectorized<float>(out_ptr3[i0]);
                auto tmp5 = at::vec::Vectorized<float>(out_ptr2[i0]);
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i1);
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i1);
                auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(512));
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = at::vec::Vectorized<float>(static_cast<float>(1e-06));
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = tmp8.sqrt();
                auto tmp10 = tmp9.reciprocal();
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr4 + (16*i1) + (512*i0));
            }
            #pragma omp simd simdlen(8) 
            for(long i1=512; i1<512; i1+=1)
            {
                auto tmp0 = out_ptr0[i1 + (512*i0)];
                auto tmp1 = out_ptr3[i0];
                auto tmp5 = out_ptr2[i0];
                auto tmp12 = in_ptr2[i1];
                auto tmp14 = in_ptr3[i1];
                auto tmp2 = static_cast<float>(512);
                auto tmp3 = tmp1 / tmp2;
                auto tmp4 = tmp0 - tmp3;
                auto tmp6 = tmp5 / tmp2;
                auto tmp7 = static_cast<float>(1e-06);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = std::sqrt(tmp8);
                auto tmp10 = 1 / tmp9;
                auto tmp11 = tmp4 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                out_ptr4[i1 + (512*i0)] = tmp15;
            }
        }
    }
}
''')


kernel_cpp_97 = async_compile.cpp('''
#include "/tmp/tmpjix3bp42/rp/crpdeql3xwpfmcyakwtqpzihz525if6mt25mozau77xvmnh7vqyu.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        #pragma omp for 
        for(long i0=0; i0<399882; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>(static_cast<float>(1.0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr0 + 16*i0);
        }
        #pragma omp for simd simdlen(8) 
        for(long i0=6398112; i0<6398112; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = static_cast<float>(1.0);
            auto tmp2 = tmp0 * tmp1;
            out_ptr0[i0] = tmp2;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1 = args
    args.clear()
    buf0 = empty_strided((32, 22, 1), (1, 32, 704), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((32, 22, 1), (1, 32, 704), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 22, 1), (1, 32, 704), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((32, 22, 512), (11264, 512, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_0(c_void_p(arg286_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg0_1
    del arg1_1
    del arg284_1
    del arg2_1
    buf4 = torch.ops.mkl._mkl_linear(buf3, arg4_1, arg3_1, None, 704)
    del arg3_1
    del arg4_1
    buf5 = torch.ops.mkl._mkl_linear(buf3, arg6_1, arg5_1, None, 704)
    del arg5_1
    del arg6_1
    buf6 = torch.ops.mkl._mkl_linear(buf3, arg8_1, arg7_1, None, 704)
    del arg7_1
    del arg8_1
    buf7 = empty_strided((32, 8, 22, 64), (11264, 1408, 64, 1), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((32, 8, 64, 22), (11264, 1408, 22, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_1(c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del buf4
    del buf5
    buf9 = empty_strided((256, 22, 22), (484, 22, 1), device='cpu', dtype=torch.float32)
    aten.bmm.out(as_strided(buf7, (256, 22, 64), (1408, 64, 1)), as_strided(buf8, (256, 64, 22), (1408, 22, 1)), out=buf9)
    del buf7
    buf10 = empty_strided((32, 8, 22, 1), (176, 22, 1, 5632), device='cpu', dtype=torch.float32)
    buf11 = as_strided(buf9, (32, 8, 22, 22), (3872, 484, 22, 1)); del buf9  # reuse
    buf12 = empty_strided((32, 8, 22, 1), (176, 22, 1, 5632), device='cpu', dtype=torch.float32)
    buf13 = buf11; del buf11  # reuse
    buf14 = as_strided(buf8, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf8  # reuse
    kernel_cpp_2(c_void_p(buf13.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf14.data_ptr()))
    buf15 = as_strided(buf6, (256, 22, 64), (1408, 64, 1)); del buf6  # reuse
    aten.bmm.out(as_strided(buf13, (256, 22, 22), (484, 22, 1)), as_strided(buf14, (256, 22, 64), (1408, 64, 1)), out=buf15)
    buf16 = as_strided(buf14, (32, 22, 512), (11264, 512, 1)); del buf14  # reuse
    kernel_cpp_3(c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()))
    del buf15
    buf17 = torch.ops.mkl._mkl_linear(buf16, arg10_1, arg9_1, None, 704)
    del arg10_1
    del arg9_1
    del buf16
    buf18 = as_strided(buf17, (32, 22, 512), (11264, 512, 1)); del buf17  # reuse
    buf19 = as_strided(buf2, (32, 22, 1), (22, 1, 704)); del buf2  # reuse
    buf20 = as_strided(buf1, (32, 22, 1), (22, 1, 704)); del buf1  # reuse
    buf21 = as_strided(buf0, (32, 22, 1), (22, 1, 704)); del buf0  # reuse
    buf22 = buf18; del buf18  # reuse
    kernel_cpp_4(c_void_p(buf22.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()))
    del arg11_1
    del arg12_1
    buf23 = torch.ops.mkl._mkl_linear(buf22, arg15_1, arg13_1, arg14_1, 704)
    del arg13_1
    del arg14_1
    del arg15_1
    buf24 = as_strided(buf23, (32, 22, 2048), (45056, 2048, 1)); del buf23  # reuse
    kernel_cpp_5(c_void_p(buf24.data_ptr()))
    buf25 = torch.ops.mkl._mkl_linear(buf24, arg18_1, arg16_1, arg17_1, 704)
    del arg16_1
    del arg17_1
    del arg18_1
    del buf24
    buf26 = as_strided(buf25, (32, 22, 512), (11264, 512, 1)); del buf25  # reuse
    buf27 = buf21; del buf21  # reuse
    buf28 = buf20; del buf20  # reuse
    buf29 = buf19; del buf19  # reuse
    buf30 = buf26; del buf26  # reuse
    kernel_cpp_6(c_void_p(buf30.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    del arg19_1
    del arg20_1
    buf31 = torch.ops.mkl._mkl_linear(buf30, arg22_1, arg21_1, None, 704)
    del arg21_1
    del arg22_1
    buf32 = torch.ops.mkl._mkl_linear(buf30, arg24_1, arg23_1, None, 704)
    del arg23_1
    del arg24_1
    buf33 = torch.ops.mkl._mkl_linear(buf30, arg26_1, arg25_1, None, 704)
    del arg25_1
    del arg26_1
    buf34 = as_strided(buf22, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf22  # reuse
    buf35 = as_strided(buf3, (32, 8, 64, 22), (11264, 1408, 22, 1)); del buf3  # reuse
    kernel_cpp_7(c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()))
    del buf31
    del buf32
    buf36 = as_strided(buf13, (256, 22, 22), (484, 22, 1)); del buf13  # reuse
    aten.bmm.out(as_strided(buf34, (256, 22, 64), (1408, 64, 1)), as_strided(buf35, (256, 64, 22), (1408, 22, 1)), out=buf36)
    del buf34
    buf37 = buf12; del buf12  # reuse
    buf38 = as_strided(buf36, (32, 8, 22, 22), (3872, 484, 22, 1)); del buf36  # reuse
    buf39 = buf10; del buf10  # reuse
    buf40 = buf38; del buf38  # reuse
    buf41 = as_strided(buf35, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf35  # reuse
    kernel_cpp_8(c_void_p(buf40.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf41.data_ptr()))
    buf42 = as_strided(buf33, (256, 22, 64), (1408, 64, 1)); del buf33  # reuse
    aten.bmm.out(as_strided(buf40, (256, 22, 22), (484, 22, 1)), as_strided(buf41, (256, 22, 64), (1408, 64, 1)), out=buf42)
    buf43 = as_strided(buf41, (32, 22, 512), (11264, 512, 1)); del buf41  # reuse
    kernel_cpp_9(c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()))
    del buf42
    buf44 = torch.ops.mkl._mkl_linear(buf43, arg28_1, arg27_1, None, 704)
    del arg27_1
    del arg28_1
    del buf43
    buf45 = as_strided(buf44, (32, 22, 512), (11264, 512, 1)); del buf44  # reuse
    buf46 = buf29; del buf29  # reuse
    buf47 = buf28; del buf28  # reuse
    buf48 = buf27; del buf27  # reuse
    buf49 = buf45; del buf45  # reuse
    kernel_cpp_10(c_void_p(buf49.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del arg29_1
    del arg30_1
    buf50 = torch.ops.mkl._mkl_linear(buf49, arg33_1, arg31_1, arg32_1, 704)
    del arg31_1
    del arg32_1
    del arg33_1
    buf51 = as_strided(buf50, (32, 22, 2048), (45056, 2048, 1)); del buf50  # reuse
    kernel_cpp_11(c_void_p(buf51.data_ptr()))
    buf52 = torch.ops.mkl._mkl_linear(buf51, arg36_1, arg34_1, arg35_1, 704)
    del arg34_1
    del arg35_1
    del arg36_1
    del buf51
    buf53 = as_strided(buf52, (32, 22, 512), (11264, 512, 1)); del buf52  # reuse
    buf54 = buf48; del buf48  # reuse
    buf55 = buf47; del buf47  # reuse
    buf56 = buf46; del buf46  # reuse
    buf57 = buf53; del buf53  # reuse
    kernel_cpp_12(c_void_p(buf57.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    del arg37_1
    del arg38_1
    buf58 = torch.ops.mkl._mkl_linear(buf57, arg40_1, arg39_1, None, 704)
    del arg39_1
    del arg40_1
    buf59 = torch.ops.mkl._mkl_linear(buf57, arg42_1, arg41_1, None, 704)
    del arg41_1
    del arg42_1
    buf60 = torch.ops.mkl._mkl_linear(buf57, arg44_1, arg43_1, None, 704)
    del arg43_1
    del arg44_1
    buf61 = as_strided(buf49, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf49  # reuse
    buf62 = as_strided(buf30, (32, 8, 64, 22), (11264, 1408, 22, 1)); del buf30  # reuse
    kernel_cpp_13(c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del buf58
    del buf59
    buf63 = as_strided(buf40, (256, 22, 22), (484, 22, 1)); del buf40  # reuse
    aten.bmm.out(as_strided(buf61, (256, 22, 64), (1408, 64, 1)), as_strided(buf62, (256, 64, 22), (1408, 22, 1)), out=buf63)
    del buf61
    buf64 = buf39; del buf39  # reuse
    buf65 = as_strided(buf63, (32, 8, 22, 22), (3872, 484, 22, 1)); del buf63  # reuse
    buf66 = buf37; del buf37  # reuse
    buf67 = buf65; del buf65  # reuse
    buf68 = as_strided(buf62, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf62  # reuse
    kernel_cpp_14(c_void_p(buf67.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf68.data_ptr()))
    buf69 = as_strided(buf60, (256, 22, 64), (1408, 64, 1)); del buf60  # reuse
    aten.bmm.out(as_strided(buf67, (256, 22, 22), (484, 22, 1)), as_strided(buf68, (256, 22, 64), (1408, 64, 1)), out=buf69)
    buf70 = as_strided(buf68, (32, 22, 512), (11264, 512, 1)); del buf68  # reuse
    kernel_cpp_15(c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del buf69
    buf71 = torch.ops.mkl._mkl_linear(buf70, arg46_1, arg45_1, None, 704)
    del arg45_1
    del arg46_1
    del buf70
    buf72 = as_strided(buf71, (32, 22, 512), (11264, 512, 1)); del buf71  # reuse
    buf73 = buf56; del buf56  # reuse
    buf74 = buf55; del buf55  # reuse
    buf75 = buf54; del buf54  # reuse
    buf76 = buf72; del buf72  # reuse
    kernel_cpp_16(c_void_p(buf76.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    del arg47_1
    del arg48_1
    buf77 = torch.ops.mkl._mkl_linear(buf76, arg51_1, arg49_1, arg50_1, 704)
    del arg49_1
    del arg50_1
    del arg51_1
    buf78 = as_strided(buf77, (32, 22, 2048), (45056, 2048, 1)); del buf77  # reuse
    kernel_cpp_17(c_void_p(buf78.data_ptr()))
    buf79 = torch.ops.mkl._mkl_linear(buf78, arg54_1, arg52_1, arg53_1, 704)
    del arg52_1
    del arg53_1
    del arg54_1
    del buf78
    buf80 = as_strided(buf79, (32, 22, 512), (11264, 512, 1)); del buf79  # reuse
    buf81 = buf75; del buf75  # reuse
    buf82 = buf74; del buf74  # reuse
    buf83 = buf73; del buf73  # reuse
    buf84 = buf80; del buf80  # reuse
    kernel_cpp_18(c_void_p(buf84.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    del arg55_1
    del arg56_1
    buf85 = torch.ops.mkl._mkl_linear(buf84, arg58_1, arg57_1, None, 704)
    del arg57_1
    del arg58_1
    buf86 = torch.ops.mkl._mkl_linear(buf84, arg60_1, arg59_1, None, 704)
    del arg59_1
    del arg60_1
    buf87 = torch.ops.mkl._mkl_linear(buf84, arg62_1, arg61_1, None, 704)
    del arg61_1
    del arg62_1
    buf88 = as_strided(buf76, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf76  # reuse
    buf89 = as_strided(buf57, (32, 8, 64, 22), (11264, 1408, 22, 1)); del buf57  # reuse
    kernel_cpp_19(c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()))
    del buf85
    del buf86
    buf90 = as_strided(buf67, (256, 22, 22), (484, 22, 1)); del buf67  # reuse
    aten.bmm.out(as_strided(buf88, (256, 22, 64), (1408, 64, 1)), as_strided(buf89, (256, 64, 22), (1408, 22, 1)), out=buf90)
    del buf88
    buf91 = buf66; del buf66  # reuse
    buf92 = as_strided(buf90, (32, 8, 22, 22), (3872, 484, 22, 1)); del buf90  # reuse
    buf93 = buf64; del buf64  # reuse
    buf94 = buf92; del buf92  # reuse
    buf95 = as_strided(buf89, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf89  # reuse
    kernel_cpp_20(c_void_p(buf94.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()))
    buf96 = as_strided(buf87, (256, 22, 64), (1408, 64, 1)); del buf87  # reuse
    aten.bmm.out(as_strided(buf94, (256, 22, 22), (484, 22, 1)), as_strided(buf95, (256, 22, 64), (1408, 64, 1)), out=buf96)
    buf97 = as_strided(buf95, (32, 22, 512), (11264, 512, 1)); del buf95  # reuse
    kernel_cpp_21(c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del buf96
    buf98 = torch.ops.mkl._mkl_linear(buf97, arg64_1, arg63_1, None, 704)
    del arg63_1
    del arg64_1
    del buf97
    buf99 = buf84; del buf84  # reuse
    buf100 = buf83; del buf83  # reuse
    buf101 = buf82; del buf82  # reuse
    buf102 = buf81; del buf81  # reuse
    buf103 = buf99; del buf99  # reuse
    kernel_cpp_22(c_void_p(buf103.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()))
    del arg65_1
    del arg66_1
    buf104 = torch.ops.mkl._mkl_linear(buf103, arg69_1, arg67_1, arg68_1, 704)
    del arg67_1
    del arg68_1
    del arg69_1
    buf105 = as_strided(buf104, (32, 22, 2048), (45056, 2048, 1)); del buf104  # reuse
    kernel_cpp_23(c_void_p(buf105.data_ptr()))
    buf106 = torch.ops.mkl._mkl_linear(buf105, arg72_1, arg70_1, arg71_1, 704)
    del arg70_1
    del arg71_1
    del arg72_1
    del buf105
    buf107 = buf103; del buf103  # reuse
    buf108 = buf102; del buf102  # reuse
    buf109 = buf101; del buf101  # reuse
    buf110 = buf100; del buf100  # reuse
    buf111 = buf107; del buf107  # reuse
    kernel_cpp_24(c_void_p(buf111.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()))
    del arg73_1
    del arg74_1
    buf112 = torch.ops.mkl._mkl_linear(buf111, arg76_1, arg75_1, None, 704)
    del arg75_1
    del arg76_1
    buf113 = torch.ops.mkl._mkl_linear(buf111, arg78_1, arg77_1, None, 704)
    del arg77_1
    del arg78_1
    buf114 = torch.ops.mkl._mkl_linear(buf111, arg80_1, arg79_1, None, 704)
    del arg79_1
    del arg80_1
    buf115 = as_strided(buf106, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf106  # reuse
    buf116 = as_strided(buf98, (32, 8, 64, 22), (11264, 1408, 22, 1)); del buf98  # reuse
    kernel_cpp_25(c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    del buf112
    del buf113
    buf117 = as_strided(buf94, (256, 22, 22), (484, 22, 1)); del buf94  # reuse
    aten.bmm.out(as_strided(buf115, (256, 22, 64), (1408, 64, 1)), as_strided(buf116, (256, 64, 22), (1408, 22, 1)), out=buf117)
    del buf115
    buf118 = buf93; del buf93  # reuse
    buf119 = as_strided(buf117, (32, 8, 22, 22), (3872, 484, 22, 1)); del buf117  # reuse
    buf120 = buf91; del buf91  # reuse
    buf121 = buf119; del buf119  # reuse
    buf122 = as_strided(buf116, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf116  # reuse
    kernel_cpp_26(c_void_p(buf121.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()))
    buf123 = as_strided(buf114, (256, 22, 64), (1408, 64, 1)); del buf114  # reuse
    aten.bmm.out(as_strided(buf121, (256, 22, 22), (484, 22, 1)), as_strided(buf122, (256, 22, 64), (1408, 64, 1)), out=buf123)
    buf124 = as_strided(buf122, (32, 22, 512), (11264, 512, 1)); del buf122  # reuse
    kernel_cpp_27(c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()))
    del buf123
    buf125 = torch.ops.mkl._mkl_linear(buf124, arg82_1, arg81_1, None, 704)
    del arg81_1
    del arg82_1
    del buf124
    buf126 = as_strided(buf125, (32, 22, 512), (11264, 512, 1)); del buf125  # reuse
    buf127 = buf110; del buf110  # reuse
    buf128 = buf109; del buf109  # reuse
    buf129 = buf108; del buf108  # reuse
    buf130 = buf126; del buf126  # reuse
    kernel_cpp_28(c_void_p(buf130.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()))
    del arg83_1
    del arg84_1
    buf131 = torch.ops.mkl._mkl_linear(buf130, arg87_1, arg85_1, arg86_1, 704)
    del arg85_1
    del arg86_1
    del arg87_1
    buf132 = as_strided(buf131, (32, 22, 2048), (45056, 2048, 1)); del buf131  # reuse
    kernel_cpp_29(c_void_p(buf132.data_ptr()))
    buf133 = torch.ops.mkl._mkl_linear(buf132, arg90_1, arg88_1, arg89_1, 704)
    del arg88_1
    del arg89_1
    del arg90_1
    del buf132
    buf134 = buf130; del buf130  # reuse
    buf135 = buf129; del buf129  # reuse
    buf136 = buf128; del buf128  # reuse
    buf137 = buf127; del buf127  # reuse
    buf138 = buf134; del buf134  # reuse
    kernel_cpp_30(c_void_p(buf138.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()))
    del arg91_1
    del arg92_1
    buf139 = torch.ops.mkl._mkl_linear(buf138, arg94_1, arg93_1, None, 704)
    del arg93_1
    del arg94_1
    buf140 = torch.ops.mkl._mkl_linear(buf138, arg96_1, arg95_1, None, 704)
    del arg95_1
    del arg96_1
    buf141 = torch.ops.mkl._mkl_linear(buf138, arg98_1, arg97_1, None, 704)
    del arg97_1
    del arg98_1
    buf142 = as_strided(buf133, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf133  # reuse
    buf143 = as_strided(buf111, (32, 8, 64, 22), (11264, 1408, 22, 1)); del buf111  # reuse
    kernel_cpp_31(c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    del buf139
    del buf140
    buf144 = as_strided(buf121, (256, 22, 22), (484, 22, 1)); del buf121  # reuse
    aten.bmm.out(as_strided(buf142, (256, 22, 64), (1408, 64, 1)), as_strided(buf143, (256, 64, 22), (1408, 22, 1)), out=buf144)
    del buf142
    buf145 = buf120; del buf120  # reuse
    buf146 = as_strided(buf144, (32, 8, 22, 22), (3872, 484, 22, 1)); del buf144  # reuse
    buf147 = buf118; del buf118  # reuse
    buf148 = buf146; del buf146  # reuse
    buf149 = as_strided(buf143, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf143  # reuse
    kernel_cpp_32(c_void_p(buf148.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()))
    del buf145
    del buf147
    buf150 = as_strided(buf141, (256, 22, 64), (1408, 64, 1)); del buf141  # reuse
    aten.bmm.out(as_strided(buf148, (256, 22, 22), (484, 22, 1)), as_strided(buf149, (256, 22, 64), (1408, 64, 1)), out=buf150)
    del buf148
    buf151 = as_strided(buf149, (32, 22, 512), (11264, 512, 1)); del buf149  # reuse
    kernel_cpp_33(c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()))
    del buf150
    buf152 = torch.ops.mkl._mkl_linear(buf151, arg100_1, arg99_1, None, 704)
    del arg100_1
    del arg99_1
    del buf151
    buf153 = as_strided(buf152, (32, 22, 512), (11264, 512, 1)); del buf152  # reuse
    buf154 = buf137; del buf137  # reuse
    buf155 = buf136; del buf136  # reuse
    buf156 = buf135; del buf135  # reuse
    buf157 = buf153; del buf153  # reuse
    kernel_cpp_34(c_void_p(buf157.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()))
    del arg101_1
    del arg102_1
    del buf138
    buf158 = torch.ops.mkl._mkl_linear(buf157, arg105_1, arg103_1, arg104_1, 704)
    del arg103_1
    del arg104_1
    del arg105_1
    buf159 = as_strided(buf158, (32, 22, 2048), (45056, 2048, 1)); del buf158  # reuse
    kernel_cpp_35(c_void_p(buf159.data_ptr()))
    buf160 = torch.ops.mkl._mkl_linear(buf159, arg108_1, arg106_1, arg107_1, 704)
    del arg106_1
    del arg107_1
    del arg108_1
    del buf159
    buf161 = as_strided(buf160, (32, 22, 512), (11264, 512, 1)); del buf160  # reuse
    buf162 = buf156; del buf156  # reuse
    buf163 = buf155; del buf155  # reuse
    buf164 = buf154; del buf154  # reuse
    buf165 = buf161; del buf161  # reuse
    buf166 = empty_strided((32, 21, 1), (1, 32, 672), device='cpu', dtype=torch.float32)
    buf168 = empty_strided((32, 21, 1), (1, 32, 672), device='cpu', dtype=torch.float32)
    buf167 = empty_strided((32, 21, 1), (1, 32, 672), device='cpu', dtype=torch.float32)
    buf169 = empty_strided((32, 21, 512), (10752, 512, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_36(c_void_p(buf165.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf169.data_ptr()))
    del arg109_1
    del arg110_1
    del arg111_1
    del arg112_1
    del arg113_1
    del arg285_1
    del buf162
    del buf163
    del buf164
    buf170 = torch.ops.mkl._mkl_linear(buf169, arg115_1, arg114_1, None, 672)
    del arg114_1
    del arg115_1
    buf171 = torch.ops.mkl._mkl_linear(buf169, arg117_1, arg116_1, None, 672)
    del arg116_1
    del arg117_1
    buf172 = torch.ops.mkl._mkl_linear(buf169, arg119_1, arg118_1, None, 672)
    del arg118_1
    del arg119_1
    buf173 = empty_strided((32, 8, 21, 64), (10752, 1344, 64, 1), device='cpu', dtype=torch.float32)
    buf174 = empty_strided((32, 8, 64, 21), (10752, 1344, 21, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_37(c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    del buf170
    del buf171
    buf175 = empty_strided((256, 21, 21), (441, 21, 1), device='cpu', dtype=torch.float32)
    aten.bmm.out(as_strided(buf173, (256, 21, 64), (1344, 64, 1)), as_strided(buf174, (256, 64, 21), (1344, 21, 1)), out=buf175)
    del buf173
    buf176 = empty_strided((32, 8, 21, 1), (168, 21, 1, 5376), device='cpu', dtype=torch.float32)
    buf177 = as_strided(buf175, (32, 8, 21, 21), (3528, 441, 21, 1)); del buf175  # reuse
    buf178 = empty_strided((32, 8, 21, 1), (168, 21, 1, 5376), device='cpu', dtype=torch.float32)
    buf179 = buf177; del buf177  # reuse
    buf180 = as_strided(buf174, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf174  # reuse
    kernel_cpp_38(c_void_p(buf179.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()))
    buf181 = as_strided(buf172, (256, 21, 64), (1344, 64, 1)); del buf172  # reuse
    aten.bmm.out(as_strided(buf179, (256, 21, 21), (441, 21, 1)), as_strided(buf180, (256, 21, 64), (1344, 64, 1)), out=buf181)
    buf182 = as_strided(buf180, (32, 21, 512), (10752, 512, 1)); del buf180  # reuse
    kernel_cpp_39(c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()))
    del buf181
    buf183 = torch.ops.mkl._mkl_linear(buf182, arg121_1, arg120_1, None, 672)
    del arg120_1
    del arg121_1
    del buf182
    buf184 = as_strided(buf183, (32, 21, 512), (10752, 512, 1)); del buf183  # reuse
    buf185 = as_strided(buf168, (32, 21, 1), (21, 1, 672)); del buf168  # reuse
    buf186 = as_strided(buf167, (32, 21, 1), (21, 1, 672)); del buf167  # reuse
    buf187 = as_strided(buf166, (32, 21, 1), (21, 1, 672)); del buf166  # reuse
    buf188 = buf184; del buf184  # reuse
    kernel_cpp_40(c_void_p(buf188.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()))
    del arg122_1
    del arg123_1
    buf189 = torch.ops.mkl._mkl_linear(buf188, arg125_1, arg124_1, None, 672)
    del arg124_1
    del arg125_1
    buf190 = torch.ops.mkl._mkl_linear(buf165, arg127_1, arg126_1, None, 704)
    del arg126_1
    del arg127_1
    buf191 = torch.ops.mkl._mkl_linear(buf165, arg129_1, arg128_1, None, 704)
    del arg128_1
    del arg129_1
    buf192 = as_strided(buf169, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf169  # reuse
    buf193 = as_strided(buf157, (32, 8, 64, 22), (11264, 1408, 22, 1)); del buf157  # reuse
    kernel_cpp_41(c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    del buf190
    buf194 = empty_strided((256, 21, 22), (462, 22, 1), device='cpu', dtype=torch.float32)
    aten.bmm.out(as_strided(buf192, (256, 21, 64), (1344, 64, 1)), as_strided(buf193, (256, 64, 22), (1408, 22, 1)), out=buf194)
    buf195 = buf178; del buf178  # reuse
    buf196 = as_strided(buf194, (32, 8, 21, 22), (3696, 462, 22, 1)); del buf194  # reuse
    buf197 = buf176; del buf176  # reuse
    buf198 = buf196; del buf196  # reuse
    buf199 = as_strided(buf193, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf193  # reuse
    kernel_cpp_42(c_void_p(buf198.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf199.data_ptr()))
    del buf191
    buf200 = as_strided(buf192, (256, 21, 64), (1344, 64, 1)); del buf192  # reuse
    aten.bmm.out(as_strided(buf198, (256, 21, 22), (462, 22, 1)), as_strided(buf199, (256, 22, 64), (1408, 64, 1)), out=buf200)
    buf201 = as_strided(buf189, (32, 21, 512), (10752, 512, 1)); del buf189  # reuse
    kernel_cpp_43(c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    del buf200
    buf202 = torch.ops.mkl._mkl_linear(buf201, arg131_1, arg130_1, None, 672)
    del arg130_1
    del arg131_1
    del buf201
    buf203 = as_strided(buf202, (32, 21, 512), (10752, 512, 1)); del buf202  # reuse
    buf204 = buf187; del buf187  # reuse
    buf205 = buf186; del buf186  # reuse
    buf206 = buf185; del buf185  # reuse
    buf207 = buf203; del buf203  # reuse
    kernel_cpp_44(c_void_p(buf207.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()))
    del arg132_1
    del arg133_1
    buf208 = torch.ops.mkl._mkl_linear(buf207, arg136_1, arg134_1, arg135_1, 672)
    del arg134_1
    del arg135_1
    del arg136_1
    buf209 = as_strided(buf208, (32, 21, 2048), (43008, 2048, 1)); del buf208  # reuse
    kernel_cpp_45(c_void_p(buf209.data_ptr()))
    buf210 = torch.ops.mkl._mkl_linear(buf209, arg139_1, arg137_1, arg138_1, 672)
    del arg137_1
    del arg138_1
    del arg139_1
    del buf209
    buf211 = buf207; del buf207  # reuse
    buf212 = buf206; del buf206  # reuse
    buf213 = buf205; del buf205  # reuse
    buf214 = buf204; del buf204  # reuse
    buf215 = buf211; del buf211  # reuse
    kernel_cpp_46(c_void_p(buf215.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()))
    del arg140_1
    del arg141_1
    buf216 = torch.ops.mkl._mkl_linear(buf215, arg143_1, arg142_1, None, 672)
    del arg142_1
    del arg143_1
    buf217 = torch.ops.mkl._mkl_linear(buf215, arg145_1, arg144_1, None, 672)
    del arg144_1
    del arg145_1
    buf218 = torch.ops.mkl._mkl_linear(buf215, arg147_1, arg146_1, None, 672)
    del arg146_1
    del arg147_1
    buf219 = as_strided(buf210, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf210  # reuse
    buf220 = as_strided(buf188, (32, 8, 64, 21), (10752, 1344, 21, 1)); del buf188  # reuse
    kernel_cpp_47(c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()))
    del buf216
    del buf217
    buf221 = as_strided(buf179, (256, 21, 21), (441, 21, 1)); del buf179  # reuse
    aten.bmm.out(as_strided(buf219, (256, 21, 64), (1344, 64, 1)), as_strided(buf220, (256, 64, 21), (1344, 21, 1)), out=buf221)
    del buf219
    buf222 = buf197; del buf197  # reuse
    buf223 = as_strided(buf221, (32, 8, 21, 21), (3528, 441, 21, 1)); del buf221  # reuse
    buf224 = buf195; del buf195  # reuse
    buf225 = buf223; del buf223  # reuse
    buf226 = as_strided(buf220, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf220  # reuse
    kernel_cpp_48(c_void_p(buf225.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf226.data_ptr()))
    buf227 = as_strided(buf218, (256, 21, 64), (1344, 64, 1)); del buf218  # reuse
    aten.bmm.out(as_strided(buf225, (256, 21, 21), (441, 21, 1)), as_strided(buf226, (256, 21, 64), (1344, 64, 1)), out=buf227)
    buf228 = as_strided(buf226, (32, 21, 512), (10752, 512, 1)); del buf226  # reuse
    kernel_cpp_49(c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    del buf227
    buf229 = torch.ops.mkl._mkl_linear(buf228, arg149_1, arg148_1, None, 672)
    del arg148_1
    del arg149_1
    del buf228
    buf230 = buf215; del buf215  # reuse
    buf231 = buf214; del buf214  # reuse
    buf232 = buf213; del buf213  # reuse
    buf233 = buf212; del buf212  # reuse
    buf234 = buf230; del buf230  # reuse
    kernel_cpp_50(c_void_p(buf234.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()))
    del arg150_1
    del arg151_1
    buf235 = torch.ops.mkl._mkl_linear(buf234, arg153_1, arg152_1, None, 672)
    del arg152_1
    del arg153_1
    buf236 = torch.ops.mkl._mkl_linear(buf165, arg155_1, arg154_1, None, 704)
    del arg154_1
    del arg155_1
    buf237 = torch.ops.mkl._mkl_linear(buf165, arg157_1, arg156_1, None, 704)
    del arg156_1
    del arg157_1
    buf238 = as_strided(buf229, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf229  # reuse
    buf239 = as_strided(buf199, (32, 8, 64, 22), (11264, 1408, 22, 1)); del buf199  # reuse
    kernel_cpp_51(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()))
    del buf236
    buf240 = as_strided(buf198, (256, 21, 22), (462, 22, 1)); del buf198  # reuse
    aten.bmm.out(as_strided(buf238, (256, 21, 64), (1344, 64, 1)), as_strided(buf239, (256, 64, 22), (1408, 22, 1)), out=buf240)
    buf241 = buf224; del buf224  # reuse
    buf242 = as_strided(buf240, (32, 8, 21, 22), (3696, 462, 22, 1)); del buf240  # reuse
    buf243 = buf222; del buf222  # reuse
    buf244 = buf242; del buf242  # reuse
    buf245 = as_strided(buf239, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf239  # reuse
    kernel_cpp_52(c_void_p(buf244.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf245.data_ptr()))
    del buf237
    buf246 = as_strided(buf238, (256, 21, 64), (1344, 64, 1)); del buf238  # reuse
    aten.bmm.out(as_strided(buf244, (256, 21, 22), (462, 22, 1)), as_strided(buf245, (256, 22, 64), (1408, 64, 1)), out=buf246)
    buf247 = as_strided(buf235, (32, 21, 512), (10752, 512, 1)); del buf235  # reuse
    kernel_cpp_53(c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()))
    del buf246
    buf248 = torch.ops.mkl._mkl_linear(buf247, arg159_1, arg158_1, None, 672)
    del arg158_1
    del arg159_1
    del buf247
    buf249 = as_strided(buf248, (32, 21, 512), (10752, 512, 1)); del buf248  # reuse
    buf250 = buf233; del buf233  # reuse
    buf251 = buf232; del buf232  # reuse
    buf252 = buf231; del buf231  # reuse
    buf253 = buf249; del buf249  # reuse
    kernel_cpp_54(c_void_p(buf253.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()))
    del arg160_1
    del arg161_1
    buf254 = torch.ops.mkl._mkl_linear(buf253, arg164_1, arg162_1, arg163_1, 672)
    del arg162_1
    del arg163_1
    del arg164_1
    buf255 = as_strided(buf254, (32, 21, 2048), (43008, 2048, 1)); del buf254  # reuse
    kernel_cpp_55(c_void_p(buf255.data_ptr()))
    buf256 = torch.ops.mkl._mkl_linear(buf255, arg167_1, arg165_1, arg166_1, 672)
    del arg165_1
    del arg166_1
    del arg167_1
    del buf255
    buf257 = buf253; del buf253  # reuse
    buf258 = buf252; del buf252  # reuse
    buf259 = buf251; del buf251  # reuse
    buf260 = buf250; del buf250  # reuse
    buf261 = buf257; del buf257  # reuse
    kernel_cpp_56(c_void_p(buf261.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    del arg168_1
    del arg169_1
    buf262 = torch.ops.mkl._mkl_linear(buf261, arg171_1, arg170_1, None, 672)
    del arg170_1
    del arg171_1
    buf263 = torch.ops.mkl._mkl_linear(buf261, arg173_1, arg172_1, None, 672)
    del arg172_1
    del arg173_1
    buf264 = torch.ops.mkl._mkl_linear(buf261, arg175_1, arg174_1, None, 672)
    del arg174_1
    del arg175_1
    buf265 = as_strided(buf256, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf256  # reuse
    buf266 = as_strided(buf234, (32, 8, 64, 21), (10752, 1344, 21, 1)); del buf234  # reuse
    kernel_cpp_57(c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    del buf262
    del buf263
    buf267 = as_strided(buf225, (256, 21, 21), (441, 21, 1)); del buf225  # reuse
    aten.bmm.out(as_strided(buf265, (256, 21, 64), (1344, 64, 1)), as_strided(buf266, (256, 64, 21), (1344, 21, 1)), out=buf267)
    del buf265
    buf268 = buf243; del buf243  # reuse
    buf269 = as_strided(buf267, (32, 8, 21, 21), (3528, 441, 21, 1)); del buf267  # reuse
    buf270 = buf241; del buf241  # reuse
    buf271 = buf269; del buf269  # reuse
    buf272 = as_strided(buf266, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf266  # reuse
    kernel_cpp_58(c_void_p(buf271.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf272.data_ptr()))
    buf273 = as_strided(buf264, (256, 21, 64), (1344, 64, 1)); del buf264  # reuse
    aten.bmm.out(as_strided(buf271, (256, 21, 21), (441, 21, 1)), as_strided(buf272, (256, 21, 64), (1344, 64, 1)), out=buf273)
    buf274 = as_strided(buf272, (32, 21, 512), (10752, 512, 1)); del buf272  # reuse
    kernel_cpp_59(c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    del buf273
    buf275 = torch.ops.mkl._mkl_linear(buf274, arg177_1, arg176_1, None, 672)
    del arg176_1
    del arg177_1
    del buf274
    buf276 = buf261; del buf261  # reuse
    buf277 = buf260; del buf260  # reuse
    buf278 = buf259; del buf259  # reuse
    buf279 = buf258; del buf258  # reuse
    buf280 = buf276; del buf276  # reuse
    kernel_cpp_60(c_void_p(buf280.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()))
    del arg178_1
    del arg179_1
    buf281 = torch.ops.mkl._mkl_linear(buf280, arg181_1, arg180_1, None, 672)
    del arg180_1
    del arg181_1
    buf282 = torch.ops.mkl._mkl_linear(buf165, arg183_1, arg182_1, None, 704)
    del arg182_1
    del arg183_1
    buf283 = torch.ops.mkl._mkl_linear(buf165, arg185_1, arg184_1, None, 704)
    del arg184_1
    del arg185_1
    buf284 = as_strided(buf275, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf275  # reuse
    buf285 = as_strided(buf245, (32, 8, 64, 22), (11264, 1408, 22, 1)); del buf245  # reuse
    kernel_cpp_61(c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    del buf282
    buf286 = as_strided(buf244, (256, 21, 22), (462, 22, 1)); del buf244  # reuse
    aten.bmm.out(as_strided(buf284, (256, 21, 64), (1344, 64, 1)), as_strided(buf285, (256, 64, 22), (1408, 22, 1)), out=buf286)
    buf287 = buf270; del buf270  # reuse
    buf288 = as_strided(buf286, (32, 8, 21, 22), (3696, 462, 22, 1)); del buf286  # reuse
    buf289 = buf268; del buf268  # reuse
    buf290 = buf288; del buf288  # reuse
    buf291 = as_strided(buf285, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf285  # reuse
    kernel_cpp_62(c_void_p(buf290.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()))
    del buf283
    buf292 = as_strided(buf284, (256, 21, 64), (1344, 64, 1)); del buf284  # reuse
    aten.bmm.out(as_strided(buf290, (256, 21, 22), (462, 22, 1)), as_strided(buf291, (256, 22, 64), (1408, 64, 1)), out=buf292)
    buf293 = as_strided(buf281, (32, 21, 512), (10752, 512, 1)); del buf281  # reuse
    kernel_cpp_63(c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()))
    del buf292
    buf294 = torch.ops.mkl._mkl_linear(buf293, arg187_1, arg186_1, None, 672)
    del arg186_1
    del arg187_1
    del buf293
    buf295 = buf280; del buf280  # reuse
    buf296 = buf279; del buf279  # reuse
    buf297 = buf278; del buf278  # reuse
    buf298 = buf277; del buf277  # reuse
    buf299 = buf295; del buf295  # reuse
    kernel_cpp_64(c_void_p(buf299.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()))
    del arg188_1
    del arg189_1
    buf300 = torch.ops.mkl._mkl_linear(buf299, arg192_1, arg190_1, arg191_1, 672)
    del arg190_1
    del arg191_1
    del arg192_1
    buf301 = as_strided(buf300, (32, 21, 2048), (43008, 2048, 1)); del buf300  # reuse
    kernel_cpp_65(c_void_p(buf301.data_ptr()))
    buf302 = torch.ops.mkl._mkl_linear(buf301, arg195_1, arg193_1, arg194_1, 672)
    del arg193_1
    del arg194_1
    del arg195_1
    del buf301
    buf303 = buf299; del buf299  # reuse
    buf304 = buf298; del buf298  # reuse
    buf305 = buf297; del buf297  # reuse
    buf306 = buf296; del buf296  # reuse
    buf307 = buf303; del buf303  # reuse
    kernel_cpp_66(c_void_p(buf307.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    del arg196_1
    del arg197_1
    buf308 = torch.ops.mkl._mkl_linear(buf307, arg199_1, arg198_1, None, 672)
    del arg198_1
    del arg199_1
    buf309 = torch.ops.mkl._mkl_linear(buf307, arg201_1, arg200_1, None, 672)
    del arg200_1
    del arg201_1
    buf310 = torch.ops.mkl._mkl_linear(buf307, arg203_1, arg202_1, None, 672)
    del arg202_1
    del arg203_1
    buf311 = as_strided(buf302, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf302  # reuse
    buf312 = as_strided(buf294, (32, 8, 64, 21), (10752, 1344, 21, 1)); del buf294  # reuse
    kernel_cpp_67(c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    del buf308
    del buf309
    buf313 = as_strided(buf271, (256, 21, 21), (441, 21, 1)); del buf271  # reuse
    aten.bmm.out(as_strided(buf311, (256, 21, 64), (1344, 64, 1)), as_strided(buf312, (256, 64, 21), (1344, 21, 1)), out=buf313)
    del buf311
    buf314 = buf289; del buf289  # reuse
    buf315 = as_strided(buf313, (32, 8, 21, 21), (3528, 441, 21, 1)); del buf313  # reuse
    buf316 = buf287; del buf287  # reuse
    buf317 = buf315; del buf315  # reuse
    buf318 = as_strided(buf312, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf312  # reuse
    kernel_cpp_68(c_void_p(buf317.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf318.data_ptr()))
    buf319 = as_strided(buf310, (256, 21, 64), (1344, 64, 1)); del buf310  # reuse
    aten.bmm.out(as_strided(buf317, (256, 21, 21), (441, 21, 1)), as_strided(buf318, (256, 21, 64), (1344, 64, 1)), out=buf319)
    buf320 = as_strided(buf318, (32, 21, 512), (10752, 512, 1)); del buf318  # reuse
    kernel_cpp_69(c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()))
    del buf319
    buf321 = torch.ops.mkl._mkl_linear(buf320, arg205_1, arg204_1, None, 672)
    del arg204_1
    del arg205_1
    del buf320
    buf322 = buf307; del buf307  # reuse
    buf323 = buf306; del buf306  # reuse
    buf324 = buf305; del buf305  # reuse
    buf325 = buf304; del buf304  # reuse
    buf326 = buf322; del buf322  # reuse
    kernel_cpp_70(c_void_p(buf326.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()))
    del arg206_1
    del arg207_1
    buf327 = torch.ops.mkl._mkl_linear(buf326, arg209_1, arg208_1, None, 672)
    del arg208_1
    del arg209_1
    buf328 = torch.ops.mkl._mkl_linear(buf165, arg211_1, arg210_1, None, 704)
    del arg210_1
    del arg211_1
    buf329 = torch.ops.mkl._mkl_linear(buf165, arg213_1, arg212_1, None, 704)
    del arg212_1
    del arg213_1
    buf330 = as_strided(buf321, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf321  # reuse
    buf331 = as_strided(buf291, (32, 8, 64, 22), (11264, 1408, 22, 1)); del buf291  # reuse
    kernel_cpp_71(c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()))
    del buf328
    buf332 = as_strided(buf290, (256, 21, 22), (462, 22, 1)); del buf290  # reuse
    aten.bmm.out(as_strided(buf330, (256, 21, 64), (1344, 64, 1)), as_strided(buf331, (256, 64, 22), (1408, 22, 1)), out=buf332)
    buf333 = buf316; del buf316  # reuse
    buf334 = as_strided(buf332, (32, 8, 21, 22), (3696, 462, 22, 1)); del buf332  # reuse
    buf335 = buf314; del buf314  # reuse
    buf336 = buf334; del buf334  # reuse
    buf337 = as_strided(buf331, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf331  # reuse
    kernel_cpp_72(c_void_p(buf336.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf337.data_ptr()))
    del buf329
    buf338 = as_strided(buf330, (256, 21, 64), (1344, 64, 1)); del buf330  # reuse
    aten.bmm.out(as_strided(buf336, (256, 21, 22), (462, 22, 1)), as_strided(buf337, (256, 22, 64), (1408, 64, 1)), out=buf338)
    buf339 = as_strided(buf327, (32, 21, 512), (10752, 512, 1)); del buf327  # reuse
    kernel_cpp_73(c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()))
    del buf338
    buf340 = torch.ops.mkl._mkl_linear(buf339, arg215_1, arg214_1, None, 672)
    del arg214_1
    del arg215_1
    del buf339
    buf341 = as_strided(buf340, (32, 21, 512), (10752, 512, 1)); del buf340  # reuse
    buf342 = buf325; del buf325  # reuse
    buf343 = buf324; del buf324  # reuse
    buf344 = buf323; del buf323  # reuse
    buf345 = buf341; del buf341  # reuse
    kernel_cpp_74(c_void_p(buf345.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()))
    del arg216_1
    del arg217_1
    buf346 = torch.ops.mkl._mkl_linear(buf345, arg220_1, arg218_1, arg219_1, 672)
    del arg218_1
    del arg219_1
    del arg220_1
    buf347 = as_strided(buf346, (32, 21, 2048), (43008, 2048, 1)); del buf346  # reuse
    kernel_cpp_75(c_void_p(buf347.data_ptr()))
    buf348 = torch.ops.mkl._mkl_linear(buf347, arg223_1, arg221_1, arg222_1, 672)
    del arg221_1
    del arg222_1
    del arg223_1
    del buf347
    buf349 = as_strided(buf348, (32, 21, 512), (10752, 512, 1)); del buf348  # reuse
    buf350 = buf344; del buf344  # reuse
    buf351 = buf343; del buf343  # reuse
    buf352 = buf342; del buf342  # reuse
    buf353 = buf349; del buf349  # reuse
    kernel_cpp_76(c_void_p(buf353.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()))
    del arg224_1
    del arg225_1
    buf354 = torch.ops.mkl._mkl_linear(buf353, arg227_1, arg226_1, None, 672)
    del arg226_1
    del arg227_1
    buf355 = torch.ops.mkl._mkl_linear(buf353, arg229_1, arg228_1, None, 672)
    del arg228_1
    del arg229_1
    buf356 = torch.ops.mkl._mkl_linear(buf353, arg231_1, arg230_1, None, 672)
    del arg230_1
    del arg231_1
    buf357 = as_strided(buf345, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf345  # reuse
    buf358 = as_strided(buf326, (32, 8, 64, 21), (10752, 1344, 21, 1)); del buf326  # reuse
    kernel_cpp_77(c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()))
    del buf354
    del buf355
    buf359 = as_strided(buf317, (256, 21, 21), (441, 21, 1)); del buf317  # reuse
    aten.bmm.out(as_strided(buf357, (256, 21, 64), (1344, 64, 1)), as_strided(buf358, (256, 64, 21), (1344, 21, 1)), out=buf359)
    del buf357
    buf360 = buf335; del buf335  # reuse
    buf361 = as_strided(buf359, (32, 8, 21, 21), (3528, 441, 21, 1)); del buf359  # reuse
    buf362 = buf333; del buf333  # reuse
    buf363 = buf361; del buf361  # reuse
    buf364 = as_strided(buf358, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf358  # reuse
    kernel_cpp_78(c_void_p(buf363.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf364.data_ptr()))
    buf365 = as_strided(buf356, (256, 21, 64), (1344, 64, 1)); del buf356  # reuse
    aten.bmm.out(as_strided(buf363, (256, 21, 21), (441, 21, 1)), as_strided(buf364, (256, 21, 64), (1344, 64, 1)), out=buf365)
    buf366 = as_strided(buf364, (32, 21, 512), (10752, 512, 1)); del buf364  # reuse
    kernel_cpp_79(c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()))
    del buf365
    buf367 = torch.ops.mkl._mkl_linear(buf366, arg233_1, arg232_1, None, 672)
    del arg232_1
    del arg233_1
    del buf366
    buf368 = buf353; del buf353  # reuse
    buf369 = buf352; del buf352  # reuse
    buf370 = buf351; del buf351  # reuse
    buf371 = buf350; del buf350  # reuse
    buf372 = buf368; del buf368  # reuse
    kernel_cpp_80(c_void_p(buf372.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()))
    del arg234_1
    del arg235_1
    buf373 = torch.ops.mkl._mkl_linear(buf372, arg237_1, arg236_1, None, 672)
    del arg236_1
    del arg237_1
    buf374 = torch.ops.mkl._mkl_linear(buf165, arg239_1, arg238_1, None, 704)
    del arg238_1
    del arg239_1
    buf375 = torch.ops.mkl._mkl_linear(buf165, arg241_1, arg240_1, None, 704)
    del arg240_1
    del arg241_1
    buf376 = as_strided(buf367, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf367  # reuse
    buf377 = as_strided(buf337, (32, 8, 64, 22), (11264, 1408, 22, 1)); del buf337  # reuse
    kernel_cpp_81(c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    del buf374
    buf378 = as_strided(buf336, (256, 21, 22), (462, 22, 1)); del buf336  # reuse
    aten.bmm.out(as_strided(buf376, (256, 21, 64), (1344, 64, 1)), as_strided(buf377, (256, 64, 22), (1408, 22, 1)), out=buf378)
    buf379 = buf362; del buf362  # reuse
    buf380 = as_strided(buf378, (32, 8, 21, 22), (3696, 462, 22, 1)); del buf378  # reuse
    buf381 = buf360; del buf360  # reuse
    buf382 = buf380; del buf380  # reuse
    buf383 = as_strided(buf377, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf377  # reuse
    kernel_cpp_82(c_void_p(buf382.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf383.data_ptr()))
    del buf375
    buf384 = as_strided(buf376, (256, 21, 64), (1344, 64, 1)); del buf376  # reuse
    aten.bmm.out(as_strided(buf382, (256, 21, 22), (462, 22, 1)), as_strided(buf383, (256, 22, 64), (1408, 64, 1)), out=buf384)
    del buf383
    buf385 = as_strided(buf373, (32, 21, 512), (10752, 512, 1)); del buf373  # reuse
    kernel_cpp_83(c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()))
    del buf384
    buf386 = torch.ops.mkl._mkl_linear(buf385, arg243_1, arg242_1, None, 672)
    del arg242_1
    del arg243_1
    del buf385
    buf387 = as_strided(buf386, (32, 21, 512), (10752, 512, 1)); del buf386  # reuse
    buf388 = buf371; del buf371  # reuse
    buf389 = buf370; del buf370  # reuse
    buf390 = buf369; del buf369  # reuse
    buf391 = buf387; del buf387  # reuse
    kernel_cpp_84(c_void_p(buf391.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()))
    del arg244_1
    del arg245_1
    buf392 = torch.ops.mkl._mkl_linear(buf391, arg248_1, arg246_1, arg247_1, 672)
    del arg246_1
    del arg247_1
    del arg248_1
    buf393 = as_strided(buf392, (32, 21, 2048), (43008, 2048, 1)); del buf392  # reuse
    kernel_cpp_85(c_void_p(buf393.data_ptr()))
    buf394 = torch.ops.mkl._mkl_linear(buf393, arg251_1, arg249_1, arg250_1, 672)
    del arg249_1
    del arg250_1
    del arg251_1
    del buf393
    buf395 = buf391; del buf391  # reuse
    buf396 = buf390; del buf390  # reuse
    buf397 = buf389; del buf389  # reuse
    buf398 = buf388; del buf388  # reuse
    buf399 = buf395; del buf395  # reuse
    kernel_cpp_86(c_void_p(buf399.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()))
    del arg252_1
    del arg253_1
    buf400 = torch.ops.mkl._mkl_linear(buf399, arg255_1, arg254_1, None, 672)
    del arg254_1
    del arg255_1
    buf401 = torch.ops.mkl._mkl_linear(buf399, arg257_1, arg256_1, None, 672)
    del arg256_1
    del arg257_1
    buf402 = torch.ops.mkl._mkl_linear(buf399, arg259_1, arg258_1, None, 672)
    del arg258_1
    del arg259_1
    buf403 = as_strided(buf394, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf394  # reuse
    buf404 = as_strided(buf372, (32, 8, 64, 21), (10752, 1344, 21, 1)); del buf372  # reuse
    kernel_cpp_87(c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()))
    del buf400
    del buf401
    buf405 = as_strided(buf363, (256, 21, 21), (441, 21, 1)); del buf363  # reuse
    aten.bmm.out(as_strided(buf403, (256, 21, 64), (1344, 64, 1)), as_strided(buf404, (256, 64, 21), (1344, 21, 1)), out=buf405)
    del buf403
    buf406 = buf381; del buf381  # reuse
    buf407 = as_strided(buf405, (32, 8, 21, 21), (3528, 441, 21, 1)); del buf405  # reuse
    buf408 = buf379; del buf379  # reuse
    buf409 = buf407; del buf407  # reuse
    buf410 = as_strided(buf404, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf404  # reuse
    kernel_cpp_88(c_void_p(buf409.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf410.data_ptr()))
    del arg287_1
    buf411 = as_strided(buf402, (256, 21, 64), (1344, 64, 1)); del buf402  # reuse
    aten.bmm.out(as_strided(buf409, (256, 21, 21), (441, 21, 1)), as_strided(buf410, (256, 21, 64), (1344, 64, 1)), out=buf411)
    del buf409
    buf412 = as_strided(buf410, (32, 21, 512), (10752, 512, 1)); del buf410  # reuse
    kernel_cpp_89(c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()))
    del buf411
    buf413 = torch.ops.mkl._mkl_linear(buf412, arg261_1, arg260_1, None, 672)
    del arg260_1
    del arg261_1
    del buf412
    buf414 = as_strided(buf413, (32, 21, 512), (10752, 512, 1)); del buf413  # reuse
    buf415 = buf398; del buf398  # reuse
    buf416 = buf397; del buf397  # reuse
    buf417 = buf396; del buf396  # reuse
    buf418 = buf414; del buf414  # reuse
    kernel_cpp_90(c_void_p(buf418.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()))
    del arg262_1
    del arg263_1
    buf419 = torch.ops.mkl._mkl_linear(buf418, arg265_1, arg264_1, None, 672)
    del arg264_1
    del arg265_1
    buf420 = torch.ops.mkl._mkl_linear(buf165, arg267_1, arg266_1, None, 704)
    del arg266_1
    del arg267_1
    buf421 = torch.ops.mkl._mkl_linear(buf165, arg269_1, arg268_1, None, 704)
    del arg268_1
    del arg269_1
    buf422 = as_strided(buf399, (32, 8, 21, 64), (10752, 1344, 64, 1)); del buf399  # reuse
    buf423 = as_strided(buf165, (32, 8, 64, 22), (11264, 1408, 22, 1)); del buf165  # reuse
    kernel_cpp_91(c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()))
    del buf420
    buf424 = as_strided(buf382, (256, 21, 22), (462, 22, 1)); del buf382  # reuse
    aten.bmm.out(as_strided(buf422, (256, 21, 64), (1344, 64, 1)), as_strided(buf423, (256, 64, 22), (1408, 22, 1)), out=buf424)
    buf425 = buf408; del buf408  # reuse
    buf426 = as_strided(buf424, (32, 8, 21, 22), (3696, 462, 22, 1)); del buf424  # reuse
    buf427 = buf406; del buf406  # reuse
    buf428 = buf426; del buf426  # reuse
    buf429 = as_strided(buf423, (32, 8, 22, 64), (11264, 1408, 64, 1)); del buf423  # reuse
    kernel_cpp_92(c_void_p(buf428.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf429.data_ptr()))
    del arg286_1
    del buf421
    del buf425
    del buf427
    buf430 = as_strided(buf422, (256, 21, 64), (1344, 64, 1)); del buf422  # reuse
    aten.bmm.out(as_strided(buf428, (256, 21, 22), (462, 22, 1)), as_strided(buf429, (256, 22, 64), (1408, 64, 1)), out=buf430)
    del buf428
    del buf429
    buf431 = as_strided(buf419, (32, 21, 512), (10752, 512, 1)); del buf419  # reuse
    kernel_cpp_93(c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()))
    del buf430
    buf432 = torch.ops.mkl._mkl_linear(buf431, arg271_1, arg270_1, None, 672)
    del arg270_1
    del arg271_1
    del buf431
    buf433 = buf418; del buf418  # reuse
    buf434 = buf417; del buf417  # reuse
    buf435 = buf416; del buf416  # reuse
    buf436 = buf415; del buf415  # reuse
    buf437 = buf433; del buf433  # reuse
    kernel_cpp_94(c_void_p(buf437.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()))
    del arg272_1
    del arg273_1
    del buf432
    buf438 = torch.ops.mkl._mkl_linear(buf437, arg276_1, arg274_1, arg275_1, 672)
    del arg274_1
    del arg275_1
    del arg276_1
    buf439 = as_strided(buf438, (32, 21, 2048), (43008, 2048, 1)); del buf438  # reuse
    kernel_cpp_95(c_void_p(buf439.data_ptr()))
    buf440 = torch.ops.mkl._mkl_linear(buf439, arg279_1, arg277_1, arg278_1, 672)
    del arg277_1
    del arg278_1
    del arg279_1
    del buf439
    buf441 = buf437; del buf437  # reuse
    buf442 = buf436; del buf436  # reuse
    buf443 = buf435; del buf435  # reuse
    buf444 = buf434; del buf434  # reuse
    buf445 = buf441; del buf441  # reuse
    kernel_cpp_96(c_void_p(buf445.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()))
    del arg280_1
    del arg281_1
    del buf440
    del buf442
    del buf443
    del buf444
    buf446 = torch.ops.mkl._mkl_linear(buf445, arg283_1, arg282_1, None, 672)
    del arg282_1
    del arg283_1
    del buf445
    buf447 = as_strided(buf446, (32, 21, 9521), (199941, 9521, 1)); del buf446  # reuse
    kernel_cpp_97(c_void_p(buf447.data_ptr()))
    return (as_strided(buf447, (672, 9521), (9521, 1)), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((9521, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((3293409, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((4079841, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((3293409, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((4079841, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((3293409, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((4079841, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((3293409, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((4079841, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((3293409, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((4079841, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((3293409, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((4079841, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((9521, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((3293409, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((4079841, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((3293409, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((4079841, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((3293409, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((4079841, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((3293409, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((4079841, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((3293409, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((4079841, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((3293409, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((4079841, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((9521, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((7880929, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((1, 200, 512), (102400, 512, 1), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((1, 200, 512), (102400, 512, 1), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((32, 22), (1, 32), device='cpu', dtype=torch.int64)
    arg287_1 = rand_strided((32, 21), (1, 32), device='cpu', dtype=torch.int64)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1]))
