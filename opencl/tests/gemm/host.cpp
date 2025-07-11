/*
* Copyright (c) 2020-2023, Intel Corporation
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
* OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*/

#include <iostream>
#include <cassert>
#include <math.h>
#include <vector>

#include <CL/cl.h>
#include "float16.hpp"
#include "tensor.hpp"
#include "common.hpp"

#if defined(BINNAME)
# define KERNEL BINNAME
#endif

#define SZ 160
#define KERNEL_SZ 16
#define CHECK(a) do { \
    err = (a); \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "FAIL: err=%d @ line=%d (%s)\n", err, __LINE__, (#a)); \
        exit(err); \
    } \
}while (0)
#define CHECK2(a) do { \
    (a); \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "FAIL: err=%d @ line=%d (%s)\n", err, __LINE__, (#a)); \
        exit(err); \
    } \
}while (0)
#ifndef KERNEL
#error "Error: KERNEL must be defined with location of kernel binary"
#endif

using namespace ov;

void matmul(const PlainTensor& a, const PlainTensor& b, PlainTensor& c) {
    PlainTensor tmp;
    tmp.resize<float>({ c.size(0), c.size(1) });
    tmp = 0.0f;
    for (int i = 0; i < c.size(0); ++i) {
        for (int j = 0; j < c.size(1); ++j) {
            for (int k = 0; k < a.size(1); ++k) {
                tmp.ptr<float>(i)[j] += a.ptr<float16_t>(i)[k] * b.ptr<float16_t>(k)[j];
            }
        }
    }
    for (int i = 0; i < c.size(0); ++i) {
        for (int j = 0; j < c.size(1); ++j) {
            c.ptr<float16_t>(i)[j] = tmp.ptr<float>(i)[j];
        }
    }
}

// [K, N] -> [N, K]
void transpose(int K, int N, float16_t* src, float16_t* dst) {
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            dst[n * K + k] = src[k * N + n];
        }
    }
}

void repack_f16(int K, int N, float16_t* src, float16_t* dst) {
    const int DEPTH = 8;
    const int BLOCK_REG_M = 8;
    const int BLOCK_REG_N = x::SG_SIZE;
    // half is 2, u8 is 4
    const int VNNI = 2;
    const int BLOCK_REG_K = 32 / VNNI;
    const int BLOCK_WG_M = x::BLOCK_SG_M * x::SG_M;
    const int BLOCK_WG_N = x::BLOCK_SG_N * x::SG_N;
    // register blocking
    const int REG_M = x::BLOCK_SG_M / BLOCK_REG_M;
    const int REG_N = x::BLOCK_SG_N / BLOCK_REG_N;
    // packed memory layout:
    struct reg {                                        // --> one DPAS op
        float16_t data[BLOCK_REG_K/2][BLOCK_REG_N*2];   // BLOCK_REG_K=16, BLOCK_REG_N=8, aka half[8,16]
    };
    struct sg_block {                                   // --> one EU
        reg regs[x::BLOCK_WG_K/BLOCK_REG_K][REG_N];     // REG_N==2, BLOCK_WG_K/BLOCK_REG_K==4
    };
    struct wg_block {                                   // --> one WG
        sg_block block[x::SG_N];
    };
    //struct weight {                                   // --> weight
    //    wg_block wg[WG_N][K/BLOCK_WG_K];
    //};

    const int WG_N = N / BLOCK_WG_N;
    const int WG_K = K / x::BLOCK_WG_K;
    wg_block* wg_block_ptr = (wg_block*)dst;
    for (int wg_n = 0; wg_n < WG_N; wg_n++) {
        for (int wg_k = 0; wg_k < WG_K; wg_k++) {
            wg_block& wg = *wg_block_ptr++;
            auto src_wg_block = src + wg_k * x::BLOCK_WG_K * N + wg_n * BLOCK_WG_N;
            for (int sg_n = 0; sg_n < x::SG_N; sg_n++) {
                sg_block& sg = wg.block[sg_n];
                auto src_sg_block = src_wg_block + sg_n * x::BLOCK_SG_N;
                for (int reg_k = 0; reg_k < x::BLOCK_WG_K / BLOCK_REG_K; reg_k++) {
                    for (int reg_n = 0; reg_n < REG_N; reg_n++) {
                        reg& cur_reg = sg.regs[reg_k][reg_n];
                        auto src_reg = src_sg_block + reg_k * BLOCK_REG_K * N + reg_n * BLOCK_REG_N;
                        for (int k = 0; k < BLOCK_REG_K / 2; k++) {
                            for (int n = 0; n < BLOCK_REG_N; n++) {
                                cur_reg.data[k][2 * n + 0] = *(src_reg + (k * 2 + 0) * N + n);
                                cur_reg.data[k][2 * n + 1] = *(src_reg + (k * 2 + 1) * N + n);
                            }
                        }
                    }
                }
            }
        }
    }
}

void fill(PlainTensor& t, float start = -2.0f, int num = 5) {
    uint32_t M = t.size(0), N = t.size(1);
    float16_t* p = (float16_t*)t.m_ptr.get();
    for (uint32_t i = 0; i < M * N; i += num) {
        for (uint32_t j = 0; j < num; j++) {
            if (i + j < M * N) p[i + j] = start + j;
        }
//#define SET(idx, val) if (i + idx < M * N) p[i + idx] = val;
//        SET(0, -2.0f);
//        SET(1, -1.0f);
//        SET(2, 0.0f);
//        SET(3, 1.0f);
//        SET(4, 2.0f);
//#undef SET
    }
}

PlainTensor get_ref(PlainTensor& a, PlainTensor& b) {
    PlainTensor c;
    uint32_t M = a.size(0);
    uint32_t N = b.size(1);
    c.resize<float16_t>({ M, N });
    matmul(a, b, c);
    return c;
}

// initialize GPU
cl_platform_id platform;  // OpenCL platform
cl_device_id device;      // device ID
cl_context context;       // context
cl_command_queue queue;   // command queue
cl_program program;       // program

void init_ocl() {
    cl_int err;
    CHECK(clGetPlatformIDs(1, &platform, NULL));
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
    CHECK2(context = clCreateContext(NULL, 1, &device, NULL, NULL, &err));
    CHECK2(queue = clCreateCommandQueueWithProperties(context, device, 0, &err));

    // diagnostic info
    char name_buffer[256];
    CHECK(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name_buffer), name_buffer, NULL));
    fprintf(stderr, "INFO: using platform: %s\n", name_buffer);
    CHECK(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name_buffer), name_buffer, NULL));
    fprintf(stderr, "INFO: using device: %s\n", name_buffer);

    // read in and initialize kernel
    FILE* fp = fopen(KERNEL, "rb");
    if (fp == NULL) {
        fprintf(stderr, "FAIL: unable to open %s\n", KERNEL);
        exit(-1);
    }
    fseek(fp, 0, SEEK_END);
    size_t sz = ftell(fp);
    rewind(fp);

    unsigned char* code = (unsigned char*)malloc(sz);
    fread(code, 1, sz, fp);
    fclose(fp);

    cl_int errNum = 0;
    const unsigned char* codes[1] = { code };
    size_t sizes[1] = { sz };
    CHECK2(program = clCreateProgramWithBinary(context, 1, &device, sizes, codes, &err, &errNum));
    CHECK(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
}

void uninit_ocl() {
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void exec_kernel(std::string name, std::vector<PlainTensor> in_bufs, std::vector<PlainTensor> out_bufs, std::vector<uint32_t> scalars,
    std::vector<size_t> globalSize, std::vector<size_t> localSize) {
    cl_int err;
    cl_kernel kernel;    // kernel

    CHECK2(kernel = clCreateKernel(program, name.c_str(), &err));

    std::vector<cl_mem> in_mems, out_mems;
    in_mems.resize(in_bufs.size());
    out_mems.resize(out_bufs.size());
    for (size_t i = 0; i < in_bufs.size(); ++i) {
        auto size = in_bufs[i].size(0) * in_bufs[i].size(1) * in_bufs[i].m_element_size;
        in_mems[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, NULL);
        CHECK(clEnqueueWriteBuffer(queue, in_mems[i], CL_TRUE, 0, size, in_bufs[i].m_ptr.get(), 0, NULL, NULL));
    }
    for (size_t i = 0; i < out_bufs.size(); ++i) {
        auto size = out_bufs[i].size(0) * out_bufs[i].size(1) * out_bufs[i].m_element_size;
        out_mems[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);
    }

    size_t arg_idx = 0;
    for (size_t i = 0; i < in_mems.size(); i++) {
        CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &in_mems[i]));
    }
    for (size_t i = 0; i < out_mems.size(); i++) {
        CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &out_mems[i]));
    }
    for (size_t i = 0; i < scalars.size(); i++) {
        CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(uint32_t), &scalars[i]));
    }
    assert(globalSize.size() == localSize.size());
    CHECK(clEnqueueNDRangeKernel(queue, kernel, globalSize.size(), NULL, globalSize.data(), localSize.data(), 0, NULL, NULL));
    clFinish(queue);

    for (size_t i = 0; i < out_bufs.size(); ++i) {
        auto size = out_bufs[i].size(0) * out_bufs[i].size(1) * out_bufs[i].m_element_size;
        clEnqueueReadBuffer(queue, out_mems[i], CL_TRUE, 0, size, out_bufs[i].m_ptr.get(), 0, NULL, NULL);
        clReleaseMemObject(out_mems[i]);
    }
    for (size_t i = 0; i < in_bufs.size(); ++i) {
        clReleaseMemObject(in_mems[i]);
    }
    clReleaseKernel(kernel);
}

void cmp(std::string prefix, PlainTensor& cur_t, PlainTensor& ref_t) {
    for (int i = 0; i < cur_t.size(0); ++i) {
        for (int j = 0; j < cur_t.size(1); ++j) {
            float cur = cur_t.ptr<float16_t>(i)[j];
            float ref = ref_t.ptr<float16_t>(i)[j];
            if (std::abs(cur - ref) > 0.01f) {
                fprintf(stderr, "FAIL: comparison '%s' at index[%d, %d], cur: %f, ref: %f\n", prefix.c_str(), i, j, cur, ref);
                exit(-1);
            }
        }
    }
}

void check_v1(PlainTensor& a, PlainTensor& b, PlainTensor& c_ref) {
    PlainTensor c;
    PlainTensor b_repack, b_repack_ref;
    uint32_t M = a.size(0);
    uint32_t N = b.size(1);
    uint32_t K = a.size(1);
    b_repack_ref.resize<float16_t>({ K, N });
    b_repack.resize<float16_t>({ K, N });
    c.resize<float16_t>({ M, N });
    repack_f16(K, N, (float16_t*)b.m_ptr.get(), (float16_t*)b_repack_ref.m_ptr.get());
    cl_int err;
    size_t BLOCK_SG_M = x::BLOCK_SG_M;
    size_t BLOCK_SG_N = x::BLOCK_SG_N;
    size_t SG_M = x::SG_M, SG_N = x::SG_N;
    std::vector<size_t> globalSize = { M / BLOCK_SG_M, N / BLOCK_SG_N };
    std::vector<size_t> localSize = { SG_M, SG_N };
    exec_kernel("gemm", { a, b_repack_ref }, { c }, { M, N, K, K, N, N }, globalSize, localSize);
    cmp("gemm v1", c, c_ref);
    {
        size_t BLOCK_SG_M = x::BLOCK_SG_M;
        size_t BLOCK_SG_N = x::BLOCK_SG_N;
        size_t SG_M = x::SG_M, SG_N = x::SG_N;
        size_t BLOCK_WG_N = BLOCK_SG_N * SG_N;
        std::vector<size_t> globalSize = { N / BLOCK_WG_N, K / x::BLOCK_WG_K };
        std::vector<size_t> localSize  = { 1, 1 };
        exec_kernel("repack_f16", { b }, { b_repack }, { K, N }, globalSize, localSize);
        cmp("repack", b_repack, b_repack_ref);
    }
}

// a: [M, K], b: [N, K]
void check_f16_xe1(PlainTensor& a, PlainTensor& b, PlainTensor& c_ref) {
    PlainTensor c;
    PlainTensor b_t;
    uint32_t M = a.size(0);
    uint32_t N = b.size(1);
    uint32_t K = a.size(1);
    b_t.resize<float16_t>({ K, N });
    c.resize<float16_t>({ M, N });
    transpose(K, N, (float16_t*)b.m_ptr.get(), (float16_t*)b_t.m_ptr.get());
    cl_int err;
    size_t BLOCK_SG_M = f16_xe1::BLOCK_SG_M;
    size_t BLOCK_SG_N = f16_xe1::BLOCK_SG_N;
    size_t SG_M = f16_xe1::SG_M, SG_N = f16_xe1::SG_N;
    size_t BLOCK_WG_M = BLOCK_SG_M * SG_M;
    size_t BLOCK_WG_N = BLOCK_SG_N * SG_N;
    std::vector<size_t> globalSize = { M / BLOCK_WG_M * N / BLOCK_WG_N * SG_N, SG_M};
    std::vector<size_t> localSize = { SG_N, SG_M };

    int slice;
    int slice_no;

    // if slice_no == 0, slice is linear type, 0 for loop M first and 1 for loop N first
    slice_no = 0;
    for (slice = 0; slice < 2; slice++) {
       exec_kernel("gemm_nocopy_xe1", { a, b_t }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice}, globalSize, localSize);
       cmp("gemm_nocopy_xe1 linear " + (slice == 0 ? std::string("M first") : std::string("N first")), c, c_ref);
    }
    // if slice_no != 0, use wg blocking schema:
    //    slice > 0 means the slice is in M dimension else is in N dimension
    //    slice_no > 0 means the slice size of reminder will be slice+1 else be slice-1
    slice = 2;
    slice_no = -2;
    exec_kernel("gemm_nocopy_xe1", { a, b_t }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_nocopy_xe1 block m:2x2+1*3", c, c_ref);
    slice = 3;
    slice_no = 2;
    exec_kernel("gemm_nocopy_xe1", { a, b_t }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_nocopy_xe1 block m:3*2+4*2", c, c_ref);

    slice = -2;
    slice_no = -1;
    exec_kernel("gemm_nocopy_xe1", { a, b_t }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_nocopy_xe1 block n:2x1+1*3", c, c_ref);
    slice = -1;
    slice_no = 3;
    exec_kernel("gemm_nocopy_xe1", { a, b_t }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_nocopy_xe1 block n:1x3+2*1", c, c_ref);
}

// a: [M, K], b: [N, K]
void check_f16_xe2(PlainTensor& a, PlainTensor& b, PlainTensor& c_ref) {
    PlainTensor c;
    PlainTensor b_t;
    uint32_t M = a.size(0);
    uint32_t N = b.size(1);
    uint32_t K = a.size(1);
    b_t.resize<float16_t>({ K, N });
    c.resize<float16_t>({ M, N });
    transpose(K, N, (float16_t*)b.m_ptr.get(), (float16_t*)b_t.m_ptr.get());
    cl_int err;
    size_t BLOCK_SG_M = f16_xe2::BLOCK_SG_M;
    size_t BLOCK_SG_N = f16_xe2::BLOCK_SG_N;
    size_t SG_M = f16_xe2::SG_M, SG_N = f16_xe2::SG_N;
    size_t BLOCK_WG_M = BLOCK_SG_M * SG_M;
    size_t BLOCK_WG_N = BLOCK_SG_N * SG_N;
    std::vector<size_t> globalSize = { M / BLOCK_WG_M * N / BLOCK_WG_N * SG_N, SG_M};
    std::vector<size_t> localSize = { SG_N, SG_M };

    int slice;
    int slice_no;

    // if slice_no == 0, slice is linear type, 0 for loop M first and 1 for loop N first
    slice_no = 0;
    for (slice = 0; slice < 2; slice++) {
       exec_kernel("gemm_nocopy_xe2", { a, b_t }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice}, globalSize, localSize);
       cmp("gemm_nocopy_xe2 linear " + (slice == 0 ? std::string("M first") : std::string("N first")), c, c_ref);
    }
    // if slice_no != 0, use wg blocking schema:
    //    slice > 0 means the slice is in M dimension else is in N dimension
    //    slice_no > 0 means the slice size of reminder will be slice+1 else be slice-1
    slice = 2;
    slice_no = -2;
    exec_kernel("gemm_nocopy_xe2", { a, b_t }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_nocopy_xe2 block m:2x2+1*3", c, c_ref);
    slice = 1;
    slice_no = 3;
    exec_kernel("gemm_nocopy_xe2", { a, b_t }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_nocopy_xe2 block m:1x3+2*2", c, c_ref);

    slice = -2;
    slice_no = -1;
    exec_kernel("gemm_nocopy_xe2", { a, b_t }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_nocopy_xe2 block n:2x1+1*3", c, c_ref);
    slice = -1;
    slice_no = 3;
    exec_kernel("gemm_nocopy_xe2", { a, b_t }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_nocopy_xe2 block n:1x3+2*1", c, c_ref);
}

template<typename T>
void pack_u4(T* src, uint8_t* dst, size_t M, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j += 2) {
            uint8_t low = (uint8_t)(float(src[i * K + j + 0]));
            uint8_t high = (uint8_t)(float(src[i * K + j + 1]));
            dst[i * K / 2 + j / 2] = (uint8_t)(high << 4 | low);
        }
    }
}

void gen_weight_u4(size_t K, size_t N, size_t GROUP_SIZE, PlainTensor& b_zp, PlainTensor& b_scale, PlainTensor& b, PlainTensor& b_ref) {
    const size_t GROUP_NUM = K / GROUP_SIZE;
    // zp f16
    PlainTensor b_zp_init;
    b_zp_init.resize<float16_t>({ GROUP_NUM, N });
    fill(b_zp_init, 1.0f, 2);
    //b_zp_init = float16_t{ 0.0f };
    // zp f16->u4
    b_zp.resize<uint8_t>({ GROUP_NUM, N / 2 });
    pack_u4(b_zp_init.ptr<float16_t>(), b_zp.ptr<uint8_t>(), GROUP_NUM, N);
    // scale f16
    b_scale.resize<float16_t>({ GROUP_NUM, N });
    fill(b_scale, -1.0f, 3);
    //b_scale = float16_t{ 1.f };
    // weight f16
    PlainTensor b_init;
    b_init.resize<float16_t>({ K, N });
    fill(b_init, 0, 5);
    //b_init = float16_t{ 1.f };
    // weight f16->u4
    {
        PlainTensor b_t;
        b_t.resize<float16_t>({ N, K });
        // transpose
        transpose(K, N, b_init.ptr<float16_t>(), b_t.ptr<float16_t>());
        // pack f16->u4
        b.resize<uint8_t>({ N, K / 2 });
        pack_u4(b_t.ptr<float16_t>(), b.ptr<uint8_t>(), N, K);
    }
    // gen ref
    b_ref.resize<float16_t>({ K, N });
    {
        for (size_t i = 0; i < K; i++) {
            for (size_t j = 0; j < N; j++) {
                b_ref.ptr<float16_t>(i)[j] = (b_init.ptr<float16_t>(i)[j] - b_zp_init.ptr<float16_t>(i / GROUP_SIZE)[j]) * 
                    b_scale.ptr<float16_t>(i / GROUP_SIZE)[j];
            }
        }
    }
}

// a: [M, K], b(u4): [N, K]
void check_a16w4_xe2() {
    PlainTensor a, b, c, b_zp, b_scale, b_ref;
    uint32_t M = 256 * 7, N = 256 * 5, K = 128 * 2;
    const size_t GROUP_SIZE = a16u4_xe2::GROUP_SIZE;
    const size_t GROUP_NUM = K / GROUP_SIZE;
    a.resize<float16_t>({ M, K });
    //a = float16_t{ 1.0f };
    fill(a);
    gen_weight_u4(K, N, GROUP_SIZE, b_zp, b_scale, b, b_ref);
    auto c_ref = get_ref(a, b_ref);

    c.resize<float16_t>({ M, N });
    cl_int err;
    size_t BLOCK_SG_M = a16u4_xe2::BLOCK_SG_M;
    size_t BLOCK_SG_N = a16u4_xe2::BLOCK_SG_N;
    size_t SG_M = a16u4_xe2::SG_M, SG_N = a16u4_xe2::SG_N;
    size_t BLOCK_WG_M = BLOCK_SG_M * SG_M;
    size_t BLOCK_WG_N = BLOCK_SG_N * SG_N;
    std::vector<size_t> globalSize = { M / BLOCK_WG_M * N / BLOCK_WG_N * SG_N, SG_M};
    std::vector<size_t> localSize = { SG_N, SG_M };

    int slice;
    int slice_no;

    // if slice_no == 0, slice is linear type, 0 for loop M first and 1 for loop N first
    slice_no = 0;
    for (slice = 0; slice < 2; slice++) {
       exec_kernel("gemm_a16w4_xe2", { a, b, b_scale, b_zp }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice}, globalSize, localSize);
       cmp("gemm_a16w4_xe2 linear " + (slice == 0 ? std::string("M first") : std::string("N first")), c, c_ref);
    }
    // if slice_no != 0, use wg blocking schema:
    //    slice > 0 means the slice is in M dimension else is in N dimension
    //    slice_no > 0 means the slice size of reminder will be slice+1 else be slice-1
    slice = 2;
    slice_no = -2;
    exec_kernel("gemm_a16w4_xe2", { a, b, b_scale, b_zp }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_a16w4_xe2 block m:2x2+1*3", c, c_ref);
    slice = 1;
    slice_no = 3;
    exec_kernel("gemm_a16w4_xe2", { a, b, b_scale, b_zp }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_a16w4_xe2 block m:1x3+2*2", c, c_ref);

    slice = -2;
    slice_no = -1;
    exec_kernel("gemm_a16w4_xe2", { a, b, b_scale, b_zp }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_a16w4_xe2 block n:2x1+1*3", c, c_ref);
    slice = -1;
    slice_no = 3;
    exec_kernel("gemm_a16w4_xe2", { a, b, b_scale, b_zp }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_a16w4_xe2 block n:1x3+2*1", c, c_ref);
}

// a: [M, K], b(u4): [N, K]
void check_a16w4_xe1() {
    PlainTensor a, b, c, b_zp, b_scale, b_ref;
    uint32_t M = 128 * 7, N = 256 * 5, K = 128 * 2;
    const size_t GROUP_SIZE = a16u4_xe1::GROUP_SIZE;
    const size_t GROUP_NUM = K / GROUP_SIZE;
    a.resize<float16_t>({ M, K });
    //a = float16_t{ 1.0f };
    fill(a);
    gen_weight_u4(K, N, GROUP_SIZE, b_zp, b_scale, b, b_ref);
    auto c_ref = get_ref(a, b_ref);

    c.resize<float16_t>({ M, N });
    cl_int err;
    size_t BLOCK_SG_M = a16u4_xe1::BLOCK_SG_M;
    size_t BLOCK_SG_N = a16u4_xe1::BLOCK_SG_N;
    size_t SG_M = a16u4_xe1::SG_M, SG_N = a16u4_xe1::SG_N;
    size_t BLOCK_WG_M = BLOCK_SG_M * SG_M;
    size_t BLOCK_WG_N = BLOCK_SG_N * SG_N;
    std::vector<size_t> globalSize = { M / BLOCK_WG_M * N / BLOCK_WG_N * SG_N, SG_M};
    std::vector<size_t> localSize = { SG_N, SG_M };

    int slice;
    int slice_no;

    // if slice_no == 0, slice is linear type, 0 for loop M first and 1 for loop N first
    slice_no = 0;
    for (slice = 0; slice < 2; slice++) {
       exec_kernel("gemm_a16w4_xe1", { a, b, b_scale, b_zp }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice}, globalSize, localSize);
       cmp("gemm_a16w4_xe1 linear " + (slice == 0 ? std::string("M first") : std::string("N first")), c, c_ref);
    }
    // if slice_no != 0, use wg blocking schema:
    //    slice > 0 means the slice is in M dimension else is in N dimension
    //    slice_no > 0 means the slice size of reminder will be slice+1 else be slice-1
    slice = 2;
    slice_no = -2;
    exec_kernel("gemm_a16w4_xe1", { a, b, b_scale, b_zp }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_a16w4_xe1 block m:2x2+1*3", c, c_ref);
    slice = 1;
    slice_no = 3;
    exec_kernel("gemm_a16w4_xe1", { a, b, b_scale, b_zp }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_a16w4_xe1 block m:1x3+2*2", c, c_ref);

    slice = -2;
    slice_no = -1;
    exec_kernel("gemm_a16w4_xe1", { a, b, b_scale, b_zp }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_a16w4_xe1 block n:2x1+1*3", c, c_ref);
    slice = -1;
    slice_no = 3;
    exec_kernel("gemm_a16w4_xe1", { a, b, b_scale, b_zp }, { c }, { M, N, K, K, K, N, (uint32_t)slice_no, (uint32_t)slice }, globalSize, localSize);
    cmp("gemm_a16w4_xe1 block n:1x3+2*1", c, c_ref);
}

int main( int argc, char* argv[])
{
    init_ocl();
    if (0) {
        PlainTensor a, b, c, b_repack_ref, b_repack;
        uint32_t M = 256 * 7, N = 256 * 5, K = 64 * 2;
        a.resize<float16_t>({ M, K });
        b.resize<float16_t>({ K, N });

        //a = float16_t{ 1.0f };
        fill(a);
        //b = float16_t{ 1.0f };
        fill(b);
        auto c_ref = get_ref(a, b);

        //check_v1(a, b, c_ref);
        check_f16_xe1(a, b, c_ref);
        //check_f16_xe2(a, b, c_ref);
    } else {
        check_a16w4_xe1();
        //check_a16w4_xe2();
    }

    uninit_ocl();

    fprintf(stderr, "PASSED\n");
    return 0;
}
