#include <cstddef>
#define CL_HPP_ENABLE_EXCEPTIONS

#include "common.hpp"

// https://github.com/intel/pti-gpu

int main(void) {
    // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html
    //
    //
    select_default_platform({"cl_intel_subgroups","cl_intel_required_subgroup_size"});

    cl::CommandQueue::setDefault(cl::CommandQueue(cl::QueueProperties::Profiling));

    // flushing denormals to zero on CPU side
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    // https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html#work-item-functions
    // 
    CLkernels kernel(R"CLC(
        ulong __attribute__((overloadable)) intel_get_cycle_counter( void );
        uint __attribute__((overloadable)) intel_get_slice_id( void );
        uint __attribute__((overloadable)) intel_get_subslice_id( void );
        uint __attribute__((overloadable)) intel_get_dual_subslice_id( void );
        uint __attribute__((overloadable)) intel_get_eu_id( void );
        uint __attribute__((overloadable)) intel_get_eu_thread_id( void );

        #define FMACNT 8
        #define UNROLL 4

        #define uint32_t uint
        #define uint64_t ulong

        struct workitem_info {
            uint32_t group_id0;
            uint32_t group_id1;
            uint32_t local_id0;
            uint32_t local_id1;
            uint32_t sub_group_id;
            uint32_t sub_group_local_id;
            uint32_t slice_id;
            uint32_t sub_slice_id;
            uint32_t eu_id;
            uint32_t eu_slot_id;
            uint64_t cycle_start;
            uint64_t cycle_dur;
        };

        __kernel void test_ids0(__global uint* id) {
            id[0] = 0x12345678;
        }

        __kernel void test_ids1(__global uint* id) {
            id[0] = intel_get_eu_thread_id();
        }

        __kernel void test_ids2(__global uint* id) {
            id[0] = intel_get_eu_id();
        }

        __kernel void test_cycles(__global ulong* id) {
            id[0] = intel_get_cycle_counter();
        }

        void set_winfo(struct workitem_info * pw) {
            pw->group_id0 = get_group_id(0);
            pw->group_id1 = get_group_id(1);
            pw->local_id0 = get_local_id(0);
            pw->local_id1 = get_global_id(0); //get_local_id(1);
            pw->sub_group_id = get_sub_group_id();
            pw->sub_group_local_id = get_sub_group_local_id();
            pw->slice_id = intel_get_slice_id();
            pw->sub_slice_id = intel_get_dual_subslice_id();
            pw->eu_id = intel_get_eu_id();
            pw->eu_slot_id = intel_get_eu_thread_id();
        }

        __attribute__((intel_reqd_sub_group_size(8)))
        __kernel void add_f32(float val, int idx, int size, __global float *result, __global struct workitem_info * winfo//, __global uint* eu_id, __global uint* dss_id
            ) {
            __local float tmp[64 * 1024 / 4];
            int id_local = get_local_id(0);
            int id_global = get_global_id(0);
            int id_sg_local = get_sub_group_local_id();
            int local_size = get_local_size(0);
            //tmp[id_local*100] = val * 10;
            for (int i = id_local; i < 64 * 1024 / 4; i+= local_size)
                tmp[i] = id_local;
            barrier(CLK_LOCAL_MEM_FENCE);
            float x = tmp[id_global + idx] * 10;
            float16 s = 0, s1 = 0, s2 = 0, s3 = 0;
            ulong beg = intel_get_cycle_counter();
            __attribute__((opencl_unroll_hint(1)))
            for (int i = 0; i < size; i++) {
                s.s0 += x;
                s.s1 += x;
                s.s2 += x;
                s.s3 += x;
                s.s4 += x;
                s.s5 += x;
                s.s6 += x;
                s.s7 += x;
                s.s8 += x;
                s.s9 += x;
                s.sa += x;
                s.sb += x;
                s.sc += x;
                s.sd += x;
                s.se += x;
                s.sf += x;
        #define ADD(n) \
                s##n.s0 += x; \
                s##n.s1 += x; \
                s##n.s2 += x; \
                s##n.s3 += x; \
                s##n.s4 += x; \
                s##n.s5 += x; \
                s##n.s6 += x; \
                s##n.s7 += x; \
                s##n.s8 += x; \
                s##n.s9 += x; \
                s##n.sa += x; \
                s##n.sb += x; \
                s##n.sc += x; \
                s##n.sd += x; \
                s##n.se += x; \
                s##n.sf += x;

                ADD(1);
                ADD(2);
                ADD(3);
            }
            ulong end = intel_get_cycle_counter();
            for (int i = 1; i < 16; i++)
                s[0] += s[i];
            for (int i = 0; i < 16; i++) {
                s[0] += s1[i];
                s[0] += s2[i];
                s[0] += s3[i];
            }
            *result = s[0];
            // (get_global_id(1) - get_global_offset(1)) * get_global_size(0) + (get_global_id(0) - get_global_offset(0))
            if (id_sg_local == 0) {
                struct workitem_info * pw = winfo + id_global / 8; //get_global_linear_id();
                set_winfo(pw);
                pw->cycle_start = beg;
                pw->cycle_dur = end - beg;
            }

            for (int i = id_local; i < 64 * 1024 / 4; i+= local_size)
                tmp[i] = id_local;
            barrier(CLK_GLOBAL_MEM_FENCE);
            if (id_sg_local == 0) {
                //ts[id_global / 8] = end - beg;
                //ts[0 / 8] = end - beg;
                //eu_id[id_global / 8] = intel_get_eu_id();
                //dss_id[id_global / 8] = intel_get_dual_subslice_id();
            }
        }
        __attribute__((intel_reqd_sub_group_size(8)))
        __kernel void fill_id(
                __global struct workitem_info * winfo,
                __global int * info,
                int K) {

            __local float Asub[128*1024/8];

            // (get_global_id(1) - get_global_offset(1)) * get_global_size(0) + (get_global_id(0) - get_global_offset(0))
            struct workitem_info * pw = winfo + get_global_linear_id();
            set_winfo(pw);

            barrier(CLK_LOCAL_MEM_FENCE);

            float c[FMACNT];
            for(int j = 0; j < FMACNT; j++) c[j] = j;

            float a = get_local_id(0);
            float b = get_local_id(1);

            pw->cycle_start = intel_get_cycle_counter();

            for(int k = 0; k < K; k += FMACNT*UNROLL) {
                // following loop will be unrolled
                for(int unroll = 0; unroll < UNROLL; unroll ++)
                    for(int j = 0; j < FMACNT; j++)
                        c[j] = fma(a, b, c[j]);
            }

            pw->cycle_dur = intel_get_cycle_counter() - pw->cycle_start;

            // prevent optimization
            float sum_c = 0;
            for(int j = 0; j < FMACNT; j++) sum_c += c[j];
            if (sum_c == 0) {
                Asub[(int)sum_c] = 1;
                info[(int)sum_c*2] = Asub[(int)sum_c/2];
            }

            /*
            if (i == 0) {
                info[0] = get_global_size(0);
                info[1] = get_global_size(1);
                info[2] = get_local_size(0);
                info[3] = get_local_size(1);
                info[4] = get_num_groups(0);
                info[5] = get_num_groups(1);
                info[6] = get_sub_group_size();
                info[7] = get_max_sub_group_size();
                info[8] = get_num_sub_groups();
                info[8] = sizeof(ulong);
            }*/
        }
            )CLC");

    size_t M = getenv("M", 1024);
    size_t N = getenv("N", 1);
    int K = getenv("K", 4096000);
    size_t LM = getenv("LM", 32);
    size_t LN = getenv("LN", 32);

    //tensorND<workitem_info> winfo({M, N}, workitem_info());
    tensorND<int> info({16}, -1);

    kernel.show_info("add_f32", {1024,1024}, 3);

    int precision = 2, width = 3;

    // cl::EnqueueArgs enqArgs({N, M},{LN, LM});
    size_t local = 8 * 16;
    size_t global = N * local;
    cl::EnqueueArgs enqArgs(global,local);
    tensorND<workitem_info> winfo({global / 8}, workitem_info());
    // float val, int idx, int size, __global float *result, __global struct workitem_info * winfo
    auto evt = kernel.call("add_f32", enqArgs, 1.0f, 0, 1000000, info.to_gpu(), winfo.to_gpu());
    evt.wait();

    cl_ulong last_evt_ns = 0;
    auto delta_ns = [&](cl_ulong ns){
        auto delta = ns - last_evt_ns;
        last_evt_ns = ns;
        return Nanoseconds(delta);
    };

    winfo.to_cpu();
    info.to_cpu();
    std::cout << "info = " << info.repr(precision, width) << "\n";
    //std::cout << "ids = " << ids.repr(precision, width) << "\n";
    std::cout << "===========================" << std::endl;

    ECOUT("CL_PROFILING_COMMAND_QUEUED  :  ", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()));
    ECOUT("CL_PROFILING_COMMAND_SUBMIT  : +", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>()));
    ECOUT("CL_PROFILING_COMMAND_START   : +", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_START>()));
    ECOUT("CL_PROFILING_COMMAND_END     : +", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_END>()));
    ECOUT("CL_PROFILING_COMMAND_COMPLETE: +", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_COMPLETE>()));

    auto latency_ns = (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() - evt.getProfilingInfo<CL_PROFILING_COMMAND_START>());

    workitem_info::Dump(winfo, latency_ns, K, 8);

    return 0;
}