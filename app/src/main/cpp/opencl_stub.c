/*
 * OpenCL runtime loader for Android.
 *
 * Key design:
 * 1. All dlopen/dlsym happens in __attribute__((constructor)) which runs
 *    on the MAIN thread (8MB stack) when System.loadLibrary() loads us.
 * 2. Only ABSOLUTE PATHS are tried — bare "libOpenCL.so" would resolve to
 *    ourselves since our SONAME is also libOpenCL.so.
 * 3. If vendor lib isn't found, all functions return CL_DEVICE_NOT_FOUND
 *    so ggml gracefully falls back to CPU.
 */
#include <dlfcn.h>
#include <stddef.h>
#include <android/log.h>

#define TAG "opencl-loader"
#define CL_SUCCESS 0
#define CL_DEVICE_NOT_FOUND -1

typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long long cl_bitfield;

/* ================================================================== */
/* Cached function pointers — resolved once at load time              */
/* ================================================================== */
static void *g_lib = NULL;

#define DECL_FN(name) static void* g_##name = NULL;
DECL_FN(clGetPlatformIDs)
DECL_FN(clGetPlatformInfo)
DECL_FN(clGetDeviceIDs)
DECL_FN(clGetDeviceInfo)
DECL_FN(clCreateContext)
DECL_FN(clReleaseContext)
DECL_FN(clRetainContext)
DECL_FN(clGetContextInfo)
DECL_FN(clCreateCommandQueue)
DECL_FN(clCreateCommandQueueWithProperties)
DECL_FN(clReleaseCommandQueue)
DECL_FN(clRetainCommandQueue)
DECL_FN(clGetCommandQueueInfo)
DECL_FN(clFlush)
DECL_FN(clFinish)
DECL_FN(clCreateBuffer)
DECL_FN(clCreateSubBuffer)
DECL_FN(clCreateBufferWithProperties)
DECL_FN(clReleaseMemObject)
DECL_FN(clRetainMemObject)
DECL_FN(clGetMemObjectInfo)
DECL_FN(clCreateImage)
DECL_FN(clGetImageInfo)
DECL_FN(clCreateProgramWithSource)
DECL_FN(clCreateProgramWithBinary)
DECL_FN(clBuildProgram)
DECL_FN(clGetProgramBuildInfo)
DECL_FN(clGetProgramInfo)
DECL_FN(clReleaseProgram)
DECL_FN(clRetainProgram)
DECL_FN(clCreateKernel)
DECL_FN(clSetKernelArg)
DECL_FN(clReleaseKernel)
DECL_FN(clRetainKernel)
DECL_FN(clGetKernelWorkGroupInfo)
DECL_FN(clEnqueueNDRangeKernel)
DECL_FN(clEnqueueReadBuffer)
DECL_FN(clEnqueueWriteBuffer)
DECL_FN(clEnqueueCopyBuffer)
DECL_FN(clEnqueueMapBuffer)
DECL_FN(clEnqueueUnmapMemObject)
DECL_FN(clEnqueueFillBuffer)
DECL_FN(clEnqueueBarrierWithWaitList)
DECL_FN(clEnqueueMarkerWithWaitList)
DECL_FN(clWaitForEvents)
DECL_FN(clReleaseEvent)
DECL_FN(clRetainEvent)
DECL_FN(clGetEventProfilingInfo)
DECL_FN(clGetEventInfo)
DECL_FN(clSetEventCallback)
DECL_FN(clCreateUserEvent)
DECL_FN(clSetUserEventStatus)
DECL_FN(clSVMAlloc)
DECL_FN(clSVMFree)
DECL_FN(clEnqueueSVMMap)
DECL_FN(clEnqueueSVMUnmap)
DECL_FN(clEnqueueSVMMemcpy)
DECL_FN(clEnqueueSVMMemFill)
DECL_FN(clSetKernelArgSVMPointer)
#undef DECL_FN

#define RESOLVE(name) g_##name = dlsym(g_lib, #name)

__attribute__((constructor))
static void opencl_loader_init(void) {
    static const char *paths[] = {
        "/vendor/lib64/libOpenCL.so",
        "/system/vendor/lib64/libOpenCL.so",
        "/vendor/lib64/egl/libOpenCL.so",
        "/system/lib64/libOpenCL.so",
        NULL
    };

    for (int i = 0; paths[i]; i++) {
        g_lib = dlopen(paths[i], RTLD_NOW);
        if (g_lib) {
            __android_log_print(ANDROID_LOG_INFO, TAG,
                "SUCCESS: Loaded vendor OpenCL from %s", paths[i]);
            break;
        }
    }

    if (!g_lib) {
        __android_log_print(ANDROID_LOG_WARN, TAG,
            "FAILED: vendor OpenCL not accessible (%s) — CPU fallback",
            dlerror());
        return;
    }

    RESOLVE(clGetPlatformIDs);
    RESOLVE(clGetPlatformInfo);
    RESOLVE(clGetDeviceIDs);
    RESOLVE(clGetDeviceInfo);
    RESOLVE(clCreateContext);
    RESOLVE(clReleaseContext);
    RESOLVE(clRetainContext);
    RESOLVE(clGetContextInfo);
    RESOLVE(clCreateCommandQueue);
    RESOLVE(clCreateCommandQueueWithProperties);
    RESOLVE(clReleaseCommandQueue);
    RESOLVE(clRetainCommandQueue);
    RESOLVE(clGetCommandQueueInfo);
    RESOLVE(clFlush);
    RESOLVE(clFinish);
    RESOLVE(clCreateBuffer);
    RESOLVE(clCreateSubBuffer);
    RESOLVE(clCreateBufferWithProperties);
    RESOLVE(clReleaseMemObject);
    RESOLVE(clRetainMemObject);
    RESOLVE(clGetMemObjectInfo);
    RESOLVE(clCreateImage);
    RESOLVE(clGetImageInfo);
    RESOLVE(clCreateProgramWithSource);
    RESOLVE(clCreateProgramWithBinary);
    RESOLVE(clBuildProgram);
    RESOLVE(clGetProgramBuildInfo);
    RESOLVE(clGetProgramInfo);
    RESOLVE(clReleaseProgram);
    RESOLVE(clRetainProgram);
    RESOLVE(clCreateKernel);
    RESOLVE(clSetKernelArg);
    RESOLVE(clReleaseKernel);
    RESOLVE(clRetainKernel);
    RESOLVE(clGetKernelWorkGroupInfo);
    RESOLVE(clEnqueueNDRangeKernel);
    RESOLVE(clEnqueueReadBuffer);
    RESOLVE(clEnqueueWriteBuffer);
    RESOLVE(clEnqueueCopyBuffer);
    RESOLVE(clEnqueueMapBuffer);
    RESOLVE(clEnqueueUnmapMemObject);
    RESOLVE(clEnqueueFillBuffer);
    RESOLVE(clEnqueueBarrierWithWaitList);
    RESOLVE(clEnqueueMarkerWithWaitList);
    RESOLVE(clWaitForEvents);
    RESOLVE(clReleaseEvent);
    RESOLVE(clRetainEvent);
    RESOLVE(clGetEventProfilingInfo);
    RESOLVE(clGetEventInfo);
    RESOLVE(clSetEventCallback);
    RESOLVE(clCreateUserEvent);
    RESOLVE(clSetUserEventStatus);
    RESOLVE(clSVMAlloc);
    RESOLVE(clSVMFree);
    RESOLVE(clEnqueueSVMMap);
    RESOLVE(clEnqueueSVMUnmap);
    RESOLVE(clEnqueueSVMMemcpy);
    RESOLVE(clEnqueueSVMMemFill);
    RESOLVE(clSetKernelArgSVMPointer);

    __android_log_print(ANDROID_LOG_INFO, TAG, "All CL symbols resolved OK");
}
#undef RESOLVE

/* ================================================================== */
/* Forwarding functions — no dlsym at call time, just pointer calls   */
/* ================================================================== */

cl_int clGetPlatformIDs(cl_uint n, void *p, cl_uint *np) {
    typedef cl_int (*fn_t)(cl_uint, void*, cl_uint*);
    if (g_clGetPlatformIDs) return ((fn_t)g_clGetPlatformIDs)(n, p, np);
    if (np) *np = 0;
    return CL_DEVICE_NOT_FOUND;
}

cl_int clGetPlatformInfo(void *p, cl_uint i, size_t s, void *v, size_t *r) {
    typedef cl_int (*fn_t)(void*, cl_uint, size_t, void*, size_t*);
    if (g_clGetPlatformInfo) return ((fn_t)g_clGetPlatformInfo)(p, i, s, v, r);
    return CL_DEVICE_NOT_FOUND;
}

cl_int clGetDeviceIDs(void *p, cl_bitfield t, cl_uint n, void *d, cl_uint *nd) {
    typedef cl_int (*fn_t)(void*, cl_bitfield, cl_uint, void*, cl_uint*);
    if (g_clGetDeviceIDs) return ((fn_t)g_clGetDeviceIDs)(p, t, n, d, nd);
    if (nd) *nd = 0;
    return CL_DEVICE_NOT_FOUND;
}

cl_int clGetDeviceInfo(void *d, cl_uint i, size_t s, void *v, size_t *r) {
    typedef cl_int (*fn_t)(void*, cl_uint, size_t, void*, size_t*);
    if (g_clGetDeviceInfo) return ((fn_t)g_clGetDeviceInfo)(d, i, s, v, r);
    return CL_DEVICE_NOT_FOUND;
}

void* clCreateContext(const void *pr, cl_uint n, const void *d, void *cb, void *ud, cl_int *e) {
    typedef void* (*fn_t)(const void*, cl_uint, const void*, void*, void*, cl_int*);
    if (g_clCreateContext) return ((fn_t)g_clCreateContext)(pr, n, d, cb, ud, e);
    if (e) *e = CL_DEVICE_NOT_FOUND; return NULL;
}

cl_int clReleaseContext(void *c) { typedef cl_int(*fn_t)(void*); return g_clReleaseContext ? ((fn_t)g_clReleaseContext)(c) : CL_SUCCESS; }
cl_int clRetainContext(void *c) { typedef cl_int(*fn_t)(void*); return g_clRetainContext ? ((fn_t)g_clRetainContext)(c) : CL_SUCCESS; }

cl_int clGetContextInfo(void *c, cl_uint i, size_t s, void *v, size_t *r) {
    typedef cl_int (*fn_t)(void*, cl_uint, size_t, void*, size_t*);
    if (g_clGetContextInfo) return ((fn_t)g_clGetContextInfo)(c, i, s, v, r);
    return CL_DEVICE_NOT_FOUND;
}

void* clCreateCommandQueue(void *c, void *d, cl_bitfield p, cl_int *e) {
    typedef void* (*fn_t)(void*, void*, cl_bitfield, cl_int*);
    if (g_clCreateCommandQueue) return ((fn_t)g_clCreateCommandQueue)(c, d, p, e);
    if (e) *e = CL_DEVICE_NOT_FOUND; return NULL;
}

void* clCreateCommandQueueWithProperties(void *c, void *d, const void *p, cl_int *e) {
    typedef void* (*fn_t)(void*, void*, const void*, cl_int*);
    if (g_clCreateCommandQueueWithProperties) return ((fn_t)g_clCreateCommandQueueWithProperties)(c, d, p, e);
    if (e) *e = CL_DEVICE_NOT_FOUND; return NULL;
}

cl_int clReleaseCommandQueue(void *q) { typedef cl_int(*fn_t)(void*); return g_clReleaseCommandQueue ? ((fn_t)g_clReleaseCommandQueue)(q) : CL_SUCCESS; }
cl_int clRetainCommandQueue(void *q) { typedef cl_int(*fn_t)(void*); return g_clRetainCommandQueue ? ((fn_t)g_clRetainCommandQueue)(q) : CL_SUCCESS; }
cl_int clGetCommandQueueInfo(void *q, cl_uint i, size_t s, void *v, size_t *r) { typedef cl_int(*fn_t)(void*,cl_uint,size_t,void*,size_t*); return g_clGetCommandQueueInfo ? ((fn_t)g_clGetCommandQueueInfo)(q,i,s,v,r) : CL_DEVICE_NOT_FOUND; }
cl_int clFlush(void *q) { typedef cl_int(*fn_t)(void*); return g_clFlush ? ((fn_t)g_clFlush)(q) : CL_SUCCESS; }
cl_int clFinish(void *q) { typedef cl_int(*fn_t)(void*); return g_clFinish ? ((fn_t)g_clFinish)(q) : CL_SUCCESS; }

void* clCreateBuffer(void *c, cl_bitfield f, size_t s, void *h, cl_int *e) {
    typedef void* (*fn_t)(void*, cl_bitfield, size_t, void*, cl_int*);
    if (g_clCreateBuffer) return ((fn_t)g_clCreateBuffer)(c, f, s, h, e);
    if (e) *e = CL_DEVICE_NOT_FOUND; return NULL;
}

void* clCreateSubBuffer(void *b, cl_bitfield f, cl_uint t, const void *i, cl_int *e) {
    typedef void* (*fn_t)(void*, cl_bitfield, cl_uint, const void*, cl_int*);
    if (g_clCreateSubBuffer) return ((fn_t)g_clCreateSubBuffer)(b, f, t, i, e);
    if (e) *e = CL_DEVICE_NOT_FOUND; return NULL;
}

void* clCreateBufferWithProperties(void *c, const void *pr, cl_bitfield f, size_t s, void *h, cl_int *e) {
    typedef void* (*fn_t)(void*, const void*, cl_bitfield, size_t, void*, cl_int*);
    if (g_clCreateBufferWithProperties) return ((fn_t)g_clCreateBufferWithProperties)(c, pr, f, s, h, e);
    if (e) *e = CL_DEVICE_NOT_FOUND; return NULL;
}

cl_int clReleaseMemObject(void *m) { typedef cl_int(*fn_t)(void*); return g_clReleaseMemObject ? ((fn_t)g_clReleaseMemObject)(m) : CL_SUCCESS; }
cl_int clRetainMemObject(void *m) { typedef cl_int(*fn_t)(void*); return g_clRetainMemObject ? ((fn_t)g_clRetainMemObject)(m) : CL_SUCCESS; }
cl_int clGetMemObjectInfo(void *m, cl_uint i, size_t s, void *v, size_t *r) { typedef cl_int(*fn_t)(void*,cl_uint,size_t,void*,size_t*); return g_clGetMemObjectInfo ? ((fn_t)g_clGetMemObjectInfo)(m,i,s,v,r) : CL_DEVICE_NOT_FOUND; }

void* clCreateImage(void *c, cl_bitfield f, const void *fmt, const void *desc, void *h, cl_int *e) {
    typedef void* (*fn_t)(void*, cl_bitfield, const void*, const void*, void*, cl_int*);
    if (g_clCreateImage) return ((fn_t)g_clCreateImage)(c, f, fmt, desc, h, e);
    if (e) *e = CL_DEVICE_NOT_FOUND; return NULL;
}
cl_int clGetImageInfo(void *img, cl_uint i, size_t s, void *v, size_t *r) { typedef cl_int(*fn_t)(void*,cl_uint,size_t,void*,size_t*); return g_clGetImageInfo ? ((fn_t)g_clGetImageInfo)(img,i,s,v,r) : CL_DEVICE_NOT_FOUND; }

void* clCreateProgramWithSource(void *c, cl_uint n, const char **s, const size_t *l, cl_int *e) {
    typedef void* (*fn_t)(void*, cl_uint, const char**, const size_t*, cl_int*);
    if (g_clCreateProgramWithSource) return ((fn_t)g_clCreateProgramWithSource)(c, n, s, l, e);
    if (e) *e = CL_DEVICE_NOT_FOUND; return NULL;
}

void* clCreateProgramWithBinary(void *c, cl_uint n, const void *d, const size_t *l, const unsigned char **b, cl_int *bs, cl_int *e) {
    typedef void* (*fn_t)(void*, cl_uint, const void*, const size_t*, const unsigned char**, cl_int*, cl_int*);
    if (g_clCreateProgramWithBinary) return ((fn_t)g_clCreateProgramWithBinary)(c, n, d, l, b, bs, e);
    if (e) *e = CL_DEVICE_NOT_FOUND; return NULL;
}

cl_int clBuildProgram(void *p, cl_uint n, const void *d, const char *o, void *cb, void *ud) { typedef cl_int(*fn_t)(void*,cl_uint,const void*,const char*,void*,void*); return g_clBuildProgram ? ((fn_t)g_clBuildProgram)(p,n,d,o,cb,ud) : CL_DEVICE_NOT_FOUND; }
cl_int clGetProgramBuildInfo(void *p, void *d, cl_uint i, size_t s, void *v, size_t *r) { typedef cl_int(*fn_t)(void*,void*,cl_uint,size_t,void*,size_t*); return g_clGetProgramBuildInfo ? ((fn_t)g_clGetProgramBuildInfo)(p,d,i,s,v,r) : CL_DEVICE_NOT_FOUND; }
cl_int clGetProgramInfo(void *p, cl_uint i, size_t s, void *v, size_t *r) { typedef cl_int(*fn_t)(void*,cl_uint,size_t,void*,size_t*); return g_clGetProgramInfo ? ((fn_t)g_clGetProgramInfo)(p,i,s,v,r) : CL_DEVICE_NOT_FOUND; }
cl_int clReleaseProgram(void *p) { typedef cl_int(*fn_t)(void*); return g_clReleaseProgram ? ((fn_t)g_clReleaseProgram)(p) : CL_SUCCESS; }
cl_int clRetainProgram(void *p) { typedef cl_int(*fn_t)(void*); return g_clRetainProgram ? ((fn_t)g_clRetainProgram)(p) : CL_SUCCESS; }

void* clCreateKernel(void *p, const char *name, cl_int *e) {
    typedef void* (*fn_t)(void*, const char*, cl_int*);
    if (g_clCreateKernel) return ((fn_t)g_clCreateKernel)(p, name, e);
    if (e) *e = CL_DEVICE_NOT_FOUND; return NULL;
}

cl_int clSetKernelArg(void *k, cl_uint i, size_t s, const void *v) { typedef cl_int(*fn_t)(void*,cl_uint,size_t,const void*); return g_clSetKernelArg ? ((fn_t)g_clSetKernelArg)(k,i,s,v) : CL_DEVICE_NOT_FOUND; }
cl_int clReleaseKernel(void *k) { typedef cl_int(*fn_t)(void*); return g_clReleaseKernel ? ((fn_t)g_clReleaseKernel)(k) : CL_SUCCESS; }
cl_int clRetainKernel(void *k) { typedef cl_int(*fn_t)(void*); return g_clRetainKernel ? ((fn_t)g_clRetainKernel)(k) : CL_SUCCESS; }
cl_int clGetKernelWorkGroupInfo(void *k, void *d, cl_uint i, size_t s, void *v, size_t *r) { typedef cl_int(*fn_t)(void*,void*,cl_uint,size_t,void*,size_t*); return g_clGetKernelWorkGroupInfo ? ((fn_t)g_clGetKernelWorkGroupInfo)(k,d,i,s,v,r) : CL_DEVICE_NOT_FOUND; }

cl_int clEnqueueNDRangeKernel(void *q, void *k, cl_uint w, const size_t *go, const size_t *gs, const size_t *ls, cl_uint ne, const void *el, void *ev) { typedef cl_int(*fn_t)(void*,void*,cl_uint,const size_t*,const size_t*,const size_t*,cl_uint,const void*,void*); return g_clEnqueueNDRangeKernel ? ((fn_t)g_clEnqueueNDRangeKernel)(q,k,w,go,gs,ls,ne,el,ev) : CL_DEVICE_NOT_FOUND; }
cl_int clEnqueueReadBuffer(void *q, void *b, cl_uint bk, size_t o, size_t s, void *p, cl_uint ne, const void *el, void *ev) { typedef cl_int(*fn_t)(void*,void*,cl_uint,size_t,size_t,void*,cl_uint,const void*,void*); return g_clEnqueueReadBuffer ? ((fn_t)g_clEnqueueReadBuffer)(q,b,bk,o,s,p,ne,el,ev) : CL_DEVICE_NOT_FOUND; }
cl_int clEnqueueWriteBuffer(void *q, void *b, cl_uint bk, size_t o, size_t s, const void *p, cl_uint ne, const void *el, void *ev) { typedef cl_int(*fn_t)(void*,void*,cl_uint,size_t,size_t,const void*,cl_uint,const void*,void*); return g_clEnqueueWriteBuffer ? ((fn_t)g_clEnqueueWriteBuffer)(q,b,bk,o,s,p,ne,el,ev) : CL_DEVICE_NOT_FOUND; }
cl_int clEnqueueCopyBuffer(void *q, void *s2, void *d, size_t so, size_t d2, size_t cb, cl_uint ne, const void *el, void *ev) { typedef cl_int(*fn_t)(void*,void*,void*,size_t,size_t,size_t,cl_uint,const void*,void*); return g_clEnqueueCopyBuffer ? ((fn_t)g_clEnqueueCopyBuffer)(q,s2,d,so,d2,cb,ne,el,ev) : CL_DEVICE_NOT_FOUND; }

void* clEnqueueMapBuffer(void *q, void *b, cl_uint bk, cl_bitfield f, size_t o, size_t s, cl_uint ne, const void *el, void *ev, cl_int *e) {
    typedef void* (*fn_t)(void*,void*,cl_uint,cl_bitfield,size_t,size_t,cl_uint,const void*,void*,cl_int*);
    if (g_clEnqueueMapBuffer) return ((fn_t)g_clEnqueueMapBuffer)(q,b,bk,f,o,s,ne,el,ev,e);
    if (e) *e = CL_DEVICE_NOT_FOUND; return NULL;
}

cl_int clEnqueueUnmapMemObject(void *q, void *m, void *p, cl_uint ne, const void *el, void *ev) { typedef cl_int(*fn_t)(void*,void*,void*,cl_uint,const void*,void*); return g_clEnqueueUnmapMemObject ? ((fn_t)g_clEnqueueUnmapMemObject)(q,m,p,ne,el,ev) : CL_DEVICE_NOT_FOUND; }
cl_int clEnqueueFillBuffer(void *q, void *b, const void *pat, size_t ps, size_t off, size_t s, cl_uint ne, const void *el, void *ev) { typedef cl_int(*fn_t)(void*,void*,const void*,size_t,size_t,size_t,cl_uint,const void*,void*); return g_clEnqueueFillBuffer ? ((fn_t)g_clEnqueueFillBuffer)(q,b,pat,ps,off,s,ne,el,ev) : CL_DEVICE_NOT_FOUND; }

cl_int clEnqueueBarrierWithWaitList(void *q, cl_uint ne, const void *el, void *ev) { typedef cl_int(*fn_t)(void*,cl_uint,const void*,void*); return g_clEnqueueBarrierWithWaitList ? ((fn_t)g_clEnqueueBarrierWithWaitList)(q,ne,el,ev) : CL_SUCCESS; }
cl_int clEnqueueMarkerWithWaitList(void *q, cl_uint ne, const void *el, void *ev) { typedef cl_int(*fn_t)(void*,cl_uint,const void*,void*); return g_clEnqueueMarkerWithWaitList ? ((fn_t)g_clEnqueueMarkerWithWaitList)(q,ne,el,ev) : CL_SUCCESS; }

cl_int clWaitForEvents(cl_uint n, const void *l) { typedef cl_int(*fn_t)(cl_uint,const void*); return g_clWaitForEvents ? ((fn_t)g_clWaitForEvents)(n,l) : CL_SUCCESS; }
cl_int clReleaseEvent(void *ev) { typedef cl_int(*fn_t)(void*); return g_clReleaseEvent ? ((fn_t)g_clReleaseEvent)(ev) : CL_SUCCESS; }
cl_int clRetainEvent(void *ev) { typedef cl_int(*fn_t)(void*); return g_clRetainEvent ? ((fn_t)g_clRetainEvent)(ev) : CL_SUCCESS; }
cl_int clGetEventProfilingInfo(void *ev, cl_uint i, size_t s, void *v, size_t *r) { typedef cl_int(*fn_t)(void*,cl_uint,size_t,void*,size_t*); return g_clGetEventProfilingInfo ? ((fn_t)g_clGetEventProfilingInfo)(ev,i,s,v,r) : CL_DEVICE_NOT_FOUND; }
cl_int clGetEventInfo(void *ev, cl_uint i, size_t s, void *v, size_t *r) { typedef cl_int(*fn_t)(void*,cl_uint,size_t,void*,size_t*); return g_clGetEventInfo ? ((fn_t)g_clGetEventInfo)(ev,i,s,v,r) : CL_DEVICE_NOT_FOUND; }
cl_int clSetEventCallback(void *ev, cl_int t, void *cb, void *ud) { typedef cl_int(*fn_t)(void*,cl_int,void*,void*); return g_clSetEventCallback ? ((fn_t)g_clSetEventCallback)(ev,t,cb,ud) : CL_DEVICE_NOT_FOUND; }
void* clCreateUserEvent(void *c, cl_int *e) { typedef void*(*fn_t)(void*,cl_int*); if (g_clCreateUserEvent) return ((fn_t)g_clCreateUserEvent)(c,e); if(e)*e=CL_DEVICE_NOT_FOUND; return NULL; }
cl_int clSetUserEventStatus(void *ev, cl_int s) { typedef cl_int(*fn_t)(void*,cl_int); return g_clSetUserEventStatus ? ((fn_t)g_clSetUserEventStatus)(ev,s) : CL_DEVICE_NOT_FOUND; }

void* clSVMAlloc(void *c, cl_bitfield f, size_t s, cl_uint a) { typedef void*(*fn_t)(void*,cl_bitfield,size_t,cl_uint); return g_clSVMAlloc ? ((fn_t)g_clSVMAlloc)(c,f,s,a) : NULL; }
void clSVMFree(void *c, void *p) { typedef void(*fn_t)(void*,void*); if (g_clSVMFree) ((fn_t)g_clSVMFree)(c,p); }
cl_int clEnqueueSVMMap(void *q, cl_uint bk, cl_bitfield f, void *p, size_t s, cl_uint ne, const void *el, void *ev) { typedef cl_int(*fn_t)(void*,cl_uint,cl_bitfield,void*,size_t,cl_uint,const void*,void*); return g_clEnqueueSVMMap ? ((fn_t)g_clEnqueueSVMMap)(q,bk,f,p,s,ne,el,ev) : CL_DEVICE_NOT_FOUND; }
cl_int clEnqueueSVMUnmap(void *q, void *p, cl_uint ne, const void *el, void *ev) { typedef cl_int(*fn_t)(void*,void*,cl_uint,const void*,void*); return g_clEnqueueSVMUnmap ? ((fn_t)g_clEnqueueSVMUnmap)(q,p,ne,el,ev) : CL_DEVICE_NOT_FOUND; }
cl_int clEnqueueSVMMemcpy(void *q, cl_uint bk, void *dp, const void *sp, size_t s, cl_uint ne, const void *el, void *ev) { typedef cl_int(*fn_t)(void*,cl_uint,void*,const void*,size_t,cl_uint,const void*,void*); return g_clEnqueueSVMMemcpy ? ((fn_t)g_clEnqueueSVMMemcpy)(q,bk,dp,sp,s,ne,el,ev) : CL_DEVICE_NOT_FOUND; }
cl_int clEnqueueSVMMemFill(void *q, void *p, const void *pat, size_t ps, size_t s, cl_uint ne, const void *el, void *ev) { typedef cl_int(*fn_t)(void*,void*,const void*,size_t,size_t,cl_uint,const void*,void*); return g_clEnqueueSVMMemFill ? ((fn_t)g_clEnqueueSVMMemFill)(q,p,pat,ps,s,ne,el,ev) : CL_DEVICE_NOT_FOUND; }
cl_int clSetKernelArgSVMPointer(void *k, cl_uint i, const void *v) { typedef cl_int(*fn_t)(void*,cl_uint,const void*); return g_clSetKernelArgSVMPointer ? ((fn_t)g_clSetKernelArgSVMPointer)(k,i,v) : CL_DEVICE_NOT_FOUND; }
