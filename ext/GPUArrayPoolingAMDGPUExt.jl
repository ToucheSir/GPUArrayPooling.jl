module GPUArrayPoolingAMDGPUExt

using GPUArrays: DataRef
using AMDGPU: AMDGPU, HIP, HIPDevice, ROCArray, ROCDeviceArray
using AMDGPU.Mem: Mem, HIPBuffer, HostBuffer
using GPUArrayPooling: GPUArrayPooling, BufferPool

struct WrappedAMDBuffer <: Mem.AbstractAMDBuffer
    inner::HIPBuffer
end

maybe_unwrap(@nospecialize(x)) = x
maybe_unwrap(buf::WrappedAMDBuffer) = buf.inner

# TODO is this required?
Base.convert(::Type{HIPBuffer}, buf::WrappedAMDBuffer) = buf.inner

# Forwarded methods

@inline Base.unsafe_convert(t::Type{<:Ptr}, buf::WrappedAMDBuffer) =
    Base.unsafe_convert(t, buf.inner)

@inline Mem.view(buf::WrappedAMDBuffer, bytesize::Int) = Mem.view(buf.inner, bytesize)

@inline Mem.free(buf::WrappedAMDBuffer; stream::HIP.HIPStream) = Mem.free(buf.inner; stream)

@inline Mem.upload!(dst::WrappedAMDBuffer, src::Ptr, bytesize::Int; stream::HIP.HIPStream) =
    Mem.upload!(dst.inner, src, bytesize; stream)

@inline Mem.download!(
    dst::Ptr, src::WrappedAMDBuffer, bytesize::Int; stream::HIP.HIPStream, async::Bool
) = Mem.download!(dst, src.inner, bytesize; stream, async)

@inline Mem.transfer!(
    dst::Union{WrappedAMDBuffer,HIPBuffer,HostBuffer},
    src::Union{WrappedAMDBuffer,HIPBuffer,HostBuffer},
    bytesize::Int;
    stream::HIP.HIPStream,
) = Mem.transfer!(maybe_unwrap(dst), maybe_unwrap(src), bytesize; stream)

@inline Mem.upload!(
    dst::HostBuffer, src::WrappedAMDBuffer, sz::Int; stream::HIP.HIPStream
) = Mem.upload!(dst, src.inner, sz; stream)

@inline Mem.download!(
    dst::WrappedAMDBuffer, src::HostBuffer, sz::Int; stream::HIP.HIPStream, async::Bool
) = Mem.download!(dst.inner, src, sz; stream, async)

# Required to fool @roc conversion logic
function Base.convert(t::Type{<:ROCDeviceArray}, a::ROCArray{<:Any,<:Any,WrappedAMDBuffer})
    tmp = a.buf.rc.obj
    a.buf.rc.obj = tmp.inner
    da = convert(t, a)
    a.buf.rc.obj = tmp
    return da
end

# This is where the magic happens.
# By intercepting calls to `similar`, we can record array allocations without modifying the running program.

function Base.similar(a::ROCArray{<:Any,<:Any,WrappedAMDBuffer})
    ret = @invoke similar(a::ROCArray{<:Any,<:Any})
    GPUArrayPooling.wrap_buffer!(ret)
    return ret
end

function Base.similar(a::ROCArray{<:Any,<:Any,WrappedAMDBuffer}, dims::Base.Dims)
    ret = @invoke similar(a::ROCArray{<:Any}, dims::Base.Dims)
    GPUArrayPooling.wrap_buffer!(ret)
    return ret
end

function Base.similar(a::ROCArray, t::Type, dims::Base.Dims)
    ret = @invoke similar(a::ROCArray, t::Type, dims::Base.Dims)
    GPUArrayPooling.wrap_buffer!(ret)
    return ret
end

# Pool management

const device_pools = Dict{HIPDevice,BufferPool{HIPBuffer}}()
const pool_lock = ReentrantLock()

function GPUArrayPooling.wrap_buffer!(arr::ROCArray{<:Any,<:Any,HIPBuffer})
    newbuf = WrappedAMDBuffer(arr.buf.rc.obj)
    arr.buf.rc.obj = newbuf
    @lock pool_lock begin
        pool = get(device_pools, AMDGPU.device(arr), nothing)
        pool === nothing && @error "No GPU array pool found for device $dev"
        push!(pool, arr.buf)
    end
end

function GPUArrayPooling.empty_pool!(dev::HIPDevice)
    @lock pool_lock begin
        pool = get(device_pools, dev, nothing)
        pool === nothing && @warn "No GPU array pool found for device $dev"
        empty!(pool)
    end
end

function __init__()
    AMDGPU.functional() || return nothing
    @lock pool_lock begin
        for dev in AMDGPU.devices()
            device_pools[dev] = BufferPool{WrappedAMDBuffer}()
        end
    end
end

end