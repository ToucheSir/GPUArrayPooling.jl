module GPUArrayPoolingAMDGPUExt

using GPUArrays: DataRef, RefCounted, retain
using AMDGPU: AMDGPU, HIP, HIPDevice, ROCArray, ROCArrayStyle, ROCDeviceArray
using AMDGPU.Mem: Mem, HIPBuffer, HostBuffer
using GPUArrayPooling: GPUArrayPooling, BufferPool

struct WrappedAMDBuffer <: Mem.AbstractAMDBuffer
    inner::HIPBuffer
end

function WrappedAMDBuffer(bytesize::Int; stream::HIP.HIPStream)
    return WrappedAMDBuffer(HIPBuffer(bytesize; stream))
end

unwrap(@nospecialize(x)) = x
unwrap(buf::WrappedAMDBuffer) = getfield(buf, :inner)

@inline Base.getproperty(buf::WrappedAMDBuffer, f::Symbol) = getproperty(unwrap(buf), f)

# TODO is this required?
# Base.convert(::Type{HIPBuffer}, buf::WrappedAMDBuffer) = unwrap(buf)
function Base.Broadcast.BroadcastStyle(
    ::ROCArrayStyle{M,<:Union{HIPBuffer,WrappedAMDBuffer}},
    ::ROCArrayStyle{N,<:Union{HIPBuffer,WrappedAMDBuffer}},
) where {M,N}
    return ROCArrayStyle{max(M,N),WrappedAMDBuffer}()
end

# Forwarded methods

@inline Base.unsafe_convert(t::Type{<:Ptr}, buf::WrappedAMDBuffer) =
    Base.unsafe_convert(t, unwrap(buf))

@inline Base.unsafe_convert(
    ::Type{Ptr{T}}, x::ROCArray{T,<:Any,WrappedAMDBuffer}
) where {T} = Base.unsafe_convert(Ptr{T}, unwrap(x.buf[])) + x.offset * sizeof(T)

function Base.convert(
    ::Type{ROCDeviceArray{T,N,AMDGPU.AS.Global}}, a::ROCArray{T,N,WrappedAMDBuffer}
) where {T,N}
    ptr = Base.unsafe_convert(Ptr{T}, unwrap(a.buf[]))
    llvm_ptr = AMDGPU.LLVMPtr{T,AMDGPU.AS.Global}(ptr + a.offset * sizeof(T))
    return ROCDeviceArray{T,N,AMDGPU.AS.Global}(a.dims, llvm_ptr)
end

@inline Mem.view(buf::WrappedAMDBuffer, bytesize::Int) = Mem.view(unwrap(buf), bytesize)

@inline Mem.free(buf::WrappedAMDBuffer; stream::HIP.HIPStream) =
    Mem.free(unwrap(buf); stream)

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
) = Mem.transfer!(unwrap(dst), unwrap(src), bytesize; stream)

@inline Mem.upload!(
    dst::HostBuffer, src::WrappedAMDBuffer, sz::Int; stream::HIP.HIPStream
) = Mem.upload!(dst, src.inner, sz; stream)

@inline Mem.download!(
    dst::WrappedAMDBuffer, src::HostBuffer, sz::Int; stream::HIP.HIPStream, async::Bool
) = Mem.download!(dst.inner, src, sz; stream, async)

# This is where the magic happens.
# By intercepting calls to `similar`, we can record array allocations without modifying the running program.

function Base.similar(a::ROCArray{T,N,WrappedAMDBuffer}) where {T,N}
    return register_array(ROCArray{T,N,WrappedAMDBuffer}(undef, size(a)))
end

function Base.similar(::ROCArray{T,<:Any,WrappedAMDBuffer}, dims::Base.Dims{N}) where {T,N}
    return register_array(ROCArray{T,N,WrappedAMDBuffer}(undef, dims))
end

function Base.similar(
    ::ROCArray{<:Any,<:Any,WrappedAMDBuffer}, ::Type{T}, dims::Base.Dims{N}
) where {T,N}
    return register_array(ROCArray{T,N,WrappedAMDBuffer}(undef, dims))
end

# Pool management

const device_pools = Dict{HIPDevice,BufferPool{WrappedAMDBuffer}}()
const pool_lock = ReentrantLock()

function register_array(a::ROCArray{<:Any,<:Any,WrappedAMDBuffer})
    @lock pool_lock begin
        dev = AMDGPU.device(a)
        pool = get(device_pools, dev, nothing)
        pool === nothing && @error "No GPU array pool found for device $dev"
        push!(pool, a.buf)
    end
    return a
end

function GPUArrayPooling.wrap_buffer(a::ROCArray{T,N,HIPBuffer}) where {T,N}
    rc = a.buf.rc
    retain(rc) # ensure the buffer isn't freed from under us
    newbuf = DataRef(rc.finalizer, WrappedAMDBuffer(rc.obj))
    wrapped = ROCArray{T,N}(newbuf, a.dims; offset=a.offset)
    return register_array(wrapped)
end

function GPUArrayPooling.empty_pool!(dev::HIPDevice)
    @lock pool_lock begin
        pool = get(device_pools, dev, nothing)
        pool === nothing && @warn "No GPU array pool found for device $dev"
        empty!(pool)
    end
end

function GPUArrayPooling.pool_info(dev::HIPDevice)
    @lock pool_lock begin
        pool = get(device_pools, dev, nothing)
        pool === nothing && @warn "No GPU array pool found for device $dev"
        num_arrays = length(pool.bufs)
        num_bytes = Base.format_bytes(sum(b -> b[].bytesize, pool.bufs))
        println("""
        Pool for $dev:
        # of arrays: $num_arrays
        size in bytes: $num_bytes
        """)
    end
end

function __init__()
    AMDGPU.functional() || return nothing
    @lock pool_lock begin
        for dev in AMDGPU.devices()
            device_pools[dev] = BufferPool{WrappedAMDBuffer}(WrappedAMDBuffer[])
        end
    end
end

end