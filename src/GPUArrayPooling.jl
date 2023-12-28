module GPUArrayPooling

using GPUArrays: DataRef, unsafe_free!

function wrap_buffer! end
function empty_pool! end

struct BufferPool{B}
    bufs::Vector{DataRef{B}}
end

function Base.empty!(bp::BufferPool)
    foreach(Base.Fix2(unsafe_free!, true), bp.bufs)
    empty!(bp.bufs)
    return bp
end

function Base.push!(bp::BufferPool, dref::DataRef)
    push!(bp.bufs, dref)
    return bp
end

end
