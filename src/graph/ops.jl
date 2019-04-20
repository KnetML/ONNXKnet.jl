# This file contains the implementation of various operators.
# Tests for them is at test/runtests.jl.

using Base
using Statistics
# TODO: we need kwarg support for many of these

# Generic
get_tuple(x) = (x...,)
get_tuple() = nothing
convert_type(x) = Base.convert(Array{Float32, 1}, x)

ops[:Reshape] = function(params, tensor1, shape...)
  if haskey(params, :shape)
    return vcall(:reshape, tensor1, vcall(:broadcast, Int64, vcall(:Tuple, params[:shape])))
  end
  vcall(:reshape, tensor1, vcall(:broadcast, Int64, vcall(:Tuple, vcall(:reverse, shape[1]))))
end

ops[:Conv] = function(params, x, w, b...)
  if (haskey(params, Symbol("auto_pad")))
    if (String(params[:auto_pad]) == "SAME_UPPER" || String(params[:auto_pad] == "SAME_LOWER"))
      temp = Base.convert(Array{Int64,1}, (params[:kernel_shape] .- 1)./2) # Only for strides = [1,1]
      params[:pads] = vcat(temp, temp)                                    # To Do: Add support for other stride values.                                                                           
    elseif String(params[:auto_pad]) == "VALID"
      params[:pads] = [0,0,0,0]
    end
  end
  if length(params[:pads]) == 4
    params[:pads] = params[:pads][1:2]
  end
    return vcall(:conv4, w, x, Symbol("padding=$(params[:pads])"), Symbol("mode = 1"))
end

ops[:Add] = function(params, A, B)
  s1 = vcall(:size, A)
  s2 = vcall(:size, B)
  if (s1==s2)
    return vcall(:Add, params[:axis], A, B)
  else
    return vcall(:.+, A, B)
  end
end

ops[:Relu] = function(params, x)
  vcall(broadcast, :relu, x)
end

ops[:MaxPool] = function(params, x)
  return vcall(:pool, x, Symbol("mode=0"), Symbol("window=$(params[:kernel_shape])"))
end

ops[:MatMul] = function(params, A, B)
  #tempa = vcall(:permutedims, A, vcall(:reverse, vcall(:range, 1, vcall(:ndims, A))))
  #tempb = vcall(:permutedims, B, vcall(:reverse, vcall(:range, 1, vcall(:ndims, B))))
  vcall(:*, B, A)
end