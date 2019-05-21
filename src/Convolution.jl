module Convolution

export restrict, discrete_convolve, convolve

struct ContinuousSignal{Func, Time <: Real}
    expr :: Func
    starting_time :: Time
    duration :: Time
end

function restrict(f :: Func, interval :: Tuple{Time, Time}) :: ContinuousSignal{Func, Time} where {Func, Time}
    @assert interval[1] <= interval[2]
    ContinuousSignal(f, interval[1], interval[2] - interval[1])
end

function (sg :: ContinuousSignal{Func, Time})(time :: Real) :: Number where {Func, Time}
    # Not bounds-checking for now
    sg.expr(Time(time))
end

struct DACSignal{T <: Number, Time <: Real, V <: AbstractVector{T}}
    v :: V
    starting_time :: Time
    dt :: Time
end

function (sg :: DACSignal{T, Time, V})(time :: Real) :: Number where {T, Time, V}
    # Reconstruct impulse train
    sample_coords = (0 : length(sg.v) - 1) .* sg.dt .+ sg.starting_time

    # Use sinc interpolation to reconstruct continuous signal
    # Julia's `sinc` is already normalized
    weights = (sample_coords .- time) .|> t -> sinc(t / sg.dt)
    sum(weights .* sg.v)
end

# Convolve 2 raw discrete signals.  The signals are treated as if they start at N = 0.
function discrete_convolve(f :: AbstractVector{T}, h :: AbstractVector{T}) :: Vector{T} where T <: Number
    result :: Vector{T} = zeros(T, length(f) + length(h) - 1)
    for i = 1 : length(f)
        for j = 1 : length(h)
            result[i + j - 1] += f[i] * h[j]
        end
    end
    result
end

# Convolve 2 continuous signals.  dt specifies sampling period (reciprocal sampling frequency).
function convolve(
    dt :: Time,
    f :: ContinuousSignal{Func1, Time},
    h :: ContinuousSignal{Func2, Time}
    ) :: DACSignal where {Func1, Func2, Time}

    # Generate impulse trains
    f_sample_coords = f.starting_time : dt : (f.starting_time + f.duration)
    h_sample_coords = h.starting_time : dt : (h.starting_time + h.duration)

    # Sample
    f_samples = f.(f_sample_coords)
    h_samples = h.(h_sample_coords)

    # Use discrete convolution in place of continuous convolution
    DACSignal(discrete_convolve(f_samples, h_samples) .* dt, f.starting_time + h.starting_time, dt)
end

end # module
