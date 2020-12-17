using Statistics
using Printf
using Flux

# Flux uses Float32 by default so let's just use that!
const FT = Float32

function generate_training_data(f; N, domain)
    x = range(domain[1], domain[2], length=N)
    return @. FT(x), FT(f(x))
end

function generate_neural_network(; layers, activation=identity)
    return Chain([Dense(1, 1, activation) for _ in 1:layers]...)
end

function learn_scalar_function!(NN, f; N, domain, epochs, optimizers=[ADAM()])
    xs, ys = generate_training_data(f; N, domain)
    training_data = zip(xs, ys) |> collect

    loss(x, y) = Flux.mse(NN([x]), y)
    function callback()
        μ_loss = mean(loss(d...) for d in training_data)
        @info @sprintf("mean squared error loss = %.16e", μ_loss)
        return μ_loss
    end

    for opt in optimizers, e in 1:epochs
        @info "Training with $(typeof(opt))(η=$(opt.eta)) [epoch $e/$epochs]..."
        Flux.train!(loss, Flux.params(NN), training_data, opt, cb=callback)
    end

    return NN
end

linear(x) = x
NN = generate_neural_network(layers=1)
learn_scalar_function!(NN, linear, N=10, domain=(-1, 1), epochs=20, optimizers=[Descent()])
