using Statistics
using Printf
using Flux
using Plots

# Flux uses Float32 by default so let's just use that!
const FT = Float32

function generate_training_data(f; N, domain)
    x = range(domain[1], domain[2], length=N)
    return @. FT(x), FT(f(x))
end

function generate_neural_network(; layers, nodes=1, activation=identity)
    return Chain([Dense(l == 1 ? 1 : nodes, l == layers ? 1 : nodes, activation) for l in 1:layers]...)
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

function test_learned_mapping(f, NN; N, domain)
    xs, ys_truth = generate_training_data(f; N, domain)
    ys_NN = [NN([x])[1] for x in xs]

    kwargs = (linewidth=3, linealpha=0.8, xlims=extrema(xs),
              grid=false, legend=:bottomright, framestyle=:box,
              foreground_color_legend=nothing, background_color_legend=nothing)

    fit_plot = plot(xs, ys_truth, label="truth", title="Fit", xlabel="x", ylabel="y"; kwargs...)
    plot!(fit_plot, xs, ys_NN, label="neural network"; kwargs...)

    loss = @. abs(ys_NN - ys_truth) / abs(ys_truth)
    ylims = (10^floor(log10(minimum(loss))), 10^ceil(log10(maximum(loss))))
    loss_plot = plot(xs, loss, label="", title="extrapolation accuracy", xlabel="x", ylabel="|y′ - y| / y", yaxis=:log, ylims=ylims; kwargs...)

    return plot(fit_plot, loss_plot, layout=(2, 1))
end

linear(x) = x
NN = generate_neural_network(layers=1)
learn_scalar_function!(NN, linear, N=10, domain=(-1, 1), epochs=20, optimizers=[Descent()])
test_learned_mapping(linear, NN, N=100, domain=(-100, 100))

quadratic(x) = x^2
NN = generate_neural_network(layers=20, nodes=20)
learn_scalar_function!(NN, quadratic, N=10, domain=(0, 1), epochs=10, optimizers=[ADAM()])
test_learned_mapping(quadratic, NN, N=100, domain=(0, 2))
