x = collect(linspace(-2π, 2π, 100))
y = sin.(x)
df = zeros(100)
df_ref = cos.(x)

@test maximum(abs.(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:central}) - df_ref)) < 1e-8
@test maximum(abs.(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:forward}) - df_ref)) < 1e-4
@test maximum(abs.(DiffEqDiffTools.finite_difference!(df, sin, x, Val{:forward}, y) - df_ref)) < 1e-4
