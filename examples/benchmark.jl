using LinearTransforms, BenchmarkTools


m, n = 100, 100
A = randn(m, n)
prod = v -> A*v
tprod = w -> A'*w
op = LinearTransform(m, n, prod, tprod)
x = randn(n)
y = randn(m)

println("Forward native multiply")
@benchmark A*x
println("Forward operator multiply")
@benchmark op*x

println("Ajoint native multiply")
@benchmark A'*y
println("Adjoint operator multiply")
@benchmark op'*y
