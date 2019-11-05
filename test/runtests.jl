using LinearTransforms
using SparseArrays, LinearAlgebra
using Test

@testset "opMatrix" begin
    m, n = 3, 5
    A = randn(m,n) + randn(m,n)im
    op = opMatrix(A)

    @test dottest(op)
    @test eltype(op) == eltype(A)

    A = randn(m,n)
    op = opMatrix(A)
    @test eltype(op) == eltype(A)

    x = drandn(op)
    w1 = op*x
    w2 = A*x
    w3 = op''*x
    @test norm(w1-w2) ≤ eps(1.)
    @test norm(w1-w3) ≤ eps(1.)

    y = randn(m)
    v1 = op'*y
    v2 = A'*y
    @test norm(v1-v2) ≤ eps(1.)
    @test isequal(-op*x, -A*x)
    @test dottest(op)

end

@testset "opMask" begin

    m, n = 123, 321
    M = sprand(Bool,m,n,0.5)
    I, J = findnz(M)[1:2]
    u, v = randn(m,2), randn(n,2)
    X = u*v'

    @testset "full matrices" begin
        op = opMask(M)

        # forward operator
        w = op*X
        wt = [X[i,j] for (i,j) in zip(I,J)]
        @test norm(w-wt, Inf) ≤ 1e-10*(1+norm(wt,Inf))

        # adjoint operator
        V = op'*w
        Vt = sparse(I, J, w, m, n)
        @test norm(V-Vt) ≤ 1e-10

        # dot test
        @test dot(op*X,w) ≈ dot(X,op'*w)

        # test eltypes
        op = opMask(M)
        @test eltype(op) == Float64
        op = opMask{Complex{Float64}}(M)
        @test eltype(op) == Complex{Float64}
    end

    @testset "factored matrices" begin
        op = opMaskFactored(M)

        # forward operator
        w = op*(u,v)
        wT = [X[i,j] for (i,j) in zip(I,J)]
        @test norm(w-wT, Inf) ≤ 1e-10*(1+norm(wT,Inf))

        # adjoint operator
        V = op'*w
        Vt = sparse(I, J, w, m, n)
        @test norm(V-Vt) ≤ 1e-10

        # dot test
        @test dot(op*(u,v),w) ≈ dot(u*v',op'*w)

        # eltypes
        op = opMaskFactored(M)
        @test eltype(op) == Float64
        op = opMaskFactored{Int64}(M)
        @test eltype(op) == Int64
    end
end

@testset "opProductForm" begin

    m, n, r = 10, 30, 3
    R = rand(m,r)
    L = rand(n,r)
    x, y = rand(n), rand(m)
    op = opProductForm(R,L)
    @test all( R*(L'x) .≈ op*x )
    @test all( L*(R'y) .≈ op'*y)
    @test dottest(op)
end

@testset "element type" begin
    m,n = 3,5
    A = randn(m,n) + randn(m,n)im
    op = opMatrix(A)
    @test eltype(op) == Complex{Float64}
end

