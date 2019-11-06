using LinearTransforms, Arpack
using SparseArrays, LinearAlgebra
using Test

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

        # test size
        @test size(op) == (nnz(M), size(M)) 

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

        # test size
        @test size(op) == (nnz(M), size(M))

    end
end

