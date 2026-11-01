println("------------------------------------")
println("|        Real-time Δt              |")
println("------------------------------------")

# ← QuAPI/test/api/realtime.jl 移植：去掉 deprecated 对照块与 infinitetime
#   测试集（infinite_Δt 属未移植模块，测试留在 QuAPI）；QuAPI. 限定名 → IFD.


@testset "fermionic Δt" begin
	β = 5
	for f in (semicircular(), DiracDelta(ω=0.5))
		for μ in (-0.3, 0, 0.3)
			x = fermionic_Δt(f, β=β, t=2, N=10, μ=μ)

			z = transpose(x)
			for i in 1:size(z, 1), b1 in (:+, :-)
				for j in 1:size(z, 2), b2 in (:+, :-)
					@test branch(x; b1=b1, b2=b2)[i, j] == branch(z; b1=b2, b2=b1)[j, i]
				end
			end
		end
	end

	f1, f2 = semicircular(), DiracDelta(ω=0.5)

	for μ in (-0.3, 0, 0.4)
		x = fermionic_Δt(f1, β=β, t=2, N=10, μ=μ)
		y = fermionic_Δt(f2, β=β, t=2, N=10, μ=μ)

		z = x + y
		for i in 1:size(z, 1), b1 in (:+, :-)
			for j in 1:size(z, 2), b2 in (:+, :-)
				@test branch(z; b1=b1, b2=b2)[i, j] ≈ branch(x; b1=b1, b2=b2)[i, j] + branch(y; b1=b1, b2=b2)[i, j]
			end
		end
	end
end

@testset "cross-block sharing at μ ≠ 0 (unified convention vs historical formulas)" begin
	# The two-matrix diagonal storage derives the cross blocks as pure sign
	# combinations of the stored halves.  This is structural (the historical
	# four-block construction shared literally identical quadratures between
	# blocks), NOT a thermal identity — but pin it at μ ≠ 0 and for a
	# complex-valued spectrum against independently recomputed historical
	# formulas, since the μ = 0 checks elsewhere do not cover it.
	β, t, N = 2.0, 1.0, 6
	δt = t / N
	for μ in (-0.3, 0.3)
		for (tag, f) in (("real", semicircular()),
		                 ("complex", IFD.get_all_fs(semicircular(), 0.2 + 0.45im)[2]))
			x = fermionic_Δt(f; β=β, N=N, t=t, μ=μ)
			gpm = branch(x, :+, :-); gmp = branch(x, :-, :+)
			g₁ = ε -> fermidirac(β, μ - ε); g₂ = ε -> fermidirac(β, ε - μ)
			fk(d) = IFD._fⱼₖ_r_disperse(f, identity, d, δt)
			fjj = IFD._fⱼⱼ_r_disperse(f, identity, δt)
			fkk = IFD._fₖₖ_r_disperse(f, identity, δt)
			hist_c = [quadgkwrapper(fjj * g₂)]; hist_b = [quadgkwrapper(fkk * g₂)]
			hist_a = [quadgkwrapper(-fjj * g₁)]; hist_d = [quadgkwrapper(-fkk * g₁)]
			for i in 1:N
				push!(hist_c, quadgkwrapper(fk(i) * g₂));  push!(hist_b, quadgkwrapper(fk(-i) * g₂))
				push!(hist_a, quadgkwrapper(-fk(i) * g₁)); push!(hist_d, quadgkwrapper(-fk(-i) * g₁))
			end
			@test gpm.ηⱼₖ ≈ -hist_c atol=1.0e-12
			@test gpm.ηₖⱼ ≈ -hist_b atol=1.0e-12
			@test gmp.ηⱼₖ ≈ -hist_a atol=1.0e-12
			@test gmp.ηₖⱼ ≈ -hist_d atol=1.0e-12
		end
	end
end

@testset "RealCorrelationFunction single-parametric type & general-matrix storage" begin
	# single type parameter: both diagonal blocks share one matrix type
	x = fermionic_Δt(semicircular(); β=2.0, N=6, t=1.0)
	@test x isa RealCorrelationFunction{CorrelationMatrix{ComplexF64}}

	# two-matrix constructor with dense matrices (cross pair): cross branches
	# return the stored blocks; diagonal branches derived by the sector rule
	# through _G₊₊/_G₋₋ — triangles exact (bitwise vs the Toeplitz derivation)
	d = RealCorrelationFunction(Matrix(branch(x, :+, :-)), Matrix(branch(x, :-, :+)))
	@test branch(d, :+, :-) == Matrix(branch(x, :+, :-))
	@test branch(d, :-, :+) == Matrix(branch(x, :-, :+))
	for (b1, b2) in ((:+, :+), (:-, :-))
		mg = branch(d, b1, b2); mt = branch(x, b1, b2)
		n = size(mg, 1)
		@test maximum(abs.(mg[i,j] - mt[i,j]) for i in 1:n, j in 1:n if i != j) == 0
	end

	# four-matrix CorrelationMatrix constructor: consistent cross blocks are
	# accepted (and reduced); inconsistent ones throw
	ok4 = RealCorrelationFunction(branch(x, :+, :+), branch(x, :+, :-),
	                              branch(x, :-, :+), branch(x, :-, :-))
	@test ok4 == x
	# branch returns the stored cross block itself — deep-copy before corrupting
	bad = deepcopy(branch(x, :+, :-)); bad.ηⱼₖ .*= 2
	@test_throws ArgumentError RealCorrelationFunction(branch(x, :+, :+), bad,
	                                                   branch(x, :-, :+), branch(x, :-, :-))
	# four general (non-Toeplitz) matrices: sector-consistent input is
	# accepted (zeros are trivially consistent); random blocks throw
	Z = zeros(ComplexF64, 7, 7)
	@test RealCorrelationFunction(Z, Z, Z, Z) isa RealCorrelationFunction{Matrix{ComplexF64}}
	R = rand(ComplexF64, 7, 7)
	@test_throws ArgumentError RealCorrelationFunction(R, R, R, R)
end


@testset "bosonic Δt" begin
	β = 5
	for f in (Leggett(), DiracDelta(ω=0.5))
		for μ in (-0.3, 0)
			x = bosonic_Δt(f, β=β, t=2, N=5, μ=μ)

			z = transpose(x)
			for i in 1:size(z, 1), b1 in (:+, :-)
				for j in 1:size(z, 2), b2 in (:+, :-)
					@test branch(x; b1=b1, b2=b2)[i, j] == branch(z; b1=b2, b2=b1)[j, i]
				end
			end
		end
	end

	f1, f2 = Leggett(), DiracDelta(ω=0.5)

	for μ in (-0.3, 0)
		x = bosonic_Δt(f1, β=β, t=2, N=5, μ=μ)
		y = bosonic_Δt(f2, β=β, t=2, N=5, μ=μ)

		z = x - y
		for i in 1:size(z, 1), b1 in (:+, :-)
			for j in 1:size(z, 2), b2 in (:+, :-)
				@test branch(z; b1=b1, b2=b2)[i, j] ≈ branch(x; b1=b1, b2=b2)[i, j] - branch(y; b1=b1, b2=b2)[i, j]
			end
		end
	end
end
