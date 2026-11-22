println("------------------------------------")
println("|      Imaginary-time Δτ           |")
println("------------------------------------")

# ← QuAPI/test/api/imagtime.jl 移植：去掉 deprecated 对照块与 infinitetime
#   测试集（infinite_Δτ 属未移植模块，测试留在 QuAPI）



@testset "fermionic Δτ" begin
	β = 5
	for f in (semicircular(), DiracDelta(ω=0.5))
		for μ in (-0.3, 0, 0.3)
			corr1 = fermionic_Δτ(f, β=β, N=10, μ=μ)

			corr3 = transpose(corr1)
			for i in 1:size(corr3, 1), j in 1:size(corr3, 2)
				@test branch(corr1)[i, j] == branch(corr3)[j, i]
			end
		end
	end
	f1, f2 = semicircular(), DiracDelta(ω=0.5)
	for μ in (-0.3, 0, 0.3)
		corr1 = fermionic_Δτ(f1, β=β, N=5, μ=μ)
		corr2 = fermionic_Δτ(f2, β=β, N=5, μ=μ)

		corr3 = corr1 + corr2
		for i in 1:size(corr3, 1), j in 1:size(corr3, 2)
			@test branch(corr3)[i, j] ≈ branch(corr1)[i, j] + branch(corr2)[i, j]
		end
	end
end

@testset "bosonic Δτ" begin
	β = 5
	for f in (Leggett(), DiracDelta(ω=0.5))
		for μ in (-0.3, 0)
			corr1 = bosonic_Δτ(f, β=β, N=10, μ=μ)

			corr3 = transpose(corr1)
			for i in 1:size(corr3, 1), j in 1:size(corr3, 2)
				@test branch(corr1)[i, j] == branch(corr3)[j, i]
			end
		end
	end
	f1, f2 = Leggett(), DiracDelta(ω=0.5)

	for μ in (-0.3, 0)
		corr1 = bosonic_Δτ(f1, β=β, N=10, μ=μ)
		corr2 = bosonic_Δτ(f2, β=β, N=10, μ=μ)

		corr3 = corr1 - corr2
		for i in 1:size(corr3, 1), j in 1:size(corr3, 2)
			@test branch(corr3)[i, j] ≈ branch(corr1)[i, j] - branch(corr2)[i, j]
		end
	end
end

@testset "CorrelationMatrix" begin
	A = CorrelationMatrix([1.0+0im, 0.5, 0.2], [0.5+0im, 0.3, 0.1])
	B = CorrelationMatrix([1.0+0im, 0.5, 0.2], [0.5+0im, 0.3, 0.1])
	C = CorrelationMatrix([1.0+0im, 0.5, 0.2], [0.5+0im, 0.3, 0.1+1.0e-9im])
	@test A == B
	@test A != C
	@test A ≈ B
	@test A ≈ C atol=1.0e-8
	@test !isapprox(A, C, atol=1.0e-12)
	@test iszero(A - A)
	@test !iszero(A)
	@test transpose(A)[1, 2] == A[2, 1]
	@test (A + B)[1, 3] ≈ 2*A[1, 3]
	@test (-A)[2, 1] ≈ -A[2, 1]
end

@testset "Δiw_to_Δτ" begin
	β = 5.0
	N = 20
	ω0 = 1.0
	nmax = 2000

	# analytic benchmark: for a single fermionic level at the grand-canonical effective energy ω0-μ the
	# (巨正则极点 ε−μ，教科书约定 1/(iω+μ−ε)；2026-11-20 方向修复——此前离散族
	# 实现为 ω0+μ，与 bounded 族及物理极点相反)
	# Matsubara-domain hybridization Δ(iωₙ) = 1/(iωₙ-(ω0-μ)) reproduces
	# fermionic_Δτ(DiracDelta(ω=ω0); μ) via the doubly-integrated Matsubara
	# sum.  The two storage conventions split the diagonal differently, so the
	# materialized Δ[i,j] matrices are compared.
	for μ in (0.0, 0.3)
		Δiw = [1/(im*ω - (ω0-μ)) for ω in ifrequencies(β, nmax)]
		m1 = Matrix(Δiw_to_Δτ(Δiw, β=β, N=N).data)
		m2 = Matrix(fermionic_Δτ(DiracDelta(ω=ω0), β=β, N=N, μ=μ).data)
		@test m1 ≈ m2 atol=1.0e-8
	end

	# an odd number of Matsubara frequencies is rejected
	@test_throws ArgumentError Δiw_to_Δτ([1.0, 2.0, 3.0], β=β, N=N)

	# zero hybridization gives the zero correlation matrix
	@test iszero(Δiw_to_Δτ(zeros(2), β=β, N=N))

	# homogeneity: scaling the hybridization scales the output linearly
	Δiw = [1/(im*ω - ω0) for ω in ifrequencies(β, nmax)]
	c1 = Δiw_to_Δτ(3 .* Δiw, β=β, N=N)
	c2 = Δiw_to_Δτ(Δiw, β=β, N=N)
	for i in 1:N, j in 1:N
		@test branch(c1)[i, j] ≈ 3*branch(c2)[i, j]
	end

	# a real, even hybridization gives a symmetric Toeplitz matrix
	Δiw = [2ω0/(ω0^2 + ω^2) for ω in ifrequencies(β, nmax)]
	c3 = Δiw_to_Δτ(Δiw, β=β, N=N)
	@test transpose(c3) == c3

	# raw-vector benchmark (Dirac): point-mass spectra are reproduced exactly,
	# equal time included, for either sign of μ.  The diagonal is split 50/50
	# between ηⱼₖ[1] and ηₖⱼ[1] by construction, while fermionic_Δτ splits it
	# into the two triangle integrals, so only lags ≥ 2 and the diagonal sum
	# agree at the raw-vector level (the materialized matrices above already
	# pin the sum).  Pole position ω0−μ（巨正则 ε−μ；2026-11-20 方向修复——`spectrumshift` 此前把
# DiracDelta 移为 ω → ω+μ，与 bounded 族及物理极点相反，见 bounded 块注释）。
	# DiracDelta as ω → ω+μ.
	for μ in (-0.3, 0.0, 0.3)
		Δiw = [1/(im*ω - (ω0-μ)) for ω in ifrequencies(β, nmax)]
		c1 = Δiw_to_Δτ(Δiw, β=β, N=N)
		c2 = fermionic_Δτ(DiracDelta(ω=ω0), β=β, N=N, μ=μ)
		@test c1.data.ηⱼₖ[2:end] ≈ c2.data.ηⱼₖ[2:end] atol=1.0e-8
		@test c1.data.ηₖⱼ[2:end] ≈ c2.data.ηₖⱼ[2:end] atol=1.0e-8
		@test c1.data.ηⱼₖ[1] + c1.data.ηₖⱼ[1] ≈ c2.data.ηⱼₖ[1] + c2.data.ηₖⱼ[1] atol=1.0e-8
	end

	# generic (bounded) spectra: `spectrumshift` moves them the opposite way,
	# f(ε+μ), so the physical Matsubara pole sits at ε−μ and the values are
	# the Hilbert transform at z = iω+μ.  Δ(iωₙ) below is the closed form for
	# the semicircular density (2/π)√(1−ε²) on [−1,1], branch fixed by
	# Im√z = Im z.  With it, the full kernel — equal time included — matches
	# fermionic_Δτ to truncation accuracy for either sign of μ.
	sc_mw(ω, μ) = begin
		z = im*ω + μ
		s = sqrt(complex(z)^2 - 1)
		sign(imag(s)) == sign(ω) || (s = -s)
		2*(z - s)
	end
	fsc = semicircular()
	for μ in (-0.3, 0.0, 0.3)
		Δiw = [sc_mw(ω, μ) for ω in ifrequencies(β, nmax)]
		c1 = Δiw_to_Δτ(Δiw, β=β, N=N)
		c2 = fermionic_Δτ(fsc, β=β, N=N, μ=μ)
		@test c1.data.ηⱼₖ[2:end] ≈ c2.data.ηⱼₖ[2:end] atol=1.0e-8
		@test c1.data.ηₖⱼ[2:end] ≈ c2.data.ηₖⱼ[2:end] atol=1.0e-8
		@test c1.data.ηⱼₖ[1] + c1.data.ηₖⱼ[1] ≈ c2.data.ηⱼₖ[1] + c2.data.ηₖⱼ[1] atol=1.0e-8
	end

	# a hybridization that is not conjugate-symmetric, Δ(−iω) ≠ conj(Δ(iω)),
	# cannot give a real Δτ kernel and is rejected
	ωs = ifrequencies(β, div(nmax, 10))
	Δiw = zeros(ComplexF64, length(ωs))
	Δiw[length(ωs)÷2+1:end] .= [1/(im*ω - ω0) for ω in ωs[length(ωs)÷2+1:end]]
	@test_throws ErrorException Δiw_to_Δτ(Δiw, β=β, N=N)

	# a single time slice keeps only the equal-time element
	Δiw = [1/(im*ω - (ω0-0.3)) for ω in ifrequencies(β, nmax)]
	c1 = Δiw_to_Δτ(Δiw, β=β, N=1)
	c2 = fermionic_Δτ(DiracDelta(ω=ω0), β=β, N=1, μ=0.3)
	@test (c1.data.ηⱼₖ[1] + c1.data.ηₖⱼ[1]) ≈ (c2.data.ηⱼₖ[1] + c2.data.ηₖⱼ[1]) atol=1.0e-8
end

# ---------------------------------------------------------------------------
# μ 折叠契约守卫（2026-11-20）：spectrumshift = 化学势吸收（参数平移 f(ε+μ)，
# 支撑集 ω − μ）。契约恒等式：Δ(f, β, N, μ) ≡ Δ(shift(f, μ), β, N, 0)。
# 离散族（DiracDelta/DiscreteSpectrum）曾以几何平移 ω+μ 实现（方向相反——有限
# μ 的离散 bath 核全体错；μ=0 时 no-op 不可见，结构断言（转置/可加）不设防）。
# 此 testset 固化修复后的方向；fhand 一律手工构造（不经 spectrumshift 原语，
# 防止以 primitive 自证）。
# ---------------------------------------------------------------------------

@testset "spectrumshift μ 吸收契约（Δτ: Δ(f,μ) ≡ Δ(shift(f,μ),0)）" begin
	β = 5; μ = 0.3
	# DiracDelta / DiscreteSpectrum（修复焦点；逐位）
	for (f, fhand) in ((DiracDelta(ω=0.5), DiracDelta(ω=0.5 - μ)),
	                   (DiscreteSpectrum([0.3, 0.9], [0.5, 0.2]),
	                    DiscreteSpectrum([0.3 - μ, 0.9 - μ], [0.5, 0.2])))
		@test branch(fermionic_Δτ(f, β=β, N=10, μ=μ)) ==
		      branch(fermionic_Δτ(fhand, β=β, N=10, μ=0))
		@test branch(bosonic_Δτ(f, β=β, N=10, μ=μ)) ==
		      branch(bosonic_Δτ(fhand, β=β, N=10, μ=0))
	end
	# bounded（generic 语义锚；未受 bug 影响，防未来反向）
	f = semicircular()
	fhand = bounded(ε -> f.f(ε + μ), lowerbound(f) - μ, upperbound(f) - μ)
	@test branch(fermionic_Δτ(f, β=β, N=10, μ=μ)) ==
	      branch(fermionic_Δτ(fhand, β=β, N=10, μ=0))
end
