import common 

open nat int real complex set filter matrix metric
open function polynomial mv_polynomial linear_map subgroup ideal ring_hom submodule
open finite_dimensional inner_product_space topological_space
open fintype char_p module zsqrtd  gaussian_int mul_aut interval_integral
open module.End nat.arithmetic_function

open_locale topology big_operators complex_conjugate filter ennreal pointwise classical real

noncomputable theory

theorem exercise_2_2_9 {G : Type*} [group G] {a b : G}
  (h : a * b = b * a) :
  ∀ x y : closure {x | x = a ∨ x = b}, x*y = y*x :=
sorry

theorem exercise_2_3_2 {G : Type*} [group G] (a b : G) :
  ∃ g : G, b* a = g * a * b * g⁻¹ :=
sorry

theorem exercise_2_4_19 {G : Type*} [group G] {x : G}
  (hx : order_of x = 2) (hx1 : ∀ y, order_of y = 2 → y = x) :
  x ∈ center G :=
sorry

theorem exercise_2_8_6 {G H : Type*} [group G] [group H] :
  center (G × H) ≃* (center G) × (center H) :=
sorry

theorem exercise_2_11_3 {G : Type*} [group G] [fintype G]
  (hG : even (card G)) : ∃ x : G, order_of x = 2 :=
sorry

theorem exercise_3_2_7 {F : Type*} [field F] {G : Type*} [field G]
  (φ : F →+* G) : injective φ :=
sorry

theorem exercise_3_5_6 {K V : Type*} [field K] [add_comm_group V]
  [module K V] {S : set V} (hS : set.countable S)
  (hS1 : span K S = ⊤) {ι : Type*} (R : ι → V)
  (hR : linear_independent K R) : countable ι :=
sorry

theorem exercise_3_7_2 {K V : Type*} [field K] [add_comm_group V]
  [module K V] {ι : Type*} [fintype ι] (γ : ι → submodule K V)
  (h : ∀ i : ι, γ i ≠ ⊤) :
  (⋂ (i : ι), (γ i : set V)) ≠ ⊤ :=
sorry

theorem exercise_6_1_14 (G : Type*) [group G]
  (hG : is_cyclic $ G ⧸ (center G)) :
  center G = ⊤  :=
sorry

theorem exercise_6_4_2 {G : Type*} [group G] [fintype G] {p q : ℕ}
  (hp : prime p) (hq : prime q) (hG : card G = p*q) :
  is_simple_group G → false :=
sorry

theorem exercise_6_4_3 {G : Type*} [group G] [fintype G] {p q : ℕ}
  (hp : prime p) (hq : prime q) (hG : card G = p^2 *q) :
  is_simple_group G → false :=
sorry

theorem exercise_6_4_12 {G : Type*} [group G] [fintype G]
  (hG : card G = 224) :
  is_simple_group G → false :=
sorry

theorem exercise_6_8_1 {G : Type*} [group G]
  (a b : G) : closure ({a, b} : set G) = closure {b*a*b^2, b*a*b^3} :=
sorry

theorem exercise_10_1_13 {R : Type*} [ring R] {x : R}
  (hx : is_nilpotent x) : is_unit (1 + x) :=
sorry

theorem exercise_10_2_4 :
  span ({2} : set $ polynomial ℤ) ⊓ (span {X}) =
  span ({2 * X} : set $ polynomial ℤ) :=
sorry

theorem exercise_10_6_7 {I : ideal gaussian_int}
  (hI : I ≠ ⊥) : ∃ (z : I), z ≠ 0 ∧ (z : gaussian_int).im = 0 :=
sorry

theorem exercise_10_4_6 {R : Type*} [comm_ring R] 
  [no_zero_divisors R] {I J : ideal R} (x : I ⊓ J) : 
  is_nilpotent ((ideal.quotient.mk (I*J)) x) :=
sorry

theorem exercise_10_4_7a {R : Type*} [comm_ring R] [no_zero_divisors R]
  (I J : ideal R) (hIJ : I + J = ⊤) : I * J = I ⊓ J :=
sorry

theorem exercise_10_7_10 {R : Type*} [ring R]
  (M : ideal R) (hM : ∀ (x : R), x ∉ M → is_unit x) :
  is_maximal M ∧ ∀ (N : ideal R), is_maximal N → N = M :=
sorry

theorem exercise_11_2_13 (a b : ℤ) :
  (of_int a : gaussian_int) ∣ of_int b → a ∣ b :=
sorry

theorem exercise_11_4_1b {F : Type*} [field F] [fintype F] (hF : card F = 2) :
  irreducible (12 + 6 * X + X ^ 3 : polynomial F) :=
sorry

theorem exercise_11_4_6a {F : Type*} [field F] [fintype F] (hF : card F = 7) :
  irreducible (X ^ 2 + 1 : polynomial F) :=
sorry

theorem exercise_11_4_6b {F : Type*} [field F] [fintype F] (hF : card F = 31) :
  irreducible (X ^ 3 - 9 : polynomial F) :=
sorry

theorem exercise_11_4_6c : irreducible (X^3 - 9 : polynomial (zmod 31)) :=
sorry

theorem exercise_11_4_8 {p : ℕ} (hp : prime p) (n : ℕ) :
  irreducible (X ^ n - p : polynomial ℚ) :=
sorry

theorem exercise_11_13_3 (N : ℕ):
  ∃ p ≥ N, nat.prime p ∧ p + 1 ≡ 0 [MOD 4] :=
sorry

theorem exercise_13_4_10 
    {p : ℕ} {hp : nat.prime p} (h : ∃ r : ℕ, p = 2 ^ r + 1) :
    ∃ (k : ℕ), p = 2 ^ (2 ^ k) + 1 :=
sorry

theorem exercise_13_6_10 {K : Type*} [field K] [fintype Kˣ] :
  ∏ (x : Kˣ), x = -1 :=
sorry

theorem exercise_1_2 :
  (⟨-1/2, real.sqrt 3 / 2⟩ : ℂ) ^ 3 = -1 :=
sorry

theorem exercise_1_3 {F V : Type*} [add_comm_group V] [field F]
  [module F V] {v : V} : -(-v) = v :=
sorry

theorem exercise_1_4 {F V : Type*} [add_comm_group V] [field F]
  [module F V] (v : V) (a : F): a • v = 0 ↔ a = 0 ∨ v = 0 :=
sorry

theorem exercise_1_6 : ∃ U : set (ℝ × ℝ),
  (U ≠ ∅) ∧
  (∀ (u v : ℝ × ℝ), u ∈ U ∧ v ∈ U → u + v ∈ U) ∧
  (∀ (u : ℝ × ℝ), u ∈ U → -u ∈ U) ∧
  (∀ U' : submodule ℝ (ℝ × ℝ), U ≠ ↑U') :=
sorry

theorem exercise_1_7 : ∃ U : set (ℝ × ℝ),
  (U ≠ ∅) ∧
  (∀ (c : ℝ) (u : ℝ × ℝ), u ∈ U → c • u ∈ U) ∧
  (∀ U' : submodule ℝ (ℝ × ℝ), U ≠ ↑U') :=
sorry

theorem exercise_1_8 {F V : Type*} [add_comm_group V] [field F]
  [module F V] {ι : Type*} (u : ι → submodule F V) :
  ∃ U : submodule F V, (⋂ (i : ι), (u i).carrier) = ↑U :=
sorry

theorem exercise_1_9 {F V : Type*} [add_comm_group V] [field F]
  [module F V] (U W : submodule F V):
  ∃ U' : submodule F V, (U'.carrier = ↑U ∩ ↑W ↔ (U ≤ W ∨ W ≤ U)) :=
sorry

theorem exercise_3_1 {F V : Type*}  
  [add_comm_group V] [field F] [module F V] [finite_dimensional F V]
  (T : V →ₗ[F] V) (hT : finrank F V = 1) :
  ∃ c : F, ∀ v : V, T v = c • v:=
sorry

theorem exercise_3_8 {F V W : Type*}  [add_comm_group V]
  [add_comm_group W] [field F] [module F V] [module F W]
  (L : V →ₗ[F] W) :
  ∃ U : submodule F V, U ⊓ L.ker = ⊥ ∧
  linear_map.range L = range (dom_restrict L U):=
sorry

theorem exercise_4_4 (p : polynomial ℂ) :
  p.degree = @card (root_set p ℂ) (polynomial.root_set_fintype p ℂ) ↔
  disjoint
  (@card (root_set p.derivative ℂ) (polynomial.root_set_fintype p.derivative ℂ))
  (@card (root_set p ℂ) (polynomial.root_set_fintype p ℂ)) :=
sorry

theorem exercise_5_1 {F V : Type*} [add_comm_group V] [field F]
  [module F V] {L : V →ₗ[F] V} {n : ℕ} (U : fin n → submodule F V)
  (hU : ∀ i : fin n, map L (U i) = U i) :
  map L (∑ i : fin n, U i : submodule F V) =
  (∑ i : fin n, U i : submodule F V) :=
sorry

theorem exercise_5_4 {F V : Type*} [add_comm_group V] [field F]
  [module F V] (S T : V →ₗ[F] V) (hST : S ∘ T = T ∘ S) (c : F):
  map S (T - c • id).ker = (T - c • id).ker :=
sorry

theorem exercise_5_11 {F V : Type*} [add_comm_group V] [field F]
  [module F V] (S T : End F V) :
  (S * T).eigenvalues = (T * S).eigenvalues  :=
sorry

theorem exercise_5_12 {F V : Type*} [add_comm_group V] [field F]
  [module F V] {S : End F V}
  (hS : ∀ v : V, ∃ c : F, v ∈ eigenspace S c) :
  ∃ c : F, S = c • id :=
sorry

theorem exercise_5_13 {F V : Type*} [add_comm_group V] [field F]
  [module F V] [finite_dimensional F V] {T : End F V}
  (hS : ∀ U : submodule F V, finrank F U = finrank F V - 1 →
  map T U = U) : ∃ c : F, T = c • id :=
sorry

theorem exercise_5_20 {F V : Type*} [add_comm_group V] [field F]
  [module F V] [finite_dimensional F V] {S T : End F V}
  (h1 : @card T.eigenvalues (eigenvalues.fintype T) = finrank F V)
  (h2 : ∀ v : V, ∃ c : F, v ∈ eigenspace S c ↔ ∃ c : F, v ∈ eigenspace T c) :
  S * T = T * S :=
sorry

theorem exercise_5_24 {V : Type*} [add_comm_group V]
  [module ℝ V] [finite_dimensional ℝ V] {T : End ℝ V}
  (hT : ∀ c : ℝ, eigenspace T c = ⊥) {U : submodule ℝ V}
  (hU : map T U = U) : even (finrank U) :=
sorry

theorem exercise_6_2 {V : Type*} [add_comm_group V] [module ℂ V]
  [inner_product_space ℂ V] (u v : V) :
  ⟪u, v⟫_ℂ = 0 ↔ ∀ (a : ℂ), ‖u‖  ≤ ‖u + a • v‖ :=
sorry

theorem exercise_6_3 {n : ℕ} (a b : fin n → ℝ) :
  (∑ i, a i * b i) ^ 2 ≤ (∑ i : fin n, i * a i ^ 2) * (∑ i, b i ^ 2 / i) :=
sorry

theorem exercise_6_7 {V : Type*} [inner_product_space ℂ V] (u v : V) :
  ⟪u, v⟫_ℂ = (‖u + v‖^2 - ‖u - v‖^2 + I*‖u + I•v‖^2 - I*‖u-I•v‖^2) / 4 :=
sorry

theorem exercise_6_13 {V : Type*} [inner_product_space ℂ V] {n : ℕ}
  {e : fin n → V} (he : orthonormal ℂ e) (v : V) :
  ‖v‖^2 = ∑ i : fin n, ‖⟪v, e i⟫_ℂ‖^2 ↔ v ∈ span ℂ (e '' univ) :=
sorry

theorem exercise_6_16 {K V : Type*} [is_R_or_C K] [inner_product_space K V]
  {U : submodule K V} : 
  U.orthogonal = ⊥  ↔ U = ⊤ :=
sorry 

theorem exercise_7_5 {V : Type*} [inner_product_space ℂ V] 
  [finite_dimensional ℂ V] (hV : finrank V ≥ 2) :
  ∀ U : submodule ℂ (End ℂ V), U.carrier ≠
  {T | T * T.adjoint = T.adjoint * T} :=
sorry

theorem exercise_7_6 {V : Type*} [inner_product_space ℂ V]
  [finite_dimensional ℂ V] (T : End ℂ V)
  (hT : T * T.adjoint = T.adjoint * T) :
  T.range = T.adjoint.range :=
sorry

theorem exercise_7_9 {V : Type*} [inner_product_space ℂ V]
  [finite_dimensional ℂ V] (T : End ℂ V)
  (hT : T * T.adjoint = T.adjoint * T) :
  is_self_adjoint T ↔ ∀ e : T.eigenvalues, (e : ℂ).im = 0 :=
sorry

theorem exercise_7_10 {V : Type*} [inner_product_space ℂ V]
  [finite_dimensional ℂ V] (T : End ℂ V)
  (hT : T * T.adjoint = T.adjoint * T) (hT1 : T^9 = T^8) :
  is_self_adjoint T ∧ T^2 = T :=
sorry

theorem exercise_7_11 {V : Type*} [inner_product_space ℂ V]
  [finite_dimensional ℂ V] {T : End ℂ V} (hT : T*T.adjoint = T.adjoint*T) :
  ∃ (S : End ℂ V), S ^ 2 = T :=
sorry

theorem exercise_7_14 {𝕜 V : Type*} [is_R_or_C 𝕜]
  [inner_product_space 𝕜 V] [finite_dimensional 𝕜 V]
  {T : End 𝕜 V} (hT : is_self_adjoint T)
  {l : 𝕜} {ε : ℝ} (he : ε > 0) : ∃ v : V, ‖v‖= 1 ∧ (‖T v - l • v‖ < ε →
  (∃ l' : T.eigenvalues, ‖l - l'‖ < ε)) :=
sorry

theorem exercise_2022_IA_4_I_1E_a : ∀ N : ℕ, ∃ n ≥ N, (3*n+1).prime ∧ (3*n+1) ≥ N :=
sorry

theorem exercise_2022_IA_4_I_2D_a : irrational (2^((1:ℝ)/3) + 3^((1:ℝ)/3)) :=
sorry

theorem exercise_2022_IB_3_II_13G_a_i (U : set ℂ) (hU : is_open U)
  (hU1 : nonempty U) (hU2 : is_connected U) (f : ℕ → ℂ → ℂ) (f' : ℂ → ℂ)
  (hf : ∀ n : ℕ, differentiable_on ℂ (f n) U)
  (hf1 : ∀ X ⊂ U, compact_space X →
  (tendsto_uniformly (λ n, set.restrict X (f n)) (set.restrict X f') at_top)) :
  differentiable_on ℂ f' U :=
sorry

theorem exercise_1_1_2a : ∃ a b : ℤ, a - b ≠ b - a :=
begin
  use [0, 1]
end

theorem exercise_1_1_3 (n : ℤ) : 
  ∀ (a b c : ℤ), (a+b)+c ≡ a+(b+c) [ZMOD n] :=
begin 
  intros a b c, 
  ring_nf
end

theorem exercise_1_1_4 (n : ℕ) : 
  ∀ (a b c : ℕ), (a * b) * c ≡ a * (b * c) [ZMOD n] :=
begin 
  intros a b c, 
  ring_nf, 
end

theorem exercise_1_1_5 (n : ℕ) (hn : 1 < n) : 
  is_empty (group (zmod n)) := 
sorry 

theorem exercise_1_1_15 {G : Type*} [group G] (as : list G) :
  as.prod⁻¹ = (as.reverse.map (λ x, x⁻¹)).prod :=
begin 
  simp only [list.prod_hom _, list.map_reverse, list.prod_reverse],
  induction as generalizing, 
  simp, 
  simp *, 
end

theorem exercise_1_1_16 {G : Type*} [group G] 
  (x : G) (hx : x ^ 2 = 1) :
  order_of x = 1 ∨ order_of x = 2 :=
sorry 

theorem exercise_1_1_17 {G : Type*} [group G] {x : G} {n : ℕ}
  (hxn: order_of x = n) :
  x⁻¹ = x ^ (n - 1 : ℤ) :=
sorry 

theorem exercise_1_1_18 {G : Type*} [group G]
  (x y : G) : x * y = y * x ↔ y⁻¹ * x * y = x ↔ x⁻¹ * y⁻¹ * x * y = 1 :=
sorry

theorem exercise_1_1_20 {G : Type*} [group G] {x : G} :
  order_of x = order_of x⁻¹ :=
sorry 

theorem exercise_1_1_22a {G : Type*} [group G] (x g : G) :
  order_of x = order_of (g⁻¹ * x * g) :=
sorry 

theorem exercise_1_1_22b {G: Type*} [group G] (a b : G) : 
  order_of (a * b) = order_of (b * a) :=
sorry

theorem exercise_1_1_25 {G : Type*} [group G] 
  (h : ∀ x : G, x ^ 2 = 1) : ∀ a b : G, a*b = b*a :=
sorry 

theorem exercise_1_1_29 {A B : Type*} [group A] [group B] :
  ∀ x y : A × B, x*y = y*x ↔ (∀ x y : A, x*y = y*x) ∧ 
  (∀ x y : B, x*y = y*x) :=
sorry

theorem exercise_1_1_34 {G : Type*} [group G] {x : G} 
  (hx_inf : order_of x = 0) (n m : ℤ) :
  x ^ n ≠ x ^ m :=
sorry

theorem exercise_1_3_8 : infinite (equiv.perm ℕ) :=
sorry

theorem exercise_1_6_4 : 
  is_empty (multiplicative ℝ ≃* multiplicative ℂ) :=
sorry

theorem exercise_1_6_11 {A B : Type*} [group A] [group B] : 
  A × B ≃* B × A :=
sorry 

theorem exercise_1_6_17 {G : Type*} [group G] (f : G → G) 
  (hf : f = λ g, g⁻¹) :
  ∀ x y : G, f x * f y = f (x*y) ↔ ∀ x y : G, x*y = y*x :=   
sorry

theorem exercise_1_6_23 {G : Type*} 
  [group G] (σ : mul_aut G) (hs : ∀ g : G, σ g = 1 → g = 1) 
  (hs2 : ∀ g : G, σ (σ g) = g) :
  ∀ x y : G, x*y = y*x :=
sorry

theorem exercise_2_1_5 {G : Type*} [group G] [fintype G] 
  (hG : card G > 2) (H : subgroup G) [fintype H] : 
  card H ≠ card G - 1 :=
sorry 

theorem exercise_2_1_13 (H : add_subgroup ℚ) {x : ℚ} 
  (hH : x ∈ H → (1 / x) ∈ H):
  H = ⊥ ∨ H = ⊤ :=
sorry

theorem exercise_2_4_4 {G : Type*} [group G] (H : subgroup G) : 
  subgroup.closure ((H : set G) \ {1}) = ⊤ :=
sorry 

theorem exercise_2_4_16a {G : Type*} [group G] {H : subgroup G}  
  (hH : H ≠ ⊤) : 
  ∃ M : subgroup G, M ≠ ⊤ ∧
  ∀ K : subgroup G, M ≤ K → K = M ∨ K = ⊤ ∧ 
  H ≤ M :=
sorry 

theorem exercise_2_4_16b {n : ℕ} {hn : n ≠ 0} 
  {R : subgroup (dihedral_group n)} 
  (hR : R = subgroup.closure {dihedral_group.r 1}) : 
  R ≠ ⊤ ∧ 
  ∀ K : subgroup (dihedral_group n), R ≤ K → K = R ∨ K = ⊤ :=
sorry 

theorem exercise_2_4_16c {n : ℕ} (H : add_subgroup (zmod n)) : 
  ∃ p : ℕ, nat.prime p ∧ H = add_subgroup.closure {p} ↔ 
  H ≠ ⊤ ∧ ∀ K : add_subgroup (zmod n), H ≤ K → K = H ∨ K = ⊤ := 
sorry 

theorem exercise_3_1_3a {A : Type*} [comm_group A] (B : subgroup A) :
  ∀ a b : A ⧸ B, a*b = b*a :=
sorry

theorem exercise_3_1_22a (G : Type*) [group G] (H K : subgroup G) 
  [subgroup.normal H] [subgroup.normal K] :
  subgroup.normal (H ⊓ K) :=
sorry

theorem exercise_3_1_22b {G : Type*} [group G] (I : Type*)
  (H : I → subgroup G) (hH : ∀ i : I, subgroup.normal (H i)) : 
  subgroup.normal (⨅ (i : I), H i):=
sorry

theorem exercise_3_2_8 {G : Type*} [group G] (H K : subgroup G)
  [fintype H] [fintype K] 
  (hHK : nat.coprime (fintype.card H) (fintype.card K)) : 
  H ⊓ K = ⊥ :=
sorry 

theorem exercise_3_2_11 {G : Type*} [group G] {H K : subgroup G}
  (hHK : H ≤ K) : 
  H.index = K.index * H.relindex K :=
sorry 

theorem exercise_3_2_16 (p : ℕ) (hp : nat.prime p) (a : ℕ) :
  nat.coprime a p → a ^ p ≡ a [ZMOD p] :=
sorry

theorem exercise_3_2_21a (H : add_subgroup ℚ) (hH : H ≠ ⊤) : H.index = 0 :=
sorry

theorem exercise_3_3_3 {p : primes} {G : Type*} [group G] 
  {H : subgroup G} [hH : H.normal] (hH1 : H.index = p) : 
  ∀ K : subgroup G, K ≤ H ∨ H ⊔ K = ⊤ ∨ (K ⊓ H).relindex K = p :=
sorry 

theorem exercise_3_4_1 (G : Type*) [comm_group G] [is_simple_group G] :
    is_cyclic G ∧ ∃ G_fin : fintype G, nat.prime (@card G G_fin) :=
sorry

theorem exercise_3_4_4 {G : Type*} [comm_group G] [fintype G] {n : ℕ}
    (hn : n ∣ (fintype.card G)) :
    ∃ (H : subgroup G) (H_fin : fintype H), @card H H_fin = n  :=
sorry

theorem exercise_3_4_5a {G : Type*} [group G] 
  (H : subgroup G) [is_solvable G] : is_solvable H :=
sorry

theorem exercise_3_4_5b {G : Type*} [group G] [is_solvable G] 
  (H : subgroup G) [subgroup.normal H] : 
  is_solvable (G ⧸ H) :=
sorry

theorem exercise_3_4_11 {G : Type*} [group G] [is_solvable G] 
  {H : subgroup G} (hH : H ≠ ⊥) [H.normal] : 
  ∃ A ≤ H, A.normal ∧ ∀ a b : A, a*b = b*a :=
sorry 

theorem exercise_4_2_8 {G : Type*} [group G] {H : subgroup G} 
  {n : ℕ} (hn : n > 0) (hH : H.index = n) : 
  ∃ K ≤ H, K.normal ∧ K.index ≤ n.factorial :=
sorry 

theorem exercise_4_3_26 {α : Type*} [fintype α] (ha : fintype.card α > 1)
  (h_tran : ∀ a b: α, ∃ σ : equiv.perm α, σ a = b) : 
  ∃ σ : equiv.perm α, ∀ a : α, σ a ≠ a := 
sorry

theorem exercise_4_2_9a {G : Type*} [fintype G] [group G] {p α : ℕ} 
  (hp : p.prime) (ha : α > 0) (hG : card G = p ^ α) : 
  ∀ H : subgroup G, H.index = p → H.normal :=
sorry 

theorem exercise_4_2_14 {G : Type*} [fintype G] [group G] 
  (hG : ¬ (card G).prime) (hG1 : ∀ k ∣ card G, 
  ∃ (H : subgroup G) (fH : fintype H), @card H fH = k) : 
  ¬ is_simple_group G :=
sorry 

theorem exercise_4_4_2 {G : Type*} [fintype G] [group G] 
  {p q : nat.primes} (hpq : p ≠ q) (hG : card G = p*q) : 
  is_cyclic G :=
sorry 

theorem exercise_4_4_6a {G : Type*} [group G] (H : subgroup G)
  [subgroup.characteristic H] : subgroup.normal H  :=
sorry

theorem exercise_4_4_6b : 
  ∃ (G : Type*) (hG : group G) (H : @subgroup G hG), @characteristic G hG H  ∧ ¬ @subgroup.normal G hG H :=
sorry 

theorem exercise_4_4_7 {G : Type*} [group G] {H : subgroup G} [fintype H]
  (hH : ∀ (K : subgroup G) (fK : fintype K), card H = @card K fK → H = K) : 
  H.characteristic :=
sorry 

theorem exercise_4_4_8a {G : Type*} [group G] (H K : subgroup G)  
  (hHK : H ≤ K) [hHK1 : (H.subgroup_of K).normal] [hK : K.normal] : 
  H.normal :=
sorry 

theorem exercise_4_5_1a {p : ℕ} {G : Type*} [group G] 
  {P : subgroup G} (hP : is_p_group p P) (H : subgroup G) 
  (hH : P ≤ H) : is_p_group p H :=
sorry

theorem exercise_4_5_13 {G : Type*} [group G] [fintype G]
  (hG : card G = 56) :
  ∃ (p : ℕ) (P : sylow p G), P.normal :=
sorry

theorem exercise_4_5_14 {G : Type*} [group G] [fintype G]
  (hG : card G = 312) :
  ∃ (p : ℕ) (P : sylow p G), P.normal :=
sorry

theorem exercise_4_5_15 {G : Type*} [group G] [fintype G] 
  (hG : card G = 351) : 
  ∃ (p : ℕ) (P : sylow p G), P.normal :=
sorry 

theorem exercise_4_5_16 {p q r : ℕ} {G : Type*} [group G] 
  [fintype G]  (hpqr : p < q ∧ q < r) 
  (hpqr1 : p.prime ∧ q.prime ∧ r.prime)(hG : card G = p*q*r) : 
  nonempty (sylow p G) ∨ nonempty(sylow q G) ∨ nonempty(sylow r G) :=
sorry 

theorem exercise_4_5_17 {G : Type*} [fintype G] [group G] 
  (hG : card G = 105) : 
  nonempty(sylow 5 G) ∧ nonempty(sylow 7 G) :=
sorry 

theorem exercise_4_5_18 {G : Type*} [fintype G] [group G] 
  (hG : card G = 200) : 
  ∃ N : sylow 5 G, N.normal :=
sorry 

theorem exercise_4_5_19 {G : Type*} [fintype G] [group G] 
  (hG : card G = 6545) : ¬ is_simple_group G :=
sorry 

theorem exercise_4_5_20 {G : Type*} [fintype G] [group G]
  (hG : card G = 1365) : ¬ is_simple_group G :=
sorry 

theorem exercise_4_5_21 {G : Type*} [fintype G] [group G]
  (hG : card G = 2907) : ¬ is_simple_group G :=
sorry 

theorem exercise_4_5_22 {G : Type*} [fintype G] [group G]
  (hG : card G = 132) : ¬ is_simple_group G :=
sorry 

theorem exercise_4_5_23 {G : Type*} [fintype G] [group G]
  (hG : card G = 462) : ¬ is_simple_group G :=
sorry 

theorem exercise_4_5_28 {G : Type*} [group G] [fintype G] 
  (hG : card G = 105) (P : sylow 3 G) [hP : P.normal] : 
  comm_group G :=
sorry 

theorem exercise_4_5_33 {G : Type*} [group G] [fintype G] {p : ℕ} 
  (P : sylow p G) [hP : P.normal] (H : subgroup G) [fintype H] : 
  ∀ R : sylow p H, R.to_subgroup = (H ⊓ P.to_subgroup).subgroup_of H ∧
  nonempty (sylow p H) :=
sorry 

theorem exercise_5_4_2 {G : Type*} [group G] (H : subgroup G) : 
  H.normal ↔ ⁅(⊤ : subgroup G), H⁆ ≤ H := 
sorry 

theorem exercise_7_1_2 {R : Type*} [ring R] {u : R}
  (hu : is_unit u) : is_unit (-u) :=
sorry 

theorem exercise_7_1_11 {R : Type*} [ring R] [is_domain R] 
  {x : R} (hx : x^2 = 1) : x = 1 ∨ x = -1 :=
sorry 

theorem exercise_7_1_12 {F : Type*} [field F] {K : subring F}
  (hK : (1 : F) ∈ K) : is_domain K :=
sorry 

theorem exercise_7_1_15 {R : Type*} [ring R] (hR : ∀ a : R, a^2 = a) :
  comm_ring R :=
sorry 

theorem exercise_7_2_2 {R : Type*} [ring R] (p : polynomial R) :
  p ∣ 0 ↔ ∃ b : R, b ≠ 0 ∧ b • p = 0 := 
sorry 

theorem exercise_7_2_12 {R G : Type*} [ring R] [group G] [fintype G] : 
  ∑ g : G, monoid_algebra.of R G g ∈ center (monoid_algebra R G) :=
sorry 

theorem exercise_7_3_16 {R S : Type*} [ring R] [ring S] 
  {φ : R →+* S} (hf : surjective φ) : 
  φ '' (center R) ⊂ center S :=
sorry 

theorem exercise_7_3_37 {R : Type*} {p m : ℕ} (hp : p.prime) 
  (N : ideal $ zmod $ p^m) : 
  is_nilpotent N ↔  is_nilpotent (ideal.span ({p} : set $ zmod $ p^m)) :=
sorry 

theorem exercise_7_4_27 {R : Type*} [comm_ring R] (hR : (0 : R) ≠ 1) 
  {a : R} (ha : is_nilpotent a) (b : R) : 
  is_unit (1-a*b) :=
sorry 

theorem exercise_8_1_12 {N : ℕ} (hN : N > 0) {M M': ℤ} {d : ℕ}
  (hMN : M.gcd N = 1) (hMd : d.gcd N.totient = 1) 
  (hM' : M' ≡ M^d [ZMOD N]) : 
  ∃ d' : ℕ, d' * d ≡ 1 [ZMOD N.totient] ∧ 
  M ≡ M'^d' [ZMOD N] :=
sorry 

theorem exercise_8_2_4 {R : Type*} [ring R][no_zero_divisors R] 
  [cancel_comm_monoid_with_zero R] [gcd_monoid R]
  (h1 : ∀ a b : R, a ≠ 0 → b ≠ 0 → ∃ r s : R, gcd a b = r*a + s*b)
  (h2 : ∀ a : ℕ → R, (∀ i j : ℕ, i < j → a i ∣ a j) → 
  ∃ N : ℕ, ∀ n ≥ N, ∃ u : R, is_unit u ∧ a n = u * a N) : 
  is_principal_ideal_ring R :=
sorry  

theorem exercise_8_3_4 {R : Type*} {n : ℤ} {r s : ℚ} 
  (h : r^2 + s^2 = n) : 
  ∃ a b : ℤ, a^2 + b^2 = n :=
sorry 

theorem exercise_8_3_5a {n : ℤ} (hn0 : n > 3) (hn1 : squarefree n) : 
  irreducible (2 :zsqrtd $ -n) ∧ 
  irreducible (⟨0, 1⟩ : zsqrtd $ -n) ∧ 
  irreducible (1 + ⟨0, 1⟩ : zsqrtd $ -n) :=
sorry 

theorem exercise_8_3_6a {R : Type*} [ring R]
  (hR : R = (gaussian_int ⧸ ideal.span ({⟨0, 1⟩} : set gaussian_int))) :
  is_field R ∧ ∃ finR : fintype R, @card R finR = 2 :=
sorry 

theorem exercise_8_3_6b {q : ℕ} (hq0 : q.prime) 
  (hq1 : q ≡ 3 [ZMOD 4]) {R : Type*} [ring R]
  (hR : R = (gaussian_int ⧸ ideal.span ({q} : set gaussian_int))) : 
  is_field R ∧ ∃ finR : fintype R, @card R finR = q^2 :=
sorry 
   
theorem exercise_9_1_6 : ¬ is_principal 
  (ideal.span ({X 0, X 1} : set (mv_polynomial (fin 2) ℚ))) :=
sorry 

theorem exercise_9_1_10 {f : ℕ → mv_polynomial ℕ ℤ} 
  (hf : f = λ i, X i * X (i+1)): 
  infinite (minimal_primes (mv_polynomial ℕ ℤ ⧸ ideal.span (range f))) := 
sorry 

theorem exercise_9_3_2 {f g : polynomial ℚ} (i j : ℕ)
  (hfg : ∀ n : ℕ, ∃ a : ℤ, (f*g).coeff = a) :
  ∃ a : ℤ, f.coeff i * g.coeff j = a :=
sorry 

theorem exercise_9_4_2a : irreducible (X^4 - 4*X^3 + 6 : polynomial ℤ) := 
sorry 

theorem exercise_9_4_2b : irreducible 
  (X^6 + 30*X^5 - 15*X^3 + 6*X - 120 : polynomial ℤ) :=
sorry 

theorem exercise_9_4_2c : irreducible 
  (X^4 + 4*X^3 + 6*X^2 + 2*X + 1 : polynomial ℤ) :=
sorry 

theorem exercise_9_4_2d {p : ℕ} (hp : p.prime ∧ p > 2) 
  {f : polynomial ℤ} (hf : f = (X + 2)^p): 
  irreducible (∑ n in (f.support \ {0}), (f.coeff n) * X ^ (n-1) : 
  polynomial ℤ) :=
sorry 

theorem exercise_9_4_9 : 
  irreducible (X^2 - C sqrtd : polynomial (zsqrtd 2)) :=
sorry 

theorem exercise_9_4_11 : 
  irreducible ((X 0)^2 + (X 1)^2 - 1 : mv_polynomial (fin 2) ℚ) :=
sorry 

theorem exercise_11_1_13 {ι : Type*} [fintype ι] : 
  (ι → ℝ) ≃ₗ[ℚ] ℝ :=
sorry 

theorem exercise_2_1_18 {G : Type*} [group G] 
  [fintype G] (hG2 : even (fintype.card G)) :
  ∃ (a : G), a ≠ 1 ∧ a = a⁻¹ :=
sorry

theorem exercise_2_1_21 (G : Type*) [group G] [fintype G]
  (hG : card G = 5) :
  comm_group G :=
sorry

theorem exercise_2_1_26 {G : Type*} [group G] 
  [fintype G] (a : G) : ∃ (n : ℕ), a ^ n = 1 :=
sorry

theorem exercise_2_1_27 {G : Type*} [group G] 
  [fintype G] : ∃ (m : ℕ), ∀ (a : G), a ^ m = 1 :=
sorry

theorem exercise_2_2_3 {G : Type*} [group G]
  {P : ℕ → Prop} {hP : P = λ i, ∀ a b : G, (a*b)^i = a^i * b^i}
  (hP1 : ∃ n : ℕ, P n ∧ P (n+1) ∧ P (n+2)) : comm_group G :=
sorry

theorem exercise_2_2_5 {G : Type*} [group G] 
  (h : ∀ (a b : G), (a * b) ^ 3 = a ^ 3 * b ^ 3 ∧ (a * b) ^ 5 = a ^ 5 * b ^ 5) :
  comm_group G :=
sorry

theorem exercise_2_2_6c {G : Type*} [group G] {n : ℕ} (hn : n > 1) 
  (h : ∀ (a b : G), (a * b) ^ n = a ^ n * b ^ n) :
  ∀ (a b : G), (a * b * a⁻¹ * b⁻¹) ^ (n * (n - 1)) = 1 :=
sorry

theorem exercise_2_3_17 {G : Type*} [has_mul G] [group G] (a x : G) :  
  set.centralizer {x⁻¹*a*x} = 
  (λ g : G, x⁻¹*g*x) '' (set.centralizer {a}) :=
sorry

theorem exercise_2_3_16 {G : Type*} [group G]
  (hG : ∀ H : subgroup G, H = ⊤ ∨ H = ⊥) :
  is_cyclic G ∧ ∃ (p : ℕ) (fin : fintype G), nat.prime p ∧ @card G fin = p :=
sorry

theorem exercise_2_4_36 {a n : ℕ} (h : a > 1) :
  n ∣ (a ^ n - 1).totient :=
sorry

theorem exercise_2_5_23 {G : Type*} [group G] 
  (hG : ∀ (H : subgroup G), H.normal) (a b : G) :
  ∃ (j : ℤ) , b*a = a^j * b:=
sorry

theorem exercise_2_5_30 {G : Type*} [group G] [fintype G]
  {p m : ℕ} (hp : nat.prime p) (hp1 : ¬ p ∣ m) (hG : card G = p*m) 
  {H : subgroup G} [fintype H] [H.normal] (hH : card H = p):
  characteristic H :=
sorry

theorem exercise_2_5_31 {G : Type*} [comm_group G] [fintype G]
  {p m n : ℕ} (hp : nat.prime p) (hp1 : ¬ p ∣ m) (hG : card G = p^n*m)
  {H : subgroup G} [fintype H] (hH : card H = p^n) : 
  characteristic H :=
sorry

theorem exercise_2_5_37 (G : Type*) [group G] [fintype G]
  (hG : card G = 6) (hG' : is_empty (comm_group G)) :
  G ≃* equiv.perm (fin 3) :=
sorry

theorem exercise_2_5_43 (G : Type*) [group G] [fintype G]
  (hG : card G = 9) :
  comm_group G :=
sorry

theorem exercise_2_5_44 {G : Type*} [group G] [fintype G] {p : ℕ}
  (hp : nat.prime p) (hG : card G = p^2) :
  ∃ (N : subgroup G) (fin : fintype N), @card N fin = p ∧ N.normal :=
sorry

theorem exercise_2_5_52 {G : Type*} [group G] [fintype G]
  (φ : G ≃* G) {I : finset G} (hI : ∀ x ∈ I, φ x = x⁻¹)
  (hI1 : (0.75 : ℚ) * card G ≤ card I) : 
  ∀ x : G, φ x = x⁻¹ ∧ ∀ x y : G, x*y = y*x :=
sorry

theorem exercise_2_6_15 {G : Type*} [comm_group G] {m n : ℕ} 
  (hm : ∃ (g : G), order_of g = m) 
  (hn : ∃ (g : G), order_of g = n) 
  (hmn : m.coprime n) :
  ∃ (g : G), order_of g = m * n :=
sorry

theorem exercise_2_7_7 {G : Type*} [group G] {G' : Type*} [group G']
  (φ : G →* G') (N : subgroup G) [N.normal] : 
  (map φ N).normal  :=
sorry

theorem exercise_2_8_12 {G H : Type*} [fintype G] [fintype H] 
  [group G] [group H] (hG : card G = 21) (hH : card H = 21) 
  (hG1 : is_empty(comm_group G)) (hH1 : is_empty (comm_group H)) :
  G ≃* H :=
sorry 

theorem exercise_2_8_15 {G H: Type*} [fintype G] [group G] [fintype H]
  [group H] {p q : ℕ} (hp : nat.prime p) (hq : nat.prime q) 
  (h : p > q) (h1 : q ∣ p - 1) (hG : card G = p*q) (hH : card G = p*q) :
  G ≃* H :=
sorry

theorem exercise_2_9_2 {G H : Type*} [fintype G] [fintype H] [group G] 
  [group H] (hG : is_cyclic G) (hH : is_cyclic H) :
  is_cyclic (G × H) ↔ (card G).coprime (card H) :=
sorry

theorem exercise_2_10_1 {G : Type*} [group G] (A : subgroup G) 
  [A.normal] {b : G} (hp : nat.prime (order_of b)) :
  A ⊓ (closure {b}) = ⊥ :=
sorry

theorem exercise_2_11_6 {G : Type*} [group G] {p : ℕ} (hp : nat.prime p) 
  {P : sylow p G} (hP : P.normal) :
  ∀ (Q : sylow p G), P = Q :=
sorry

theorem exercise_2_11_7 {G : Type*} [group G] {p : ℕ} (hp : nat.prime p)
  {P : sylow p G} (hP : P.normal) : 
  characteristic (P : subgroup G) :=
sorry

theorem exercise_2_11_22 {p : ℕ} {n : ℕ} {G : Type*} [fintype G] 
  [group G] (hp : nat.prime p) (hG : card G = p ^ n) {K : subgroup G}
  [fintype K] (hK : card K = p ^ (n-1)) : 
  K.normal :=
sorry

theorem exercise_3_2_21 {α : Type*} [fintype α] {σ τ: equiv.perm α} 
  (h1 : ∀ a : α, σ a = a ↔ τ a ≠ a) (h2 : τ ∘ σ = id) : 
  σ = 1 ∧ τ = 1 :=
sorry

theorem exercise_4_1_19 : infinite {x : quaternion ℝ | x^2 = -1} :=
sorry

theorem exercise_4_1_34 : equiv.perm (fin 3) ≃* general_linear_group (fin 2) (zmod 2) :=
sorry

theorem exercise_4_2_5 {R : Type*} [ring R] 
  (h : ∀ x : R, x ^ 3 = x) : comm_ring R :=
sorry

theorem exercise_4_2_6 {R : Type*} [ring R] (a x : R) 
  (h : a ^ 2 = 0) : a * (a * x + x * a) = (x + x * a) * a :=
sorry

theorem exercise_4_2_9 {p : ℕ} (hp : nat.prime p) (hp1 : odd p) :
  ∃ (a b : ℤ), (a / b : ℚ) = ∑ i in finset.range p, 1 / (i + 1) → ↑p ∣ a :=
sorry

theorem exercise_4_3_1 {R : Type*} [comm_ring R] (a : R) :
  ∃ I : ideal R, {x : R | x*a=0} = I :=
sorry

theorem exercise_4_3_25 (I : ideal (matrix (fin 2) (fin 2) ℝ)) : 
  I = ⊥ ∨ I = ⊤ :=
sorry

theorem exercise_4_4_9 (p : ℕ) (hp : nat.prime p) :
  (∃ S : finset (zmod p), S.card = (p-1)/2 ∧ ∃ x : zmod p, x^2 = p) ∧ 
  (∃ S : finset (zmod p), S.card = (p-1)/2 ∧ ¬ ∃ x : zmod p, x^2 = p) :=
sorry

theorem exercise_4_5_16 {p n: ℕ} (hp : nat.prime p) 
  {q : polynomial (zmod p)} (hq : irreducible q) (hn : q.degree = n) :
  ∃ is_fin : fintype $ polynomial (zmod p) ⧸ ideal.span ({q} : set (polynomial $ zmod p)), 
  @card (polynomial (zmod p) ⧸ ideal.span {q}) is_fin = p ^ n ∧ 
  is_field (polynomial $ zmod p):=
sorry

theorem exercise_4_5_23 {p q: polynomial (zmod 7)} 
  (hp : p = X^3 - 2) (hq : q = X^3 + 2) : 
  irreducible p ∧ irreducible q ∧ 
  (nonempty $ polynomial (zmod 7) ⧸ ideal.span ({p} : set $ polynomial $ zmod 7) ≃+*
  polynomial (zmod 7) ⧸ ideal.span ({q} : set $ polynomial $ zmod 7)) :=
sorry

theorem exercise_4_5_25 {p : ℕ} (hp : nat.prime p) :
  irreducible (∑ i : finset.range p, X ^ p : polynomial ℚ) :=
sorry

theorem exercise_4_6_2 : irreducible (X^3 + 3*X + 2 : polynomial ℚ) :=
sorry

theorem exercise_4_6_3 :
  infinite {a : ℤ | irreducible (X^7 + 15*X^2 - 30*X + a : polynomial ℚ)} :=
sorry

theorem exercise_5_1_8 {p m n: ℕ} {F : Type*} [field F] 
  (hp : nat.prime p) (hF : char_p F p) (a b : F) (hm : m = p ^ n) : 
  (a + b) ^ m = a^m + b^m :=
sorry

theorem exercise_5_2_20 {F V ι: Type*} [infinite F] [field F] 
  [add_comm_group V] [module F V] {u : ι → submodule F V} 
  (hu : ∀ i : ι, u i ≠ ⊤) : 
  (⋃ i : ι, (u i : set V)) ≠ ⊤ :=
sorry

theorem exercise_5_3_7 {K : Type*} [field K] {F : subfield K} 
  {a : K} (ha : is_algebraic F (a ^ 2)) : is_algebraic F a :=
sorry 

theorem exercise_5_3_10 : is_algebraic ℚ (cos (real.pi / 180)) :=
sorry

theorem exercise_5_4_3 {a : ℂ} {p : ℂ → ℂ} 
  (hp : p = λ x, x^5 + real.sqrt 2 * x^3 + real.sqrt 5 * x^2 + 
  real.sqrt 7 * x + 11)
  (ha : p a = 0) : 
  ∃ p : polynomial ℂ , p.degree < 80 ∧ a ∈ p.roots ∧ 
  ∀ n : p.support, ∃ a b : ℤ, p.coeff n = a / b :=
sorry

theorem exercise_5_5_2 : irreducible (X^3 - 3*X - 1 : polynomial ℚ) :=
sorry 

theorem exercise_5_6_14 {p m n: ℕ} (hp : nat.prime p) {F : Type*} 
  [field F] [char_p F p] (hm : m = p ^ n) : 
  card (root_set (X ^ m - X : polynomial F) F) = m :=
sorry

theorem exercise_1_27 {n : ℕ} (hn : odd n) : 8 ∣ (n^2 - 1) :=
sorry 

theorem exercise_1_30 {n : ℕ} : 
  ¬ ∃ a : ℤ, ∑ (i : fin n), (1 : ℚ) / (n+2) = a :=
sorry 

theorem exercise_1_31  : (⟨1, 1⟩ : gaussian_int) ^ 2 ∣ 2 := 
sorry 

theorem exercise_2_4 {a : ℤ} (ha : a ≠ 0) 
  (f_a := λ n m : ℕ, int.gcd (a^(2^n) + 1) (a^(2^m)+1)) {n m : ℕ} 
  (hnm : n > m) : 
  (odd a → f_a n m = 1) ∧ (even a → f_a n m = 2) :=
sorry 

theorem exercise_2_21 {l : ℕ → ℝ} 
  (hl : ∀ p n : ℕ, p.prime → l (p^n) = log p )
  (hl1 : ∀ m : ℕ, ¬ is_prime_pow m → l m = 0) :
  l = λ n, ∑ d : divisors n, moebius (n/d) * log d  := 
sorry 

theorem exercise_2_27a : 
  ¬ summable (λ i : {p : ℤ // squarefree p}, (1 : ℚ) / i) :=
sorry 

theorem exercise_3_1 : infinite {p : primes // p ≡ -1 [ZMOD 6]} :=
sorry 

theorem exercise_3_4 : ¬ ∃ x y : ℤ, 3*x^2 + 2 = y^2 :=
sorry 

theorem exercise_3_5 : ¬ ∃ x y : ℤ, 7*x^3 + 2 = y^3 :=
sorry 

theorem exercise_3_10 {n : ℕ} (hn0 : ¬ n.prime) (hn1 : n ≠ 4) : 
  factorial (n-1) ≡ 0 [MOD n] :=
sorry 

theorem exercise_3_14 {p q n : ℕ} (hp0 : p.prime ∧ p > 2) 
  (hq0 : q.prime ∧ q > 2) (hpq0 : p ≠ q) (hpq1 : p - 1 ∣ q - 1)
  (hn : n.gcd (p*q) = 1) : 
  n^(q-1) ≡ 1 [MOD p*q] :=
sorry 

theorem exercise_4_4 {p t: ℕ} (hp0 : p.prime) (hp1 : p = 4*t + 1) 
  (a : zmod p) : 
  is_primitive_root a p ↔ is_primitive_root (-a) p :=
sorry 

theorem exercise_4_5 {p t : ℕ} (hp0 : p.prime) (hp1 : p = 4*t + 3)
  (a : zmod p) :
  is_primitive_root a p ↔ ((-a) ^ ((p-1)/2) = 1 ∧ ∀ (k : ℕ), k < (p-1)/2 → (-a)^k ≠ 1) :=
sorry 

theorem exercise_4_6 {p n : ℕ} (hp : p.prime) (hpn : p = 2^n + 1) : 
  is_primitive_root 3 p :=
sorry 

theorem exercise_4_8 {p a : ℕ} (hp : odd p) : 
  is_primitive_root a p ↔ (∀ q ∣ (p-1), q.prime → ¬ a^(p-1) ≡ 1 [MOD p]) :=
sorry 

theorem exercise_4_11 {p : ℕ} (hp : p.prime) (k s: ℕ) 
  (s := ∑ (n : fin p), (n : ℕ) ^ k) : 
  ((¬ p - 1 ∣ k) → s ≡ 0 [MOD p]) ∧ (p - 1 ∣ k → s ≡ 0 [MOD p]) :=
sorry 

theorem exercise_5_13 {p x: ℤ} (hp : prime p) 
  (hpx : p ∣ (x^4 - x^2 + 1)) : p ≡ 1 [ZMOD 12] :=
sorry 

theorem exercise_5_28 {p : ℕ} (hp : p.prime) (hp1 : p ≡ 1 [MOD 4]): 
  ∃ x, x^4 ≡ 2 [MOD p] ↔ ∃ A B, p = A^2 + 64*B^2 :=
sorry 

theorem exercise_5_37 {p q : ℕ} [fact(p.prime)] [fact(q.prime)] {a : ℤ}
  (ha : a < 0) (h0 : p ≡ q [ZMOD 4*a]) (h1 : ¬ ((p : ℤ) ∣ a)) :
  legendre_sym p a = legendre_sym q a :=
sorry 

theorem exercise_12_12 : is_algebraic ℚ (sin (real.pi/12)) :=
sorry 

theorem exercise_18_4 {n : ℕ} (hn : ∃ x y z w : ℤ, 
  x^3 + y^3 = n ∧ z^3 + w^3 = n ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w) : 
  n ≥ 1729 :=
sorry 

theorem exercise_13_1 (X : Type*) [topological_space X] (A : set X)
  (h1 : ∀ x ∈ A, ∃ U : set X, x ∈ U ∧ is_open U ∧ U ⊆ A) :
  is_open A :=
begin
  have : A = ⋃ x, ⋃ h : x ∈ A, (classical.some (h1 x h)),
  { ext x, simp, split,
  { intro xA,
  use [x, xA],
  exact (classical.some_spec (h1 x xA)).1},
  { rintros ⟨y, yA, yspec⟩,
  have h := classical.some_spec (h1 y yA),
  exact h.2.2 yspec }, },
  rw this,
  apply is_open_Union,
  intro x,
  apply is_open_Union,
  intro xA,
  have h := classical.some_spec (h1 x xA),
  exact h.2.1
end

theorem exercise_13_3b : ¬ ∀ X : Type, ∀s : set (set X),
  (∀ t : set X, t ∈ s → (set.infinite tᶜ ∨ t = ∅ ∨ t = ⊤)) → 
  (set.infinite (⋃₀ s)ᶜ ∨ (⋃₀ s) = ∅ ∨ (⋃₀ s) = ⊤) :=
sorry

def is_topology (X : Type*) (T : set (set X)) :=
  univ ∈ T ∧
  (∀ s t, s ∈ T → t ∈ T → s ∩ t ∈ T) ∧
  (∀s, (∀t ∈ s, t ∈ T) → ⋃₀ s ∈ T)

theorem exercise_13_4a1 (X I : Type*) (T : I → set (set X)) (h : ∀ i, is_topology X (T i)) :
  is_topology X (⋂ i : I, T i) :=
sorry

theorem exercise_13_4a2 :
  ∃ (X I : Type*) (T : I → set (set X)),
  (∀ i, is_topology X (T i)) ∧ ¬  is_topology X (⋂ i : I, T i) :=
sorry

theorem exercise_13_4b1 (X I : Type*) (T : I → set (set X)) (h : ∀ i, is_topology X (T i)) :
  ∃! T', is_topology X T' ∧ (∀ i, T i ⊆ T') ∧
  ∀ T'', is_topology X T'' → (∀ i, T i ⊆ T'') → T'' ⊆ T' :=
sorry

theorem exercise_13_4b2 (X I : Type*) (T : I → set (set X)) (h : ∀ i, is_topology X (T i)) :
  ∃! T', is_topology X T' ∧ (∀ i, T' ⊆ T i) ∧
  ∀ T'', is_topology X T'' → (∀ i, T'' ⊆ T i) → T' ⊆ T'' :=
sorry

theorem exercise_13_5a {X : Type*}
  [topological_space X] (A : set (set X)) (hA : is_topological_basis A) :
  generate_from A = generate_from (sInter {T | is_topology X T ∧ A ⊆ T}) :=
sorry

theorem exercise_13_5b {X : Type*}
  [t : topological_space X] (A : set (set X)) (hA : t = generate_from A) :
  generate_from A = generate_from (sInter {T | is_topology X T ∧ A ⊆ T}) :=
sorry

def lower_limit_topology (X : Type) [preorder X] :=
  topological_space.generate_from {S : set X | ∃ a b, a < b ∧ S = Ico a b}

def Rl := lower_limit_topology ℝ

def K : set ℝ := {r | ∃ n : ℕ, r = 1 / n}

def K_topology := topological_space.generate_from
  ({S : set ℝ | ∃ a b, a < b ∧ S = Ioo a b} ∪ {S : set ℝ | ∃ a b, a < b ∧ S = Ioo a b \ K})

theorem exercise_13_6 :
  ¬ (∀ U, Rl.is_open U → K_topology.is_open U) ∧ ¬ (∀ U, K_topology.is_open U → Rl.is_open U) :=
sorry

theorem exercise_13_8a :
  topological_space.is_topological_basis {S : set ℝ | ∃ a b : ℚ, a < b ∧ S = Ioo a b} :=
sorry

theorem exercise_13_8b :
  (topological_space.generate_from {S : set ℝ | ∃ a b : ℚ, a < b ∧ S = Ico a b}).is_open ≠
  (lower_limit_topology ℝ).is_open :=
sorry

theorem exercise_16_1 {X : Type*} [topological_space X]
  (Y : set X)
  (A : set Y) :
  ∀ U : set A, is_open U ↔ is_open (subtype.val '' U) :=
sorry

theorem exercise_16_4 {X Y : Type*} [topological_space X] [topological_space Y]
  (π₁ : X × Y → X)
  (π₂ : X × Y → Y)
  (h₁ : π₁ = prod.fst)
  (h₂ : π₂ = prod.snd) :
  is_open_map π₁ ∧ is_open_map π₂ :=
sorry

def rational (x : ℝ) := x ∈ set.range (coe : ℚ → ℝ)

theorem exercise_16_6
  (S : set (set (ℝ × ℝ)))
  (hS : ∀ s, s ∈ S → ∃ a b c d, (rational a ∧ rational b ∧ rational c ∧ rational d
  ∧ s = {x | ∃ x₁ x₂, x = (x₁, x₂) ∧ a < x₁ ∧ x₁ < b ∧ c < x₂ ∧ x₂ < d})) :
  is_topological_basis S :=
sorry

theorem exercise_17_4 {X : Type*} [topological_space X]
  (U A : set X) (hU : is_open U) (hA : is_closed A) :
  is_open (U \ A) ∧ is_closed (A \ U) :=
sorry

theorem exercise_18_8a {X Y : Type*} [topological_space X] [topological_space Y]
  [linear_order Y] [order_topology Y] {f g : X → Y}
  (hf : continuous f) (hg : continuous g) :
  is_closed {x | f x ≤ g x} :=
sorry

theorem exercise_18_8b {X Y : Type*} [topological_space X] [topological_space Y]
  [linear_order Y] [order_topology Y] {f g : X → Y}
  (hf : continuous f) (hg : continuous g) :
  continuous (λ x, min (f x) (g x)) :=
sorry

theorem exercise_18_13
  {X : Type*} [topological_space X] {Y : Type*} [topological_space Y]
  [t2_space Y] {A : set X} {f : A → Y} (hf : continuous f)
  (g : closure A → Y)
  (g_con : continuous g) :
  ∀ (g' : closure A → Y), continuous g' →  (∀ (x : closure A), g x = g' x) :=
sorry

theorem exercise_19_6a
  {n : ℕ}
  {f : fin n → Type*} {x : ℕ → Πa, f a}
  (y : Πi, f i)
  [Πa, topological_space (f a)] :
  tendsto x at_top (𝓝 y) ↔ ∀ i, tendsto (λ j, (x j) i) at_top (𝓝 (y i)) :=
sorry

theorem exercise_20_2
  [topological_space (ℝ ×ₗ ℝ)] [order_topology (ℝ ×ₗ ℝ)]
  : metrizable_space (ℝ ×ₗ ℝ) :=
sorry

abbreviation I : set ℝ := set.Icc 0 1

theorem exercise_21_6a
  (f : ℕ → I → ℝ )
  (h : ∀ x n, f n x = x ^ n) :
  ∀ x, ∃ y, tendsto (λ n, f n x) at_top (𝓝 y) :=
sorry

theorem exercise_21_6b
  (f : ℕ → I → ℝ )
  (h : ∀ x n, f n x = x ^ n) :
  ¬ ∃ f₀, tendsto_uniformly f f₀ at_top :=
sorry

theorem exercise_21_8
  {X : Type*} [topological_space X] {Y : Type*} [metric_space Y]
  {f : ℕ → X → Y} {x : ℕ → X}
  (hf : ∀ n, continuous (f n))
  (x₀ : X)
  (hx : tendsto x at_top (𝓝 x₀))
  (f₀ : X → Y)
  (hh : tendsto_uniformly f f₀ at_top) :
  tendsto (λ n, f n (x n)) at_top (𝓝 (f₀ x₀)) :=
sorry

theorem exercise_22_2a {X Y : Type*} [topological_space X]
  [topological_space Y] (p : X → Y) (h : continuous p) :
  quotient_map p ↔ ∃ (f : Y → X), continuous f ∧ p ∘ f = id :=
sorry

theorem exercise_22_2b {X : Type*} [topological_space X]
  {A : set X} (r : X → A) (hr : continuous r) (h : ∀ x : A, r x = x) :
  quotient_map r :=
sorry

theorem exercise_22_5 {X Y : Type*} [topological_space X]
  [topological_space Y] (p : X → Y) (hp : is_open_map p)
  (A : set X) (hA : is_open A) : is_open_map (p ∘ subtype.val : A → Y) :=
sorry

theorem exercise_23_2 {X : Type*}
  [topological_space X] {A : ℕ → set X} (hA : ∀ n, is_connected (A n))
  (hAn : ∀ n, A n ∩ A (n + 1) ≠ ∅) :
  is_connected (⋃ n, A n) :=
sorry

theorem exercise_23_3 {X : Type*} [topological_space X]
  [topological_space X] {A : ℕ → set X}
  (hAn : ∀ n, is_connected (A n))
  (A₀ : set X)
  (hA : is_connected A₀)
  (h : ∀ n, A₀ ∩ A n ≠ ∅) :
  is_connected (A₀ ∪ (⋃ n, A n)) :=
sorry

theorem exercise_23_4 {X : Type*} [topological_space X] [cofinite_topology X]
  (s : set X) : set.infinite s → is_connected s :=
sorry

theorem exercise_23_6 {X : Type*}
  [topological_space X] {A C : set X} (hc : is_connected C)
  (hCA : C ∩ A ≠ ∅) (hCXA : C ∩ Aᶜ ≠ ∅) :
  C ∩ (frontier A) ≠ ∅ :=
sorry

theorem exercise_23_9 {X Y : Type*}
  [topological_space X] [topological_space Y]
  (A₁ A₂ : set X)
  (B₁ B₂ : set Y)
  (hA : A₁ ⊂ A₂)
  (hB : B₁ ⊂ B₂)
  (hA : is_connected A₂)
  (hB : is_connected B₂) :
  is_connected ({x | ∃ a b, x = (a, b) ∧ a ∈ A₂ ∧ b ∈ B₂} \
      {x | ∃ a b, x = (a, b) ∧ a ∈ A₁ ∧ b ∈ B₁}) :=
sorry

theorem exercise_23_11 {X Y : Type*} [topological_space X] [topological_space Y]
  (p : X → Y) (hq : quotient_map p)
  (hY : connected_space Y) (hX : ∀ y : Y, is_connected (p ⁻¹' {y})) :
  connected_space X :=
sorry

theorem exercise_24_2 {f : (metric.sphere 0 1 : set ℝ) → ℝ}
  (hf : continuous f) : ∃ x, f x = f (-x) :=
sorry

theorem exercise_24_3a [topological_space I] [compact_space I]
  (f : I → I) (hf : continuous f) :
  ∃ (x : I), f x = x :=
sorry

theorem exercise_25_4 {X : Type*} [topological_space X]
  [loc_path_connected_space X] (U : set X) (hU : is_open U)
  (hcU : is_connected U) : is_path_connected U :=
sorry

theorem exercise_25_9 {G : Type*} [topological_space G] [group G]
  [topological_group G] (C : set G) (h : C = connected_component 1) :
  is_normal_subgroup C :=
sorry

theorem exercise_26_11
  {X : Type*} [topological_space X] [compact_space X] [t2_space X]
  (A : set (set X)) (hA : ∀ (a b : set X), a ∈ A → b ∈ A → a ⊆ b ∨ b ⊆ a)
  (hA' : ∀ a ∈ A, is_closed a) (hA'' : ∀ a ∈ A, is_connected a) :
  is_connected (⋂₀ A) :=
sorry

theorem exercise_26_12 {X Y : Type*} [topological_space X] [topological_space Y]
  (p : X → Y) (h : function.surjective p) (hc : continuous p) (hp : ∀ y, is_compact (p ⁻¹' {y}))
  (hY : compact_space Y) : compact_space X :=
sorry

theorem exercise_27_4
  {X : Type*} [metric_space X] [connected_space X] (hX : ∃ x y : X, x ≠ y) :
  ¬ countable (univ : set X) :=
sorry

def countably_compact (X : Type*) [topological_space X] :=
  ∀ U : ℕ → set X,
  (∀ i, is_open (U i)) ∧ ((univ : set X) ⊆ ⋃ i, U i) →
  (∃ t : finset ℕ, (univ : set X) ⊆ ⋃ i ∈ t, U i)

def limit_point_compact (X : Type*) [topological_space X] :=
  ∀ U : set X, set.infinite U → ∃ x ∈ U, cluster_pt x (𝓟 U)

theorem exercise_28_4 {X : Type*}
  [topological_space X] (hT1 : t1_space X) :
  countably_compact X ↔ limit_point_compact X :=
sorry

theorem exercise_28_5
  (X : Type*) [topological_space X] :
  countably_compact X ↔ ∀ (C : ℕ → set X), (∀ n, is_closed (C n)) ∧
  (∀ n, C n ≠ ∅) ∧ (∀ n, C n ⊆ C (n + 1)) → ∃ x, ∀ n, x ∈ C n :=
sorry

theorem exercise_28_6 {X : Type*} [metric_space X]
  [compact_space X] {f : X → X} (hf : isometry f) :
  function.bijective f :=
sorry

theorem exercise_29_1 : ¬ locally_compact_space ℚ :=
sorry

theorem exercise_29_4 [topological_space (ℕ → I)] :
  ¬ locally_compact_space (ℕ → I) :=
sorry 

theorem exercise_29_10 {X : Type*}
  [topological_space X] [t2_space X] (x : X)
  (hx : ∃ U : set X, x ∈ U ∧ is_open U ∧ (∃ K : set X, U ⊂ K ∧ is_compact K))
  (U : set X) (hU : is_open U) (hxU : x ∈ U) :
  ∃ (V : set X), is_open V ∧ x ∈ V ∧ is_compact (closure V) ∧ closure V ⊆ U :=
sorry

theorem exercise_30_10
  {X : ℕ → Type*} [∀ i, topological_space (X i)]
  (h : ∀ i, ∃ (s : set (X i)), countable s ∧ dense s) :
  ∃ (s : set (Π i, X i)), countable s ∧ dense s :=
sorry

theorem exercise_30_13 {X : Type*} [topological_space X]
  (h : ∃ (s : set X), countable s ∧ dense s) (U : set (set X))
  (hU : ∀ (x y : set X), x ∈ U → y ∈ U → x ≠ y → x ∩ y = ∅) :
  countable U :=
sorry

theorem exercise_31_1 {X : Type*} [topological_space X]
  (hX : regular_space X) (x y : X) :
  ∃ (U V : set X), is_open U ∧ is_open V ∧ x ∈ U ∧ y ∈ V ∧ closure U ∩ closure V = ∅ :=
sorry

theorem exercise_31_2 {X : Type*}
  [topological_space X] [normal_space X] {A B : set X}
  (hA : is_closed A) (hB : is_closed B) (hAB : disjoint A B) :
  ∃ (U V : set X), is_open U ∧ is_open V ∧ A ⊆ U ∧ B ⊆ V ∧ closure U ∩ closure V = ∅ :=
sorry

theorem exercise_31_3 {α : Type*} [partial_order α]
  [topological_space α] (h : order_topology α) : regular_space α :=
sorry

theorem exercise_32_1 {X : Type*} [topological_space X]
  (hX : normal_space X) (A : set X) (hA : is_closed A) :
  normal_space {x // x ∈ A} :=
sorry

theorem exercise_32_2a
  {ι : Type*} {X : ι → Type*} [∀ i, topological_space (X i)]
  (h : ∀ i, nonempty (X i)) (h2 : t2_space (Π i, X i)) :
  ∀ i, t2_space (X i) :=
sorry

theorem exercise_32_2b
  {ι : Type*} {X : ι → Type*} [∀ i, topological_space (X i)]
  (h : ∀ i, nonempty (X i)) (h2 : regular_space (Π i, X i)) :
  ∀ i, regular_space (X i) :=
sorry

theorem exercise_32_2c
  {ι : Type*} {X : ι → Type*} [∀ i, topological_space (X i)]
  (h : ∀ i, nonempty (X i)) (h2 : normal_space (Π i, X i)) :
  ∀ i, normal_space (X i) :=
sorry

theorem exercise_32_3 {X : Type*} [topological_space X]
  (hX : locally_compact_space X) (hX' : t2_space X) :
  regular_space X :=
sorry

theorem exercise_33_7 {X : Type*} [topological_space X]
  (hX : locally_compact_space X) (hX' : t2_space X) :
  ∀ x A, is_closed A ∧ ¬ x ∈ A →
  ∃ (f : X → I), continuous f ∧ f x = 1 ∧ f '' A = {0}
  :=
sorry

theorem exercise_33_8
  (X : Type*) [topological_space X] [regular_space X]
  (h : ∀ x A, is_closed A ∧ ¬ x ∈ A →
  ∃ (f : X → I), continuous f ∧ f x = (1 : I) ∧ f '' A = {0})
  (A B : set X) (hA : is_closed A) (hB : is_closed B)
  (hAB : disjoint A B)
  (hAc : is_compact A) :
  ∃ (f : X → I), continuous f ∧ f '' A = {0} ∧ f '' B = {1} :=
sorry

theorem exercise_34_9
  (X : Type*) [topological_space X] [compact_space X]
  (X1 X2 : set X) (hX1 : is_closed X1) (hX2 : is_closed X2)
  (hX : X1 ∪ X2 = univ) (hX1m : metrizable_space X1)
  (hX2m : metrizable_space X2) : metrizable_space X :=
sorry

theorem exercise_38_6 {X : Type*}
  (X : Type*) [topological_space X] [regular_space X]
  (h : ∀ x A, is_closed A ∧ ¬ x ∈ A →
  ∃ (f : X → I), continuous f ∧ f x = (1 : I) ∧ f '' A = {0}) :
  is_connected (univ : set X) ↔ is_connected (univ : set (stone_cech X)) :=
sorry

theorem exercise_43_2 {X : Type*} [metric_space X]
  {Y : Type*} [metric_space Y] [complete_space Y] (A : set X)
  (f : X → Y) (hf : uniform_continuous_on f A) :
  ∃! (g : X → Y), continuous_on g (closure A) ∧
  uniform_continuous_on g (closure A) ∧ ∀ (x : A), g x = f x :=
sorry

theorem exercise_2_12a (f : ℕ → ℕ) (p : ℕ → ℝ) (a : ℝ)
  (hf : injective f) (hp : tendsto p at_top (𝓝 a)) :
  tendsto (λ n, p (f n)) at_top (𝓝 a) :=
sorry

theorem exercise_2_26 {M : Type*} [topological_space M]
  (U : set M) : is_open U ↔ ∀ x ∈ U, ¬ cluster_pt x (𝓟 Uᶜ) :=
sorry

theorem exercise_2_29 (M : Type*) [metric_space M]
  (O C : set (set M))
  (hO : O = {s | is_open s})
  (hC : C = {s | is_closed s}) :
  ∃ f : O → C, bijective f :=
sorry

theorem exercise_2_32a (A : set ℕ) : is_clopen A :=
sorry

theorem exercise_2_41 (m : ℕ) {X : Type*} [normed_space ℝ ((fin m) → ℝ)] :
  is_compact (metric.closed_ball 0 1) :=
sorry

theorem exercise_2_46 {M : Type*} [metric_space M]
  {A B : set M} (hA : is_compact A) (hB : is_compact B)
  (hAB : disjoint A B) (hA₀ : A ≠ ∅) (hB₀ : B ≠ ∅) :
  ∃ a₀ b₀, a₀ ∈ A ∧ b₀ ∈ B ∧ ∀ (a : M) (b : M),
  a ∈ A → b ∈ B → dist a₀ b₀ ≤ dist a b :=
sorry

theorem exercise_2_57 {X : Type*} [topological_space X]
  : ∃ (S : set X), is_connected S ∧ ¬ is_connected (interior S) :=
sorry

theorem exercise_2_92 {α : Type*} [topological_space α]
  {s : ℕ → set α}
  (hs : ∀ i, is_compact (s i))
  (hs : ∀ i, (s i).nonempty)
  (hs : ∀ i, (s i) ⊃ (s (i + 1))) :
  (⋂ i, s i).nonempty :=
sorry

theorem exercise_2_126 {E : set ℝ}
  (hE : ¬ set.countable E) : ∃ (p : ℝ), cluster_pt p (𝓟 E) :=
sorry

theorem exercise_3_1 {f : ℝ → ℝ}
  (hf : ∀ x y, |f x - f y| ≤ |x - y| ^ 2) :
  ∃ c, f = λ x, c :=
sorry

theorem exercise_3_4 (n : ℕ) :
  tendsto (λ n, (sqrt (n + 1) - sqrt n)) at_top (𝓝 0) :=
sorry

theorem exercise_3_63a (p : ℝ) (f : ℕ → ℝ) (hp : p > 1)
  (h : f = λ k, (1 : ℝ) / (k * (log k) ^ p)) :
  ∃ l, tendsto f at_top (𝓝 l) :=
sorry

theorem exercise_3_63b (p : ℝ) (f : ℕ → ℝ) (hp : p ≤ 1)
  (h : f = λ k, (1 : ℝ) / (k * (log k) ^ p)) :
  ¬ ∃ l, tendsto f at_top (𝓝 l) :=
sorry

theorem exercise_4_15a {α : Type*}
  (a b : ℝ) (F : set (ℝ → ℝ)) :
  (∀ (x : ℝ) (ε > 0), ∃ (U ∈ (𝓝 x)),
  (∀ (y z ∈ U) (f : ℝ → ℝ), f ∈ F → (dist (f y) (f z) < ε)))
  ↔
  ∃ (μ : ℝ → ℝ), ∀ (x : ℝ), (0 : ℝ) ≤ μ x ∧ tendsto μ (𝓝 0) (𝓝 0) ∧
  (∀ (s t : ℝ) (f : ℝ → ℝ), f ∈ F → |(f s) - (f t)| ≤ μ (|s - t|)) :=
sorry

theorem exercise_2020_b5 (z : fin 4 → ℂ) (hz0 : ∀ n, ‖z n‖ < 1) 
  (hz1 : ∀ n : fin 4, z n ≠ 1) : 
  3 - z 0 - z 1 - z 2 - z 3 + (z 0) * (z 1) * (z 2) * (z 3) ≠ 0 := 
sorry 

theorem exercise_2018_a5 (f : ℝ → ℝ) (hf : cont_diff ℝ ⊤ f)
  (hf0 : f 0 = 0) (hf1 : f 1 = 1) (hf2 : ∀ x, f x ≥ 0) :
  ∃ (n : ℕ) (x : ℝ), iterated_deriv n f x = 0 := 
sorry 

theorem exercise_2018_b2 (n : ℕ) (hn : n > 0) (f : ℕ → ℂ → ℂ) 
  (hf : ∀ n : ℕ, f n = λ z, (∑ (i : fin n), (n-i)* z^(i : ℕ))) : 
  ¬ (∃ z : ℂ, ‖z‖ ≤ 1 ∧ f n z = 0) :=
sorry 

theorem exercise_2018_b4 (a : ℝ) (x : ℕ → ℝ) (hx0 : x 0 = a)
  (hx1 : x 1 = a) 
  (hxn : ∀ n : ℕ, n ≥ 2 → x (n+1) = 2*(x n)*(x (n-1)) - x (n-2))
  (h : ∃ n, x n = 0) : 
  ∃ c, function.periodic x c :=
sorry 

theorem exercise_2017_b3 (f : ℝ → ℝ) (c : ℕ → ℝ)
  (hf : f = λ x, (∑' (i : ℕ), (c i) * x^i)) 
  (hc : ∀ n, c n = 0 ∨ c n = 1)
  (hf1 : f (2/3) = 3/2) : 
  irrational (f (1/2)) :=
sorry 

theorem exercise_2014_a5 (P : ℕ → polynomial ℤ) 
  (hP : ∀ n, P n = ∑ (i : fin n), (n+1) * X ^ n) : 
  ∀ (j k : ℕ), j ≠ k → is_coprime (P j) (P k) :=
sorry 

theorem exercise_2010_a4 (n : ℕ) : 
  ¬ nat.prime (10^10^10^n + 10^10^n + 10^n - 1) :=
sorry 

theorem exercise_2001_a5 : 
  ∃! a n : ℕ, a > 0 ∧ n > 0 ∧ a^(n+1) - (a+1)^n = 2001 :=
sorry 

theorem exercise_2000_a2 : 
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ ∃ i : fin 6 → ℕ, n = (i 0)^2 + (i 1)^2 ∧ 
  n + 1 = (i 2)^2 + (i 3)^2 ∧ n + 2 = (i 4)^2 + (i 5)^2 :=
sorry 

theorem exercise_1999_b4 (f : ℝ → ℝ) (hf: cont_diff ℝ 3 f) 
  (hf1 : ∀ (n ≤ 3) (x : ℝ), iterated_deriv n f x > 0) 
  (hf2 : ∀ x : ℝ, iterated_deriv 3 f x ≤ f x) : 
  ∀ x : ℝ, deriv f x < 2 * f x :=
sorry 

theorem exercise_1998_a3 (f : ℝ → ℝ) (hf : cont_diff ℝ 3 f) : 
  ∃ a : ℝ, (f a) * (deriv f a) * (iterated_deriv 2 f a) * (iterated_deriv 3 f a) ≥ 0 :=
sorry 

theorem exercise_1998_b6 (a b c : ℤ) : 
  ∃ n : ℤ, n > 0 ∧ ¬ ∃ m : ℤ, sqrt (n^3 + a*n^2 + b*n + c) = m :=
sorry 

theorem exercise_1_1a
  (x : ℝ) (y : ℚ) :
  ( irrational x ) -> irrational ( x + y ) :=
begin
  apply irrational.add_rat,
end

theorem exercise_1_1b
(x : ℝ)
(y : ℚ)
(h : y ≠ 0)
: ( irrational x ) -> irrational ( x * y ) :=
begin
  intro g,
  apply irrational.mul_rat g h,
end

theorem exercise_1_2 : ¬ ∃ (x : ℚ), ( x ^ 2 = 12 ) :=
sorry

theorem exercise_1_4
(α : Type*) [partial_order α]
(s : set α)
(x y : α)
(h₀ : set.nonempty s)
(h₁ : x ∈ lower_bounds s)
(h₂ : y ∈ upper_bounds s)
: x ≤ y :=
begin
  have h : ∃ z, z ∈ s := h₀,
  cases h with z,
  have xlez : x ≤ z :=
  begin
  apply h₁,
  assumption,
  end,
  have zley : z ≤ y :=
  begin
  apply h₂,
  assumption,
  end,
  exact xlez.trans zley,
end

theorem exercise_1_5 (A minus_A : set ℝ) (hA : A.nonempty)
  (hA_bdd_below : bdd_below A) (hminus_A : minus_A = {x | -x ∈ A}) :
  Inf A = Sup minus_A :=
sorry

theorem exercise_1_8 : ¬ ∃ (r : ℂ → ℂ → Prop), is_linear_order ℂ r :=
  sorry

theorem exercise_1_11a (z : ℂ) :
  ∃ (r : ℝ) (w : ℂ), abs w = 1 ∧ z = r * w :=
begin
  by_cases h : z = 0,
  {
  use [0, 1],
  simp,
  assumption,
  },
  {
  use abs z,
  use z / ↑(abs z),
  split,
  {
  simp,
  field_simp [h],
  },
  {
  field_simp [h],
  apply mul_comm,
  },
  },
end

theorem exercise_1_12 (n : ℕ) (f : ℕ → ℂ) :
  abs (∑ i in finset.range n, f i) ≤ ∑ i in finset.range n, abs (f i) :=
sorry

theorem exercise_1_13 (x y : ℂ) :
  |(abs x) - (abs y)| ≤ abs (x - y) :=
sorry

theorem exercise_1_14
  (z : ℂ) (h : abs z = 1)
  : (abs (1 + z)) ^ 2 + (abs (1 - z)) ^ 2 = 4 :=
sorry

theorem exercise_1_16a
  (n : ℕ)
  (d r : ℝ)
  (x y z : euclidean_space ℝ (fin n)) -- R^n
  (h₁ : n ≥ 3)
  (h₂ : ‖x - y‖ = d)
  (h₃ : d > 0)
  (h₄ : r > 0)
  (h₅ : 2 * r > d)
  : set.infinite {z : euclidean_space ℝ (fin n) | ‖z - x‖ = r ∧ ‖z - y‖ = r} :=
sorry

theorem exercise_1_17
  (n : ℕ)
  (x y : euclidean_space ℝ (fin n)) -- R^n
  : ‖x + y‖^2 + ‖x - y‖^2 = 2*‖x‖^2 + 2*‖y‖^2 :=
sorry

theorem exercise_1_18a
  (n : ℕ)
  (h : n > 1)
  (x : euclidean_space ℝ (fin n)) -- R^n
  : ∃ (y : euclidean_space ℝ (fin n)), y ≠ 0 ∧ (inner x y) = (0 : ℝ) :=
sorry

theorem exercise_1_18b
  : ¬ ∀ (x : ℝ), ∃ (y : ℝ), y ≠ 0 ∧ x * y = 0 :=
begin
  simp,
  use 1,
  intros x h₁ h₂,
  cases h₂,
  {norm_num at h₂},
  {exact absurd h₂ h₁},
end

theorem exercise_1_19
  (n : ℕ)
  (a b c x : euclidean_space ℝ (fin n))
  (r : ℝ)
  (h₁ : r > 0)
  (h₂ : 3 • c = 4 • b - a)
  (h₃ : 3 * r = 2 * ‖x - b‖)
  : ‖x - a‖ = 2 * ‖x - b‖ ↔ ‖x - c‖ = r :=
sorry

theorem exercise_2_19a {X : Type*} [metric_space X]
  (A B : set X) (hA : is_closed A) (hB : is_closed B) (hAB : disjoint A B) :
  separated_nhds A B :=
sorry

theorem exercise_2_24 {X : Type*} [metric_space X]
  (hX : ∀ (A : set X), infinite A → ∃ (x : X), x ∈ closure A) :
  separable_space X :=
sorry

theorem exercise_2_25 {K : Type*} [metric_space K] [compact_space K] :
  ∃ (B : set (set K)), set.countable B ∧ is_topological_basis B :=
sorry

theorem exercise_2_27a (k : ℕ) (E P : set (euclidean_space ℝ (fin k)))
  (hE : E.nonempty ∧ ¬ set.countable E)
  (hP : P = {x | ∀ U ∈ 𝓝 x, ¬ set.countable (P ∩ E)}) :
  is_closed P ∧ P = {x | cluster_pt x (𝓟 P)}  :=
sorry

theorem exercise_2_27b (k : ℕ) (E P : set (euclidean_space ℝ (fin k)))
  (hE : E.nonempty ∧ ¬ set.countable E)
  (hP : P = {x | ∀ U ∈ 𝓝 x, (P ∩ E).nonempty ∧ ¬ set.countable (P ∩ E)}) :
  set.countable (E \ P) :=
sorry

theorem exercise_2_28 (X : Type*) [metric_space X] [separable_space X]
  (A : set X) (hA : is_closed A) :
  ∃ P₁ P₂ : set X, A = P₁ ∪ P₂ ∧
  is_closed P₁ ∧ P₁ = {x | cluster_pt x (𝓟 P₁)} ∧
  set.countable P₂ :=
sorry

theorem exercise_2_29 (U : set ℝ) (hU : is_open U) :
  ∃ (f : ℕ → set ℝ), (∀ n, ∃ a b : ℝ, f n = {x | a < x ∧ x < b}) ∧ (∀ n, f n ⊆ U) ∧
  (∀ n m, n ≠ m → f n ∩ f m = ∅) ∧
  U = ⋃ n, f n :=
sorry

theorem exercise_3_1a
  (f : ℕ → ℝ)
  (h : ∃ (a : ℝ), tendsto (λ (n : ℕ), f n) at_top (𝓝 a))
  : ∃ (a : ℝ), tendsto (λ (n : ℕ), |f n|) at_top (𝓝 a) :=
begin
  cases h with a h,
  use |a|,
  apply filter.tendsto.abs h,
end

theorem exercise_3_2a
  : tendsto (λ (n : ℝ), (sqrt (n^2 + n) - n)) at_top (𝓝 (1/2)) :=
begin
  have h : ∀ (n : ℝ), n > 0 → sqrt (n^2 + n) - n = 1 / (sqrt (1 + 1 / n) + 1) :=
  begin
  intro n,
  intro h,
  have h₁ : sqrt (n^2 + n) + n ≠ 0 := by {intro h₁, simp at *, rw ←h₁ at h, simp at h,
    have : sqrt (n ^ 2 + n) ≥ 0 := sqrt_nonneg (n ^ 2 + n), linarith,},
  have h₂ : sqrt (n^2 + n) + n = sqrt (n^2 + n) + n := by refl,
  have h₃ : n ≥ 0 := by linarith,
  have h₄ : n ≠ 0 := by linarith,
  have h₅ : n^2 + n ≥ 0 := by {simp, transitivity, apply h₃, simp, apply sq_nonneg},
  calc  _ = (sqrt (n^2 + n) - n) * 1 : by rw mul_one _
  ... = (sqrt (n^2 + n) - n) * ((sqrt (n^2 + n) + n) /
          (sqrt (n^2 + n) + n)) : by rw ←((div_eq_one_iff_eq h₁).2 h₂)
  ... = n / (sqrt (n^2 + n) + n) : by {field_simp, ring, sorry}
  ... = 1 / (sqrt (n^2 + n) / sqrt (n^2) + n / sqrt (n^2)) : by {field_simp, simp [sqrt_sq h₃]}
  ... = 1 / (sqrt (n^2 + n) / sqrt (n^2) + 1) : by simp [sqrt_sq h₃, (div_eq_one_iff_eq h₄).2]
  ... = 1 / (sqrt (1 + n / (n ^ 2)) + 1): by {rw ←(sqrt_div h₅ (n^2)), field_simp}
  ... = 1 / (sqrt (1 + 1 / n) + 1): by simp [pow_succ]
  end,
  refine (tendsto_congr' _).mp _,
  exact λ n, 1 / (sqrt (1 + 1 / n) + 1),
  refine eventually_at_top.mpr _,
  use 1,
  intros b bgt1, symmetry, apply h, linarith,
  have g : tendsto (λ (n : ℝ), 1 / n) at_top (𝓝 0) :=
  begin
  simp,
  apply tendsto_inv_at_top_zero,
  end,
  have h : tendsto (λ (n : ℝ), 1 / (sqrt (1 + n) + 1)) (𝓝 0) (𝓝 (1/2)) :=
  begin
  have : (1/2 : ℝ) = (λ (n : ℝ), 1 / (sqrt (1 + n) + 1)) 0 := by {simp, norm_num}, rw this,
  apply continuous_at.tendsto, simp,
  refine continuous_at.comp _ _, simp,
  refine continuous_at.add _ _,
  refine continuous_at.sqrt _, simp,
  refine continuous_at.add _ _,
  exact continuous_at_const,
  exact continuous_at_id,
  exact continuous_at_const,
  end,
  apply tendsto.comp h g,
end

noncomputable def f : ℕ → ℝ
| 0 := sqrt 2
| (n + 1) := sqrt (2 + sqrt (f n))

theorem exercise_3_3
  : ∃ (x : ℝ), tendsto f at_top (𝓝 x) ∧ ∀ n, f n < 2 :=
sorry

theorem exercise_3_5 -- TODO fix
  (a b : ℕ → ℝ)
  (h : limsup a + limsup b ≠ 0) :
  limsup (λ n, a n + b n) ≤ limsup a + limsup b :=
sorry

def g (n : ℕ) : ℝ := sqrt (n + 1) - sqrt n

theorem exercise_3_6a
: tendsto (λ (n : ℕ), (∑ i in finset.range n, g i)) at_top at_top :=
begin
  simp,
  have : (λ (n : ℕ), (∑ i in finset.range n, g i)) = (λ (n : ℕ), sqrt (n + 1)) := by sorry,
  rw this,
  apply tendsto_at_top_at_top_of_monotone,
  unfold monotone,
  intros a b a_le_b,
  apply sqrt_le_sqrt,
  simp, assumption,
  intro x,
  --use x ^ 2 - 1,
  --apply filter.tendsto.sqrt,
  sorry
end

theorem exercise_3_7
  (a : ℕ → ℝ)
  (h : ∃ y, (tendsto (λ n, (∑ i in (finset.range n), a i)) at_top (𝓝 y))) :
  ∃ y, tendsto (λ n, (∑ i in (finset.range n), sqrt (a i) / n)) at_top (𝓝 y) :=
sorry

theorem exercise_3_8
  (a b : ℕ → ℝ)
  (h1 : ∃ y, (tendsto (λ n, (∑ i in (finset.range n), a i)) at_top (𝓝 y)))
  (h2 : monotone b)
  (h3 : metric.bounded (set.range b)) :
  ∃ y, tendsto (λ n, (∑ i in (finset.range n), (a i) * (b i))) at_top (𝓝 y) :=
sorry

theorem exercise_3_13
  (a b : ℕ → ℝ)
  (ha : ∃ y, (tendsto (λ n, (∑ i in (finset.range n), |a i|)) at_top (𝓝 y)))
  (hb : ∃ y, (tendsto (λ n, (∑ i in (finset.range n), |b i|)) at_top (𝓝 y))) :
  ∃ y, (tendsto (λ n, (∑ i in (finset.range n),
  λ i, (∑ j in finset.range (i + 1), a j * b (i - j)))) at_top (𝓝 y)) :=
sorry

theorem exercise_3_20 {X : Type*} [metric_space X]
  (p : ℕ → X) (l : ℕ) (r : X)
  (hp : cauchy_seq p)
  (hpl : tendsto (λ n, p (l * n)) at_top (𝓝 r)) :
  tendsto p at_top (𝓝 r) :=
sorry

theorem exercise_3_21
  {X : Type*} [metric_space X] [complete_space X]
  (E : ℕ → set X)
  (hE : ∀ n, E n ⊃ E (n + 1))
  (hE' : tendsto (λ n, metric.diam (E n)) at_top (𝓝 0)) :
  ∃ a, set.Inter E = {a} :=
sorry

theorem exercise_3_22 (X : Type*) [metric_space X] [complete_space X]
  (G : ℕ → set X) (hG : ∀ n, is_open (G n) ∧ dense (G n)) :
  ∃ x, ∀ n, x ∈ G n :=
sorry

theorem exercise_4_1a
  : ∃ (f : ℝ → ℝ), (∀ (x : ℝ), tendsto (λ y, f(x + y) - f(x - y)) (𝓝 0) (𝓝 0)) ∧ ¬ continuous f :=
begin
  let f := λ x : ℝ, if x = 0 then (1 : ℝ) else (0 : ℝ),
  use f, split,
  { intro x,
    suffices : (λ y, f (x + y) - f(x - y)) =ᶠ[𝓝 0] (λ y, 0),
    { simp [filter.tendsto_congr' this,  tendsto_const_nhds_iff] },
    by_cases h : x = 0,
    { dsimp [f], simp [h] },
    have : set.Ioo (-|x|) (|x|) ∈ 𝓝 (0 : ℝ),
    { apply Ioo_mem_nhds; simp [h], },
    apply eventually_of_mem this,
    intro y, simp, dsimp [f],
    intros h1 h2,
    rw [if_neg, if_neg]; simp [lt_abs, neg_lt] at *; cases h1; cases h2; linarith },
  simp [continuous_iff_continuous_at, continuous_at, tendsto_nhds],
  use [0, set.Ioo 0 2, is_open_Ioo], split,
  { dsimp [f], simp, norm_num },
  simp [mem_nhds_iff_exists_Ioo_subset],
  intros a b aneg bpos h,
  have : b / 2 ∈ set.Ioo a b,
  { simp, split; linarith },
  have := h this,
  simpa [f, (ne_of_lt bpos).symm] using this,
end

theorem exercise_4_2a
  {α : Type} [metric_space α]
  {β : Type} [metric_space β]
  (f : α → β)
  (h₁ : continuous f)
  : ∀ (x : set α), f '' (closure x) ⊆ closure (f '' x) :=
begin
  intros X x h₂ Y h₃,
  simp at *,
  cases h₃ with h₃ h₄,
  cases h₂ with w h₅,
  cases h₅ with h₅ h₆,
  have h₈ : is_closed (f ⁻¹' Y) := is_closed.preimage h₁ h₃,
  have h₉ : closure X ⊆ f ⁻¹' Y := closure_minimal h₄ h₈,
  rw ←h₆,
  exact h₉ h₅,
end

theorem exercise_4_3
  {α : Type} [metric_space α]
  (f : α → ℝ) (h : continuous f) (z : set α) (g : z = f⁻¹' {0})
  : is_closed z :=
begin
  rw g,
  apply is_closed.preimage h,
  exact is_closed_singleton,
end

theorem exercise_4_4a
  {α : Type} [metric_space α]
  {β : Type} [metric_space β]
  (f : α → β)
  (s : set α)
  (h₁ : continuous f)
  (h₂ : dense s)
  : f '' set.univ ⊆ closure (f '' s) :=
begin
  simp,
  exact continuous.range_subset_closure_image_dense h₁ h₂,
end

theorem exercise_4_4b
  {α : Type} [metric_space α]
  {β : Type} [metric_space β]
  (f g : α → β)
  (s : set α)
  (h₁ : continuous f)
  (h₂ : continuous g)
  (h₃ : dense s)
  (h₄ : ∀ x ∈ s, f x = g x)
  : f = g :=
begin
  have h₅ : is_closed {x | f x = g x} := is_closed_eq h₁ h₂,
  unfold dense at h₃,
  set t := {x : α | f x = g x} with h,
  have h₆ : s ⊆ t := h₄,
  have h₇ : closure s ⊆ closure t := closure_mono h₆,
  --have h₁₀ : closure s = set.univ := by { ext, simp, apply h₃,},
  --exact h₃, -- does not work ...
  have h₈ : ∀ x, x ∈ closure t := by { intro, apply h₇ (h₃ x), },
  have h₉ : closure t = t := closure_eq_iff_is_closed.2 h₅,
  rw h₉ at h₈,
  ext,
  exact h₈ x,
end

theorem exercise_4_5a
  (f : ℝ → ℝ)
  (E : set ℝ)
  (h₁ : is_closed E)
  (h₂ : continuous_on f E)
  : ∃ (g : ℝ → ℝ), continuous g ∧ ∀ x ∈ E, f x = g x :=
sorry

theorem exercise_4_5b
  : ∃ (E : set ℝ) (f : ℝ → ℝ), (continuous_on f E) ∧
  (¬ ∃ (g : ℝ → ℝ), continuous g ∧ ∀ x ∈ E, f x = g x) :=
begin
  set E : set ℝ := (set.Iio 0) ∪ (set.Ioi 0) with hE,
  let f : ℝ → ℝ := λ x, if x < 0 then 0 else 1,
  use E, use f,
  split,
  {
  refine continuous_on_iff.mpr _,
  intros x h₁ X h₂ h₃,
  by_cases h₄ : x < 0,
  {
  use set.Ioo (x - 1) 0,
  have h₅ : f x = 0 := if_pos h₄,
  split, exact is_open_Ioo,
  split,
  {
  have h₆ : x - 1 < x := by linarith,
  exact set.mem_sep h₆ h₄,
  },
  have h₆ : set.Ioo (x - 1) 0 ⊆ set.Iio 0 := set.Ioo_subset_Iio_self,
  have h₇ : set.Ioo (x - 1) 0 ∩ E = set.Ioo (x - 1) 0 := by {
  rw hE, simp,
  },
  rw h₇,
  have h₈ : (0 : ℝ) ∈ X := by {rw h₅ at h₃, exact h₃,},
  have h₉ : {(0 : ℝ)} ⊆ X := set.singleton_subset_iff.mpr h₈,
  have h₁₀ : set.Iio 0 ⊆ f ⁻¹' {0} := by {
  intros y hy,
  apply set.mem_preimage.2,
  have : f y = 0 := if_pos hy,
  rw this, simp,
  },
  have h₁₁ : f ⁻¹' {0} ⊆ f ⁻¹' X := set.preimage_mono h₉,
  have h₁₂ : set.Iio 0 ⊆ f ⁻¹' X := set.subset.trans h₁₀ h₁₁,
  exact set.subset.trans h₆ h₁₂,
  },
  {
  use set.Ioo 0 (x + 1),
  have h₄' : x > 0  := by {
  have : x ≠ 0 := by {rw hE at h₁, simp at h₁, exact h₁,},
  refine lt_of_le_of_ne _ this.symm,
  exact not_lt.mp h₄,
  },
  have h₅ : f x = 1 := if_neg h₄,
  split, exact is_open_Ioo,
  split,
  {
  have h₆ : x < x + 1:= by linarith,
  exact set.mem_sep h₄' h₆,
  },
  have h₆ : set.Ioo 0 (x + 1) ⊆ set.Ioi 0 := set.Ioo_subset_Ioi_self,
  have h₇ : set.Ioo 0 (x + 1) ∩ E = set.Ioo 0 (x + 1) := by {
  rw hE, simp,
  },
  rw h₇,
  have h₈ : (1 : ℝ) ∈ X := by {rw h₅ at h₃, exact h₃,},
  have h₉ : {(1 : ℝ)} ⊆ X := set.singleton_subset_iff.mpr h₈,
  have h₁₀ : set.Ioi 0 ⊆ f ⁻¹' {1} := by {
  intros y hy,
  have : y ∈ set.Ici (0 : ℝ) := set.mem_Ici_of_Ioi hy,
  have : ¬ y < 0 := asymm hy,
  apply set.mem_preimage.2,
  have : f y = 1 := if_neg this,
  rw this, simp,
  },
  have h₁₁ : f ⁻¹' {1} ⊆ f ⁻¹' X := set.preimage_mono h₉,
  have h₁₂ : set.Ioi 0 ⊆ f ⁻¹' X := set.subset.trans h₁₀ h₁₁,
  exact set.subset.trans h₆ h₁₂,
  },
  },
  {
  by_contradiction h₁,
  cases h₁ with g h₁,
  cases h₁ with h₁ h₂,
  have h₃ : continuous_at g 0 := continuous.continuous_at h₁,
  have h₄ := continuous_at.tendsto h₃,
  unfold tendsto at h₄,
  have h₅ := filter.le_def.1 h₄,
  simp at h₅,
  by_cases h₆ : g 0 > 0.5,
  {
  have h₇ : set.Ioi (0 : ℝ) ∈ 𝓝 (g 0) := by { refine Ioi_mem_nhds _, linarith,},
  have h₈ := h₅ (set.Ioi (0 : ℝ)) h₇,
  have h₉ : g ⁻¹' set.Ioi 0 = set.Ici 0 := by {
  ext,
  split,
  {
    intro h,
    simp at h,
    by_cases hw : x = 0,
    {rw hw, exact set.left_mem_Ici,},
    {
    have : x ∈ E := by {rw hE, simp, exact hw,},
    rw ←(h₂ x this) at h,
    by_contradiction hh,
    simp at hh,
    have : f x = 0 := if_pos hh,
    linarith,
    },
  },
  {
    intro h,
    simp,
    by_cases hw : x = 0,
    {rw hw, linarith,},
    {
    have h₉ : x > 0 := (ne.symm hw).le_iff_lt.mp h,
    have : x ∈ E := (set.Iio 0).mem_union_right h₉,
    rw ←(h₂ x this),
    have : ¬ x < 0 := asymm h₉,
    have : f x = 1 := if_neg this,
    linarith,
    },
  },
  },
  rw h₉ at h₈,
  have h₁₀ := interior_mem_nhds.2 h₈,
  simp at h₁₀,
  have := mem_of_mem_nhds h₁₀,
  simp at this,
  exact this,
  },
  {
  have h₇ : set.Iio (1 : ℝ) ∈ 𝓝 (g 0) := by { refine Iio_mem_nhds _, linarith, },
  have h₈ := h₅ (set.Iio (1 : ℝ)) h₇,
  have h₉ : g ⁻¹' set.Iio 1 = set.Iic 0 := by {
  ext,
  split,
  {
    intro h,
    simp at h,
    by_cases hw : x = 0,
    {simp [hw],},
    {
    have : x ∈ E := by {rw hE, simp, exact hw,},
    rw ←(h₂ x this) at h,
    by_contradiction hh,
    simp at hh,
    have : f x = 1 := if_neg ((by linarith) : ¬x < 0),
    linarith,
    },
  },
  {
    intro h,
    simp,
    by_cases hw : x = 0,
    {rw hw, linarith,},
    {
    have h₉ : x < 0 := (ne.le_iff_lt hw).mp h,
    have : x ∈ E := (set.Ioi 0).mem_union_left h₉,
    rw ←(h₂ x this),
    have : f x = 0 := if_pos h₉,
    linarith,
    },
  },
  },
  rw h₉ at h₈,
  have h₁₀ := interior_mem_nhds.2 h₈,
  simp at h₁₀,
  have := mem_of_mem_nhds h₁₀,
  simp at this,
  exact this,
  }
  }
end

theorem exercise_4_6
  (f : ℝ → ℝ)
  (E : set ℝ)
  (G : set (ℝ × ℝ))
  (h₁ : is_compact E)
  (h₂ : G = {(x, f x) | x ∈ E})
  : continuous_on f E ↔ is_compact G :=
sorry

theorem exercise_4_8a
  (E : set ℝ) (f : ℝ → ℝ) (hf : uniform_continuous_on f E)
  (hE : metric.bounded E) : metric.bounded (set.image f E) :=
sorry

theorem exercise_4_8b
  (E : set ℝ) :
  ∃ f : ℝ → ℝ, uniform_continuous_on f E ∧ ¬ metric.bounded (set.image f E) :=
sorry

theorem exercise_4_11a
  {X : Type*} [metric_space X]
  {Y : Type*} [metric_space Y]
  (f : X → Y) (hf : uniform_continuous f)
  (x : ℕ → X) (hx : cauchy_seq x) :
  cauchy_seq (λ n, f (x n)) :=
sorry

theorem exercise_4_12
  {α β γ : Type*} [uniform_space α] [uniform_space β] [uniform_space γ]
  {f : α → β} {g : β → γ}
  (hf : uniform_continuous f) (hg : uniform_continuous g) :
  uniform_continuous (g ∘ f) :=
sorry

theorem exercise_4_15 {f : ℝ → ℝ}
  (hf : continuous f) (hof : is_open_map f) :
  monotone f :=
sorry

theorem exercise_4_19
  {f : ℝ → ℝ} (hf : ∀ a b c, a < b → f a < c → c < f b → ∃ x, a < x ∧ x < b ∧ f x = c)
  (hg : ∀ r : ℚ, is_closed {x | f x = r}) : continuous f :=
sorry

theorem exercise_4_21a {X : Type*} [metric_space X]
  (K F : set X) (hK : is_compact K) (hF : is_closed F) (hKF : disjoint K F) :
  ∃ (δ : ℝ), δ > 0 ∧ ∀ (p q : X), p ∈ K → q ∈ F → dist p q ≥ δ :=
sorry

theorem exercise_4_24 {f : ℝ → ℝ}
  (hf : continuous f) (a b : ℝ) (hab : a < b)
  (h : ∀ x y : ℝ, a < x → x < b → a < y → y < b → f ((x + y) / 2) ≤ (f x + f y) / 2) :
  convex_on ℝ (set.Ioo a b) f :=
sorry

theorem exercise_5_1
  {f : ℝ → ℝ} (hf : ∀ x y : ℝ, | (f x - f y) | ≤ (x - y) ^ 2) :
  ∃ c, f = λ x, c :=
sorry

theorem exercise_5_2 {a b : ℝ}
  {f g : ℝ → ℝ} (hf : ∀ x ∈ set.Ioo a b, deriv f x > 0)
  (hg : g = f⁻¹)
  (hg_diff : differentiable_on ℝ g (set.Ioo a b)) :
  differentiable_on ℝ g (set.Ioo a b) ∧
  ∀ x ∈ set.Ioo a b, deriv g x = 1 / deriv f x :=
sorry

theorem exercise_5_3 {g : ℝ → ℝ} (hg : continuous g)
  (hg' : ∃ M : ℝ, ∀ x : ℝ, | deriv g x | ≤ M) :
  ∃ N, ∀ ε > 0, ε < N → function.injective (λ x : ℝ, x + ε * g x) :=
sorry

theorem exercise_5_4 {n : ℕ}
  (C : ℕ → ℝ)
  (hC : ∑ i in (finset.range (n + 1)), (C i) / (i + 1) = 0) :
  ∃ x, x ∈ (set.Icc (0 : ℝ) 1) ∧ ∑ i in finset.range (n + 1), (C i) * (x^i) = 0 :=
sorry

theorem exercise_5_5
  {f : ℝ → ℝ}
  (hfd : differentiable ℝ f)
  (hf : tendsto (deriv f) at_top (𝓝 0)) :
  tendsto (λ x, f (x + 1) - f x) at_top at_top :=
sorry

theorem exercise_5_6
  {f : ℝ → ℝ}
  (hf1 : continuous f)
  (hf2 : ∀ x, differentiable_at ℝ f x)
  (hf3 : f 0 = 0)
  (hf4 : monotone (deriv f)) :
  monotone_on (λ x, f x / x) (set.Ioi 0) :=
sorry

theorem exercise_5_7
  {f g : ℝ → ℝ} {x : ℝ}
  (hf' : differentiable_at ℝ f 0)
  (hg' : differentiable_at ℝ g 0)
  (hg'_ne_0 : deriv g 0 ≠ 0)
  (f0 : f 0 = 0) (g0 : g 0 = 0) :
  tendsto (λ x, f x / g x) (𝓝 x) (𝓝 (deriv f x / deriv g x)) :=
sorry

theorem exercise_5_15 {f : ℝ → ℝ} (a M0 M1 M2 : ℝ)
  (hf' : differentiable_on ℝ f (set.Ici a))
  (hf'' : differentiable_on ℝ (deriv f) (set.Ici a))
  (hM0 : M0 = Sup {(| f x | )| x ∈ (set.Ici a)})
  (hM1 : M1 = Sup {(| deriv f x | )| x ∈ (set.Ici a)})
  (hM2 : M2 = Sup {(| deriv (deriv f) x | )| x ∈ (set.Ici a)}) :
  (M1 ^ 2) ≤ 4 * M0 * M2 :=
sorry

theorem exercise_5_17
  {f : ℝ → ℝ}
  (hf' : differentiable_on ℝ f (set.Icc (-1) 1))
  (hf'' : differentiable_on ℝ (deriv f) (set.Icc 1 1))
  (hf''' : differentiable_on ℝ (deriv (deriv f)) (set.Icc 1 1))
  (hf0 : f (-1) = 0)
  (hf1 : f 0 = 0)
  (hf2 : f 1 = 1)
  (hf3 : deriv f 0 = 0) :
  ∃ x, x ∈ set.Ioo (-1 : ℝ) 1 ∧ deriv (deriv (deriv f)) x ≥ 3 :=
sorry

theorem exercise_1_13a {f : ℂ → ℂ} (Ω : set ℂ) (a b : Ω) (h : is_open Ω)
  (hf : differentiable_on ℂ f Ω) (hc : ∃ (c : ℝ), ∀ z ∈ Ω, (f z).re = c) :
  f a = f b :=
sorry

theorem exercise_1_13b {f : ℂ → ℂ} (Ω : set ℂ) (a b : Ω) (h : is_open Ω)
  (hf : differentiable_on ℂ f Ω) (hc : ∃ (c : ℝ), ∀ z ∈ Ω, (f z).im = c) :
  f a = f b :=
sorry

theorem exercise_1_13c {f : ℂ → ℂ} (Ω : set ℂ) (a b : Ω) (h : is_open Ω)
  (hf : differentiable_on ℂ f Ω) (hc : ∃ (c : ℝ), ∀ z ∈ Ω, abs (f z) = c) :
  f a = f b :=
sorry

theorem exercise_1_19a (z : ℂ) (hz : abs z = 1) (s : ℕ → ℂ)
    (h : s = (λ n, ∑ i in (finset.range n), i * z ^ i)) :
    ¬ ∃ y, tendsto s at_top (𝓝 y) :=
sorry

theorem exercise_1_19b (z : ℂ) (hz : abs z = 1) (s : ℕ → ℂ)
    (h : s = (λ n, ∑ i in (finset.range n), i * z / i ^ 2)) :
    ∃ y, tendsto s at_top (𝓝 y) :=
sorry

theorem exercise_1_19c (z : ℂ) (hz : abs z = 1) (hz2 : z ≠ 1) (s : ℕ → ℂ)
    (h : s = (λ n, ∑ i in (finset.range n), i * z / i)) :
    ∃ z, tendsto s at_top (𝓝 z) :=
sorry

theorem exercise_1_26
  (f F₁ F₂ : ℂ → ℂ) (Ω : set ℂ) (h1 : is_open Ω) (h2 : is_connected Ω)
  (hF₁ : differentiable_on ℂ F₁ Ω) (hF₂ : differentiable_on ℂ F₂ Ω)
  (hdF₁ : ∀ x ∈ Ω, deriv F₁ x = f x) (hdF₂ : ∀ x ∈ Ω, deriv F₂ x = f x)
  : ∃ c : ℂ, ∀ x, F₁ x = F₂ x + c :=
sorry

theorem exercise_2_2 :
  tendsto (λ y, ∫ x in 0..y, real.sin x / x) at_top (𝓝 (real.pi / 2)) :=
sorry

theorem exercise_2_9
  {f : ℂ → ℂ} (Ω : set ℂ) (b : metric.bounded Ω) (h : is_open Ω)
  (hf : differentiable_on ℂ f Ω) (z ∈ Ω) (hz : f z = z) (h'z : deriv f z = 1) :
  ∃ (f_lin : ℂ →L[ℂ] ℂ), ∀ x ∈ Ω, f x = f_lin x :=
sorry

theorem exercise_2_13 {f : ℂ → ℂ}
    (hf : ∀ z₀ : ℂ, ∃ (s : set ℂ) (c : ℕ → ℂ), is_open s ∧ z₀ ∈ s ∧
      ∀ z ∈ s, tendsto (λ n, ∑ i in finset.range n, (c i) * (z - z₀)^i) at_top (𝓝 (f z₀))
      ∧ ∃ i, c i = 0) :
    ∃ (c : ℕ → ℂ) (n : ℕ), f = λ z, ∑ i in finset.range n, (c i) * z ^ n :=
sorry


theorem exercise_3_3 (a : ℝ) (ha : 0 < a) :
    tendsto (λ y, ∫ x in -y..y, real.cos x / (x ^ 2 + a ^ 2))
    at_top (𝓝 (real.pi * (real.exp (-a) / a))) :=
sorry

theorem exercise_3_4 (a : ℝ) (ha : 0 < a) :
    tendsto (λ y, ∫ x in -y..y, x * real.sin x / (x ^ 2 + a ^ 2))
    at_top (𝓝 (real.pi * (real.exp (-a)))) :=
sorry

theorem exercise_3_9 : ∫ x in 0..1, real.log (real.sin (real.pi * x)) = - real.log 2 :=
  sorry

theorem exercise_3_14 {f : ℂ → ℂ} (hf : differentiable ℂ f)
    (hf_inj : function.injective f) :
    ∃ (a b : ℂ), f = (λ z, a * z + b) ∧ a ≠ 0 :=
sorry

theorem exercise_3_22 (D : set ℂ) (hD : D = ball 0 1) (f : ℂ → ℂ)
    (hf : differentiable_on ℂ f D) (hfc : continuous_on f (closure D)) :
    ¬ ∀ z ∈ (sphere (0 : ℂ) 1), f z = 1 / z :=
sorry

theorem exercise_5_1 (f : ℂ → ℂ) (hf : differentiable_on ℂ f (ball 0 1))
  (hb : bounded (set.range f)) (h0 : f ≠ 0) (zeros : ℕ → ℂ) (hz : ∀ n, f (zeros n) = 0)
  (hzz : set.range zeros = {z | f z = 0 ∧ z ∈ (ball (0 : ℂ) 1)}) :
  ∃ (z : ℂ), tendsto (λ n, (∑ i in finset.range n, (1 - zeros i))) at_top (𝓝 z) :=
sorry
