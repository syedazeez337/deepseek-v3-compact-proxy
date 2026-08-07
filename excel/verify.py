"""Recompute every section from the values Excel produced and compare.

Reads excel/_verify_dump.json, written by build_sheet.ps1 in one pass under
manual calculation so the volatile RANDARRAY inputs cannot shift between
reads. Pure stdlib: this has to run without the project venv.
"""

import json
import math
import pathlib
import sys

TOL = 1e-9
HERE = pathlib.Path(__file__).parent
payload = json.loads((HERE / "_verify_dump.json").read_text(encoding="utf-8-sig"))

failures: list[str] = []
checks = 0

# A dump address that has drifted from the builder reads empty cells and hands
# us nulls. That used to surface as a TypeError several hundred lines into a
# recomputation, which said nothing about the real cause. Catch it up front and
# name the block instead.
#
# Smask is exempt: section 2 blanks the masked half of S on purpose, so nulls
# there are the expected result. labels and toc hold text.
NON_NUMERIC = {"Smask", "labels", "toc"}


def find_nulls(node, path=""):
    bad = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in NON_NUMERIC:
                continue
            bad += find_nulls(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            bad += find_nulls(value, f"{path}[{i}]")
    elif node is None:
        bad.append(path or "(root)")
    return bad


empties = find_nulls(payload)
if empties:
    print("EMPTY CELLS IN DUMP")
    print("A dump address in build_sheet.ps1 is stale, or the cell it reads is")
    print("showing an Excel error. Check the builder's ERROR CELLS line first.")
    for p in empties[:20]:
        print(f"  - {p}")
    if len(empties) > 20:
        print(f"  ... and {len(empties) - 20} more")
    sys.exit(1)


def matmul(a, b):
    inner = len(b)
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a):
    return [list(row) for row in zip(*a)]


def close(name, got, want, tol=TOL):
    global checks
    checks += 1
    if len(got) != len(want) or len(got[0]) != len(want[0]):
        failures.append(f"{name}: shape {len(got)}x{len(got[0])} != {len(want)}x{len(want[0])}")
        return
    worst = max(abs(got[i][j] - want[i][j]) for i in range(len(got)) for j in range(len(got[0])))
    if worst > tol:
        failures.append(f"{name}: max abs diff {worst:.3e} > {tol:.0e}")
    else:
        print(f"  ok  {name:<24} max abs diff {worst:.2e}")


def shape(name, m, rows, cols):
    global checks
    checks += 1
    if len(m) != rows or len(m[0]) != cols:
        failures.append(f"{name}: shape {len(m)}x{len(m[0])}, expected {rows}x{cols}")
    else:
        print(f"  ok  {name:<24} shape {rows}x{cols}")


seq = int(payload["seq"])
d_model = int(payload["d_model"])
d_k = int(payload["d_k"])
d_v = int(payload["d_v"])
X, Wq, Wk, Wv = payload["X"], payload["Wq"], payload["Wk"], payload["Wv"]
Q, K, V, KT = payload["Q"], payload["K"], payload["V"], payload["KT"]
D, S, A, O = payload["D"], payload["S"], payload["A"], payload["O"]

print(f"config: seq={seq} d_model={d_model} d_k={d_k} d_v={d_v}\n")

print("shapes")
shape("X", X, d_model, seq)
shape("Wq", Wq, d_k, d_model)
shape("Wv", Wv, d_v, d_model)
shape("Q", Q, d_k, seq)
shape("K", K, d_k, seq)
shape("V", V, d_v, seq)
shape("K^T", KT, seq, d_k)
shape("D", D, seq, seq)
shape("S", S, seq, seq)
shape("A", A, seq, seq)
shape("O", O, d_v, seq)

print("\nprojections")
close("Q = Wq X", Q, matmul(Wq, X))
close("K = Wk X", K, matmul(Wk, X))
close("V = Wv X", V, matmul(Wv, X))

print("\nattention")
close("K^T = transpose(K)", KT, transpose(K))
close("D = K^T Q", D, matmul(KT, Q))
close("S = D / sqrt(dk)", S, [[v / math.sqrt(d_k) for v in row] for row in D])

expected_A = []
for i in range(seq):
    expected_A.append([0.0] * seq)
for j in range(seq):
    column = [S[i][j] for i in range(seq)]
    total = sum(math.exp(v) for v in column)
    for i in range(seq):
        expected_A[i][j] = math.exp(column[i]) / total
close("A = softmax(S) by column", A, expected_A)

checks += 1
worst_sum = max(abs(sum(A[i][j] for i in range(seq)) - 1.0) for j in range(seq))
if worst_sum > 1e-12:
    failures.append(f"A columns do not sum to 1: worst deviation {worst_sum:.3e}")
else:
    print(f"  ok  {'A columns sum to 1':<24} worst deviation {worst_sum:.2e}")

close("O = V A", O, matmul(V, A))

print("\nlabels")
labels = payload["labels"]
for key, prefix in (("x", "x"), ("q", "q"), ("k", "k"), ("v", "v"), ("o", "o")):
    checks += 1
    want = [f"{prefix}{i + 1}" for i in range(seq)]
    if labels[key] != want:
        failures.append(f"{key} labels: {labels[key]} != {want}")
    else:
        print(f"  ok  {key} labels{'':<15} {' '.join(want)}")

checks += 1
if labels["toc"].strip() != "# Attention":
    failures.append(f"table of contents resolved to {labels['toc']!r}, expected '# Attention'")
else:
    print(f"  ok  {'table of contents':<24} resolves to '# Attention'")

print("\n=== section 2: causal attention and the KV cache ===")
s2 = payload["s2"]
seq2, d_k2, d_v2 = int(s2["seq"]), int(s2["d_k"]), int(s2["d_v"])
X2, Wq2, Wk2, Wv2 = s2["X"], s2["Wq"], s2["Wk"], s2["Wv"]
Q2, K2, V2, KT2 = s2["Q"], s2["K"], s2["V"], s2["KT"]
D2, S2, M2, A2, O2 = s2["D"], s2["S"], s2["M"], s2["A"], s2["O"]

print("carried over from section 1")
close("Q = Wq X", Q2, matmul(Wq2, X2))
close("K = Wk X", K2, matmul(Wk2, X2))
close("V = Wv X", V2, matmul(Wv2, X2))
close("K^T = transpose(K)", KT2, transpose(K2))
close("D = K^T Q", D2, matmul(KT2, Q2))
close("S = D / sqrt(dk)", S2, [[v / math.sqrt(d_k2) for v in row] for row in D2])

print("\nthe causal delta")
# rows are keys i, columns are queries j; query j may read key i only when i <= j
want_mask = [[1.0 if i <= j else 0.0 for j in range(seq2)] for i in range(seq2)]
close("M is upper triangular", M2, want_mask)

checks += 1
leaks = [(i, j) for i in range(seq2) for j in range(seq2) if i > j and abs(A2[i][j]) > 0]
if leaks:
    failures.append(f"A leaks attention to future keys at {leaks[:5]}")
else:
    print(f"  ok  {'A is exactly zero above':<24} no query attends to a later key")

checks += 1
worst_sum2 = max(abs(sum(A2[i][j] for i in range(seq2)) - 1.0) for j in range(seq2))
if worst_sum2 > 1e-12:
    failures.append(f"A columns do not sum to 1: worst deviation {worst_sum2:.3e}")
else:
    print(f"  ok  {'A columns sum to 1':<24} worst deviation {worst_sum2:.2e}")

checks += 1
if abs(A2[0][0] - 1.0) > 1e-12:
    failures.append(f"A[k1][q1] is {A2[0][0]}, expected exactly 1 (only one key is visible)")
else:
    print(f"  ok  {'first query is forced':<24} A[k1][q1] = 1, its only visible key")

# softmax over the visible prefix of each column
expected_A2 = [[0.0] * seq2 for _ in range(seq2)]
for j in range(seq2):
    visible = [S2[i][j] for i in range(j + 1)]
    total = sum(math.exp(v) for v in visible)
    for i in range(j + 1):
        expected_A2[i][j] = math.exp(S2[i][j]) / total
close("A = softmax over prefix", A2, expected_A2)

# the blanked display block: masked entries come back as empty strings
checks += 1
bad = []
for i in range(seq2):
    for j in range(seq2):
        cell = s2["Smask"][i][j]
        if i <= j:
            if not isinstance(cell, (int, float)) or abs(cell - S2[i][j]) > TOL:
                bad.append((i, j, "should show S"))
        elif cell not in ("", None):
            bad.append((i, j, f"should be blank, got {cell!r}"))
if bad:
    failures.append(f"S' masked view wrong at {bad[:4]}")
else:
    print(f"  ok  {'S-prime blanks the past':<24} numbers on and above diagonal, empty below")

close("O = V A", O2, matmul(V2, A2))

print("\nKV cache ledger")
close(
    "cached = t*(d_k+d_v)",
    [s2["ledger_cached"]],
    [[(t) * (d_k2 + d_v2) for t in s2["ledger_t"]]],
)
close("K floats = t*d_k", [s2["ledger_k"]], [[t * d_k2 for t in s2["ledger_t"]]])
close("V floats = t*d_v", [s2["ledger_v"]], [[t * d_v2 for t in s2["ledger_t"]]])
close("per token = d_k+d_v", [[s2["per_token"]]], [[d_k2 + d_v2]])
close("cache at context", [[s2["at_context"]]], [[s2["context"] * (d_k2 + d_v2)]])

checks += 1
if s2["toc"].strip() != "# Causal Attention and the KV Cache":
    failures.append(f"section 2 not in table of contents, got {s2['toc']!r}")
else:
    print(f"  ok  {'table of contents':<24} lists section 2")

print("\ndiagnostics (not assertions: the inputs are random each rebuild)")
flat_s = [v for row in S for v in row]
col_max = [max(A[i][j] for i in range(seq)) for j in range(seq)]
print(f"  S range          {min(flat_s):+.2f} .. {max(flat_s):+.2f}")
print(f"  peak attention   {min(col_max):.2f} .. {max(col_max):.2f} per column (uniform would be {1 / seq:.2f})")
spread = [max(O[r]) - min(O[r]) for r in range(d_v)]
print(f"  O row spread     {min(spread):.2f} .. {max(spread):.2f}  (0 means every query got the same output)")

print("\n=== section 3: low-rank compression ===")
s3 = payload["s3"]
seq3 = int(s3["seq"])
d_model3 = int(s3["d_model"])
d_compressed = int(s3["d_compressed"])
X3, Wd, c3, Wu, x_hat3, err3 = (
    s3["X"], s3["W_down"], s3["c"], s3["W_up"], s3["x_hat"], s3["err"],
)

print("shapes")
shape("X", X3, d_model3, seq3)
shape("W_down", Wd, d_compressed, d_model3)
shape("c", c3, d_compressed, seq3)
shape("W_up", Wu, d_model3, d_compressed)
shape("x_hat", x_hat3, d_model3, seq3)
shape("x_hat - X", err3, d_model3, seq3)

print("\ncompression")
close("c = W_down X", c3, matmul(Wd, X3))

print("\nreconstruction")
close("x_hat = W_up c", x_hat3, matmul(Wu, c3))

print("\nreconstruction error")
close("err = x_hat - X", err3, [
    [x_hat3[i][j] - X3[i][j] for j in range(seq3)] for i in range(d_model3)
])

# the teaching point: compression really loses information. The residual is
# not machine noise; it is the part of x that rank 3 cannot carry.
checks += 1
biggest = max(abs(err3[i][j]) for i in range(d_model3) for j in range(seq3))
if biggest < 1e-6:
    failures.append(f"reconstruction error is machine noise ({biggest:.1e}); the rank-3")
    failures.append("bottleneck did not compress anything, so the section has no story")
else:
    print(f"  ok  {'error is real, not noise':<24} rank-3 bottleneck loses up to {biggest:.3f}")

checks += 1
if s3["toc"].strip() != "# Low-Rank Compression":
    failures.append(f"section 3 not in table of contents, got {s3['toc']!r}")
else:
    print(f"  ok  {'table of contents':<24} lists section 3")

print("\n=== section 4: multi-head vs latent attention ===")
s4 = payload["s4"]
seq4 = int(s4["seq"])
d_model4 = int(s4["d_model"])
kv_lora_rank = int(s4["kv_lora_rank"])
n_heads = int(s4["n_heads"])
head_dim = int(s4["head_dim"])
kv_rows = n_heads * head_dim
X4 = s4["X"]
W_K4, K4m = s4["W_K"], s4["K_mha"]
W_V4, V4m = s4["W_V"], s4["V_mha"]
Wd4, c4 = s4["W_DKV"], s4["c_KV"]
Wuk, K4l = s4["W_UK"], s4["K_mla"]
Wuv, V4l = s4["W_UV"], s4["V_mla"]

print("shapes")
shape("X", X4, d_model4, seq4)
shape("W_K", W_K4, kv_rows, d_model4)
shape("K (MHA)", K4m, kv_rows, seq4)
shape("W_V", W_V4, kv_rows, d_model4)
shape("V (MHA)", V4m, kv_rows, seq4)
shape("W_DKV", Wd4, kv_lora_rank, d_model4)
shape("c_KV", c4, kv_lora_rank, seq4)
shape("W_UK", Wuk, kv_rows, kv_lora_rank)
shape("K (MLA)", K4l, kv_rows, seq4)
shape("W_UV", Wuv, kv_rows, kv_lora_rank)
shape("V (MLA)", V4l, kv_rows, seq4)

print("\nmulti-head panel: per-head K and V straight from x")
close("K = W_K X", K4m, matmul(W_K4, X4))
close("V = W_V X", V4m, matmul(W_V4, X4))

print("\nlatent panel: compress once, rebuild per head")
close("c_KV = W_DKV X", c4, matmul(Wd4, X4))
close("K = W_UK c_KV", K4l, matmul(Wuk, c4))
close("V = W_UV c_KV", V4l, matmul(Wuv, c4))

# the cache point: MHA has to cache all n_heads*head_dim rows of K and V,
# 12 floats per token; MLA caches only the rank-3 latent, 3 floats per token.
print("\ncache ledger")
mha_floats = n_heads * (head_dim + head_dim)
close("MHA per token = 2*n_heads*head_dim", [[s4["mha_per_token"]]], [[mha_floats]])
close("MLA per token = kv_lora_rank", [[s4["mla_per_token"]]], [[kv_lora_rank]])

checks += 1
ratio = mha_floats / kv_lora_rank
if abs(ratio - 4.0) > 1e-9:
    failures.append(f"MHA/MLA per-token ratio is {ratio:.2f}, expected 4 (12/3)")
else:
    print(f"  ok  {'MLA cuts cache 4x':<24} MHA {mha_floats} floats/token vs MLA {kv_lora_rank}")

checks += 1
if s4["toc"].strip() != "# Multi-head vs Latent Attention":
    failures.append(f"section 4 not in table of contents, got {s4['toc']!r}")
else:
    print(f"  ok  {'table of contents':<24} lists section 4")

print("\n=== section 5: the RoPE conflict ===")
s5 = payload["s5"]
seq5 = int(s5["seq"])
d_model5 = int(s5["d_model"])
kv_lora_rank5 = int(s5["kv_lora_rank"])
n_heads5 = int(s5["n_heads"])
head_dim5 = int(s5["head_dim"])
rope_dim5 = int(s5["rope_dim"])
kv_rows5 = n_heads5 * head_dim5
X5, Wd5, c5 = s5["X"], s5["W_DKV"], s5["c_KV"]
Wuk5, K5 = s5["W_UK"], s5["K"]
theta5 = s5["theta"]
rope_K5, rope_c5, WUK_rope_c5 = s5["rope_K"], s5["rope_c"], s5["WUK_rope_c"]


def rotate_pair_rows(block, theta):
    """RoPE on rows 0-1 of each head-triple, dim 2 unchanged.

    The repo's rotate_half pairs [x0, x1] -> [-x1, x0] and rotates
    x*cos + rotate_half(x)*sin. Here the rotary pair is the first 2 rows of
    each 3-row head; the third row is untouched. theta is per column (token).
    """
    out = [row[:] for row in block]
    n = len(block)
    for t in range(len(theta)):
        c, s = math.cos(theta[t]), math.sin(theta[t])
        for h in range(n // head_dim5):
            r0 = h * head_dim5
            x0, x1 = block[r0][t], block[r0 + 1][t]
            out[r0][t] = x0 * c - x1 * s
            out[r0 + 1][t] = x0 * s + x1 * c
    return out


print("shapes")
shape("X", X5, d_model5, seq5)
shape("W_DKV", Wd5, kv_lora_rank5, d_model5)
shape("c_KV", c5, kv_lora_rank5, seq5)
shape("W_UK", Wuk5, kv_rows5, kv_lora_rank5)
shape("K", K5, kv_rows5, seq5)
shape("RoPE(K)", rope_K5, kv_rows5, seq5)
shape("RoPE(c_KV)", rope_c5, kv_lora_rank5, seq5)
shape("W_UK RoPE(c)", WUK_rope_c5, kv_rows5, seq5)

print("\ncompression and projection")
close("c_KV = W_DKV X", c5, matmul(Wd5, X5))
close("K = W_UK c_KV", K5, matmul(Wuk5, c5))

print("\nthe two orders")
close("RoPE(K) = rotate K", rope_K5, rotate_pair_rows(K5, theta5))
close("RoPE(c) = rotate c", rope_c5, rotate_pair_rows(c5, theta5))
close("W_UK RoPE(c) = project rotated", WUK_rope_c5, matmul(Wuk5, rope_c5))

# the teaching point: the two cells that should match and do not.
checks += 1
if abs(s5["cell_a"] - s5["cell_b"]) < 1e-9:
    failures.append(
        "RoPE(K)[1,1] == W_UK RoPE(c)[1,1]; the conflict is invisible, "
        "expected them to differ"
    )
else:
    print(
        f"  ok  {'conflict is real':<24} RoPE(K)[1,1] {s5['cell_a']:.4f} "
        f"vs W_UK RoPE(c)[1,1] {s5['cell_b']:.4f}"
    )

checks += 1
if s5["toc"].strip() != "# The RoPE Conflict":
    failures.append(f"section 5 not in table of contents, got {s5['toc']!r}")
else:
    print(f"  ok  {'table of contents':<24} lists section 5")


def rope_pair_rows(block, theta, pair_rows=2):
    """RoPE on rows 0-1 of each consecutive pair; others unchanged.

    The repo's rotate_half pairs [x0, x1] -> [-x1, x0] and rotates
    x*cos + rotate_half(x)*sin. Here each head's rotary pair is its first 2
    rows; any third row (the content dim) is untouched.
    """
    out = [row[:] for row in block]
    for t in range(len(theta)):
        c, s = math.cos(theta[t]), math.sin(theta[t])
        for r0 in range(0, len(block), pair_rows):
            x0, x1 = block[r0][t], block[r0 + 1][t]
            out[r0][t] = x0 * c - x1 * s
            out[r0 + 1][t] = x0 * s + x1 * c
    return out


print("\n=== section 6: decoupled RoPE ===")
s6 = payload["s6"]
seq6 = int(s6["seq"])
d_model6 = int(s6["d_model"])
q_lora_rank6 = int(s6["q_lora_rank"])
kv_lora_rank6 = int(s6["kv_lora_rank"])
n_heads6 = int(s6["n_heads"])
qk_nope6 = int(s6["qk_nope"])
qk_rope6 = int(s6["qk_rope"])
X6, Wd6, cQ6 = s6["X"], s6["W_DQ"], s6["c_Q"]
WUQ1_6, qc1_6 = s6["W_UQ1"], s6["q_c1"]
WQR1_6, qr1_6 = s6["W_QR1"], s6["q_r1"]
WdKV6, cKV6 = s6["W_DKV"], s6["c_KV"]
WUK1_6, kc1_6 = s6["W_UK1"], s6["k_c1"]
WKR6, kr6 = s6["W_KR"], s6["k_r"]
theta6 = s6["theta"]
rope_qr6, rope_kr6 = s6["rope_qr"], s6["rope_kr"]
Sc6, Sr6, S6 = s6["S_c"], s6["S_r"], s6["S"]

print("shapes")
shape("X", X6, d_model6, seq6)
shape("W_DQ", Wd6, q_lora_rank6, d_model6)
shape("c_Q", cQ6, q_lora_rank6, seq6)
shape("W_UQ1", WUQ1_6, qk_nope6, q_lora_rank6)
shape("q_c1", qc1_6, qk_nope6, seq6)
shape("W_QR1", WQR1_6, qk_rope6, q_lora_rank6)
shape("q_r1", qr1_6, qk_rope6, seq6)
shape("c_KV", cKV6, kv_lora_rank6, seq6)
shape("k_c1", kc1_6, qk_nope6, seq6)
shape("k_r", kr6, qk_rope6, seq6)
shape("RoPE(q_r)", rope_qr6, qk_rope6, seq6)
shape("RoPE(k_r)", rope_kr6, qk_rope6, seq6)

print("\nprojections")
close("c_Q = W_DQ X", cQ6, matmul(Wd6, X6))
close("q_c1 = W_UQ1 c_Q", qc1_6, matmul(WUQ1_6, cQ6))
close("q_r1 = W_QR1 c_Q", qr1_6, matmul(WQR1_6, cQ6))
close("c_KV = W_DKV X", cKV6, matmul(WdKV6, X6))
close("k_c1 = W_UK1 c_KV", kc1_6, matmul(WUK1_6, cKV6))
close("k_r = W_KR X", kr6, matmul(WKR6, X6))

print("\nrotations")
close("RoPE(q_r) = rotate q_r1", rope_qr6, rope_pair_rows(qr1_6, theta6))
close("RoPE(k_r) = rotate k_r", rope_kr6, rope_pair_rows(kr6, theta6))

print("\nscore, head 1")
# head 1: k_c1 is qk_nope rows, q_c1 is qk_nope rows, q_r1 is qk_rope rows
close("S_c = K_c1^T Q_c1", Sc6, matmul(transpose(kc1_6), qc1_6))
close("S_r = K_r^T Q_r1", Sr6, matmul(transpose(kr6), qr1_6))
close("S = S_c + S_r", S6, [
    [Sc6[i][j] + Sr6[i][j] for j in range(seq6)] for i in range(seq6)
])

checks += 1
if s6["toc"].strip() != "# Decoupled RoPE":
    failures.append(f"section 6 not in table of contents, got {s6['toc']!r}")
else:
    print(f"  ok  {'table of contents':<24} lists section 6")

print("\n=== section 7: weight absorption and the cache ledger ===")
s7 = payload["s7"]
seq7 = int(s7["seq"])
d_model7 = int(s7["d_model"])
q_lora_rank7 = int(s7["q_lora_rank"])
kv_lora_rank7 = int(s7["kv_lora_rank"])
n_heads7 = int(s7["n_heads"])
qk_nope7 = int(s7["qk_nope"])
qk_rope7 = int(s7["qk_rope"])
v_head7 = int(s7["v_head"])
X7, Wd7, cQ7 = s7["X"], s7["W_DQ"], s7["c_Q"]
WUQ1_7, qc1_7 = s7["W_UQ1"], s7["q_c1"]
WQR1_7, qr1_7 = s7["W_QR1"], s7["q_r1"]
WdKV7, cKV7 = s7["W_DKV"], s7["c_KV"]
WKR7, kr7 = s7["W_KR"], s7["k_r"]
theta7 = s7["theta"]
rope_qr7, rope_kr7 = s7["rope_qr"], s7["rope_kr"]
WUK1_7, kc1_7 = s7["W_UK1"], s7["k_c1"]
WUV1_7, v1_7 = s7["W_UV1"], s7["v1"]
SA7, AA7, OA7 = s7["S_A"], s7["A_A"], s7["O_A"]
qpc7, SB7, AB7, cKV_A7, OB7 = s7["qp_c"], s7["S_B"], s7["A_B"], s7["cKV_A"], s7["O_B"]

print("shapes")
shape("X", X7, d_model7, seq7)
shape("c_KV", cKV7, kv_lora_rank7, seq7)
shape("W_UK1", WUK1_7, qk_nope7, kv_lora_rank7)
shape("k_c1", kc1_7, qk_nope7, seq7)
shape("W_UV1", WUV1_7, v_head7, kv_lora_rank7)
shape("v1", v1_7, v_head7, seq7)
shape("q_c1", qc1_7, qk_nope7, seq7)
shape("q_r1", qr1_7, qk_rope7, seq7)
shape("S_A", SA7, seq7, seq7)
shape("A_A", AA7, seq7, seq7)
shape("O_A", OA7, v_head7, seq7)
shape("S_B", SB7, seq7, seq7)
shape("O_B", OB7, v_head7, seq7)

print("\nshared preparation")
close("c_KV = W_DKV X", cKV7, matmul(WdKV7, X7))
close("q_c1 = W_UQ1 c_Q", qc1_7, matmul(WUQ1_7, cQ7))
close("q_r1 = W_QR1 c_Q", qr1_7, matmul(WQR1_7, cQ7))
close("k_c1 = W_UK1 c_KV", kc1_7, matmul(WUK1_7, cKV7))
close("v1 = W_UV1 c_KV", v1_7, matmul(WUV1_7, cKV7))

print("\npath A (naive)")
# The rope half uses the ROTATED vectors. The section used to score with the
# raw k_r and q_r while displaying RoPE(k_r) and RoPE(q_r) beside them, which
# made those two blocks decorative and the term wrong. Both paths shared the
# error, so S_A = S_B still held and nothing caught it.
close("S_A = K_c1^T Q_c1 + RoPE(K_r)^T RoPE(Q_r)", SA7, [
    [sum(kc1_7[i][t] * qc1_7[i][q] for i in range(qk_nope7))
     + sum(rope_kr7[i][t] * rope_qr7[i][q] for i in range(qk_rope7))
     for q in range(seq7)] for t in range(seq7)
])
expected_AA = [[0.0] * seq7 for _ in range(seq7)]
for j in range(seq7):
    col = [SA7[i][j] for i in range(seq7)]
    tot = sum(math.exp(v) for v in col)
    for i in range(seq7):
        expected_AA[i][j] = math.exp(col[i]) / tot
close("A_A = softmax(S_A)", AA7, expected_AA)
close("O_A = V1 A_A", OA7, matmul(v1_7, AA7))

print("\npath B (absorbed)")
# q'_c is 6x3: tokens x latent, q'_c = q_c1^T W_UK1
close("q'_c = q_c1^T W_UK1", qpc7, matmul(transpose(qc1_7), WUK1_7))
close("S_B = c_KV^T q'_c + RoPE(K_r)^T RoPE(Q_r)", SB7, [
    [sum(cKV7[k][t] * qpc7[q][k] for k in range(kv_lora_rank7))
     + sum(rope_kr7[i][t] * rope_qr7[i][q] for i in range(qk_rope7))
     for q in range(seq7)] for t in range(seq7)
])
expected_AB = [[0.0] * seq7 for _ in range(seq7)]
for j in range(seq7):
    col = [SB7[i][j] for i in range(seq7)]
    tot = sum(math.exp(v) for v in col)
    for i in range(seq7):
        expected_AB[i][j] = math.exp(col[i]) / tot
close("A_B = softmax(S_B)", AB7, expected_AB)
close("c_KV A_B", cKV_A7, matmul(cKV7, AB7))
# O_B = W_UV1 (c_KV A_B)
close("O_B = W_UV1 (c_KV A_B)", OB7, matmul(WUV1_7, cKV_A7))

print("\nthe equality that is the point")
close("S_A == S_B", SA7, SB7)
close("O_A == O_B", OA7, OB7)

print("\ncache ledger")
mha7 = n_heads7 * (qk_nope7 + v_head7 + qk_rope7)
gqa7 = qk_nope7 + v_head7 + qk_rope7
mla7 = kv_lora_rank7 + qk_rope7
close("MHA per token", [[s7["mha_per_token"]]], [[mha7]])
close("GQA per token", [[s7["gqa_per_token"]]], [[gqa7]])
close("MLA per token", [[s7["mla_per_token"]]], [[mla7]])
close("MLA saving ratio", [[s7["mla_ratio"]]], [[mha7 / mla7]])

checks += 1
if abs(s7["mha_per_token"] / s7["mla_per_token"] - 3.2) > 1e-9:
    failures.append("MHA/MLA per-token ratio is not 3.2x")
else:
    print(f"  ok  {'MLA cuts cache 3.2x':<24} MHA {mha7} vs MLA {mla7}")

checks += 1
if s7["toc"].strip() != "# Weight Absorption and the Cache Ledger":
    failures.append(f"section 7 not in table of contents, got {s7['toc']!r}")
else:
    print(f"  ok  {'table of contents':<24} lists section 7")

print("\n=== section 8: SwiGLU and top-k MoE ===")
s8 = payload["s8"]
seq8 = int(s8["seq"])
d_model8 = int(s8["d_model"])
expert_hidden8 = int(s8["expert_hidden"])
n_experts8 = int(s8["n_experts"])
top_k8 = int(s8["top_k"])
X8, Wg8, g8 = s8["X"], s8["W_gate"], s8["g"]
Wv8, v8 = s8["W_value"], s8["v"]
h8, Wo8, out8 = s8["h"], s8["W_out"], s8["out"]
Wr8, logits8 = s8["W_router"], s8["logits"]
A8, argmax8 = s8["A"], s8["argmax"]

print("shapes")
shape("X", X8, d_model8, seq8)
shape("W_gate", Wg8, expert_hidden8, d_model8)
shape("g", g8, expert_hidden8, seq8)
shape("W_value", Wv8, expert_hidden8, d_model8)
shape("v", v8, expert_hidden8, seq8)
shape("h", h8, expert_hidden8, seq8)
shape("W_out", Wo8, d_model8, expert_hidden8)
shape("out", out8, d_model8, seq8)
shape("W_router", Wr8, n_experts8, d_model8)
shape("logits", logits8, n_experts8, seq8)
shape("A", A8, n_experts8, seq8)

print("\nSwiGLU expert")
gate_in = matmul(Wg8, X8)
close("g = silu(W_g X)", g8, [
    [gate_in[i][j] / (1 + math.exp(-gate_in[i][j])) for j in range(seq8)]
    for i in range(expert_hidden8)
])
close("v = W_v X", v8, matmul(Wv8, X8))
close("h = g * v", h8, [
    [g8[i][j] * v8[i][j] for j in range(seq8)] for i in range(expert_hidden8)
])
close("out = W_o h", out8, matmul(Wo8, h8))

print("\nrouter")
close("logits = W_r X", logits8, matmul(Wr8, X8))
expected_A8 = [[0.0] * seq8 for _ in range(n_experts8)]
for j in range(seq8):
    col = [logits8[i][j] for i in range(n_experts8)]
    tot = sum(math.exp(v) for v in col)
    for i in range(n_experts8):
        expected_A8[i][j] = math.exp(col[i]) / tot
close("A = softmax(logits) by token", A8, expected_A8)

checks += 1
for j in range(seq8):
    col = [A8[i][j] for i in range(n_experts8)]
    if abs(sum(col) - 1.0) > 1e-9:
        failures.append(f"token {j} A column does not sum to 1")
        break
else:
    print(f"  ok  {'A columns sum to 1':<24} per-token distribution")

checks += 1
# argmax: the expert with the largest A (and thus largest logit, since exp is
# monotone)
for j in range(seq8):
    want = max(range(n_experts8), key=lambda i: A8[i][j]) + 1  # 1-indexed
    if argmax8[j] != want:
        failures.append(f"token {j} argmax {argmax8[j]}, expected {want}")
        break
else:
    print(f"  ok  {'argmax matches max A':<24} top_k={top_k8}")

checks += 1
if s8["toc"].strip() != "# SwiGLU and Top-k MoE":
    failures.append(f"section 8 not in table of contents, got {s8['toc']!r}")
else:
    print(f"  ok  {'table of contents':<24} lists section 8")

print("\n=== section 9: fine-grained and shared experts ===")
s9 = payload["s9"]
seq9 = int(s9["seq"])
d_model9 = int(s9["d_model"])
expert_hidden9 = int(s9["expert_hidden"])
n_experts9 = int(s9["n_experts"])
top_k9 = int(s9["top_k"])
X9, Wr9, logits9 = s9["X"], s9["W_router"], s9["logits"]
A9, sel1_9, sel2_9 = s9["A"], s9["sel1"], s9["sel2"]
w1_9, w2_9 = s9["w1"], s9["w2"]
Wsg9, Wsv9, Wso9, shared9 = s9["W_sg"], s9["W_sv"], s9["W_so"], s9["shared_out"]

print("shapes")
shape("X", X9, d_model9, seq9)
shape("W_router", Wr9, n_experts9, d_model9)
shape("logits", logits9, n_experts9, seq9)
shape("A", A9, n_experts9, seq9)
shape("W_sg", Wsg9, expert_hidden9, d_model9)
shape("W_sv", Wsv9, expert_hidden9, d_model9)
shape("W_so", Wso9, d_model9, expert_hidden9)
shape("shared out", shared9, d_model9, seq9)

print("\nrouter")
close("logits = W_r X", logits9, matmul(Wr9, X9))
expected_A9 = [[0.0] * seq9 for _ in range(n_experts9)]
for j in range(seq9):
    col = [logits9[i][j] for i in range(n_experts9)]
    tot = sum(math.exp(v) for v in col)
    for i in range(n_experts9):
        expected_A9[i][j] = math.exp(col[i]) / tot
close("A = softmax(logits)", A9, expected_A9)

print("\ntop-2 selection")
checks += 1
for j in range(seq9):
    col = [A9[i][j] for i in range(n_experts9)]
    order = sorted(range(n_experts9), key=lambda i: col[i], reverse=True)
    if int(sel1_9[j]) != order[0] + 1 or int(sel2_9[j]) != order[1] + 1:
        failures.append(f"token {j}: sel {sel1_9[j]},{sel2_9[j]}, want {order[0]+1},{order[1]+1}")
        break
else:
    print(f"  ok  {'top-2 picks match':<24} largest two affinities per token")

checks += 1
for j in range(seq9):
    a1, a2 = A9[int(sel1_9[j]) - 1][j], A9[int(sel2_9[j]) - 1][j]
    w1_want, w2_want = a1 / (a1 + a2), a2 / (a1 + a2)
    if abs(w1_9[j] - w1_want) > 1e-9 or abs(w2_9[j] - w2_want) > 1e-9:
        failures.append(f"token {j}: weights {w1_9[j]:.4f},{w2_9[j]:.4f}, want {w1_want:.4f},{w2_want:.4f}")
        break
else:
    print(f"  ok  {'renormalised weights':<24} w1+w2=1 over the selected pair")

print("\nshared expert (always on)")
g_in9 = matmul(Wsg9, X9)
g9 = [[g_in9[i][j] / (1 + math.exp(-g_in9[i][j])) for j in range(seq9)] for i in range(expert_hidden9)]
v9 = matmul(Wsv9, X9)
h9 = [[g9[i][j] * v9[i][j] for j in range(seq9)] for i in range(expert_hidden9)]
close("shared = W_so silu(W_sg X) * W_sv X", shared9, matmul(Wso9, h9))

checks += 1
if s9["toc"].strip() != "# DeepSeekMoE: Fine-Grained and Shared Experts":
    failures.append(f"section 9 not in table of contents, got {s9['toc']!r}")
else:
    print(f"  ok  {'table of contents':<24} lists section 9")

print("\n=== section 10: sigmoid affinity and route scale ===")
s10 = payload["s10"]
seq10 = int(s10["seq"])
d_model10 = int(s10["d_model"])
n_experts10 = int(s10["n_experts"])
top_k10 = int(s10["top_k"])
route_scale = float(s10["route_scale"])
X10, Wr10, logits10 = s10["X"], s10["W_router"], s10["logits"]
aff10, sel1_10, sel2_10 = s10["affinity"], s10["sel1"], s10["sel2"]
w1_10, w2_10, wscaled10 = s10["w1"], s10["w2"], s10["w_scaled"]

print("shapes")
shape("X", X10, d_model10, seq10)
shape("W_router", Wr10, n_experts10, d_model10)
shape("logits", logits10, n_experts10, seq10)
shape("affinity", aff10, n_experts10, seq10)
shape("w * route_scale", wscaled10, 2, seq10)

print("\nrouter")
close("logits = W_r X", logits10, matmul(Wr10, X10))
close("affinity = sigmoid(logits)", aff10, [
    [1 / (1 + math.exp(-logits10[i][j])) for j in range(seq10)]
    for i in range(n_experts10)
])

print("\ntop-2 and renormalisation")
checks += 1
for j in range(seq10):
    col = [aff10[i][j] for i in range(n_experts10)]
    order = sorted(range(n_experts10), key=lambda i: col[i], reverse=True)
    if int(sel1_10[j]) != order[0] + 1 or int(sel2_10[j]) != order[1] + 1:
        failures.append(f"token {j}: sel {sel1_10[j]},{sel2_10[j]}, want {order[0]+1},{order[1]+1}")
        break
else:
    print(f"  ok  {'top-2 picks match':<24} largest two affinities per token")

checks += 1
for j in range(seq10):
    a1, a2 = aff10[int(sel1_10[j]) - 1][j], aff10[int(sel2_10[j]) - 1][j]
    w1_want, w2_want = a1 / (a1 + a2), a2 / (a1 + a2)
    if abs(w1_10[j] - w1_want) > 1e-9 or abs(w2_10[j] - w2_want) > 1e-9:
        failures.append(f"token {j}: weights {w1_10[j]:.4f},{w2_10[j]:.4f}, want {w1_want:.4f},{w2_want:.4f}")
        break
else:
    print(f"  ok  {'renormalised weights':<24} over selected pair only")

checks += 1
for j in range(seq10):
    if abs(wscaled10[0][j] - w1_10[j] * route_scale) > 1e-9 or abs(wscaled10[1][j] - w2_10[j] * route_scale) > 1e-9:
        failures.append(f"token {j}: route scale not applied")
        break
else:
    print(f"  ok  {'route scale applied':<24} weights * {route_scale}")

checks += 1
if s10["toc"].strip() != "# Sigmoid Affinity and Route Scale":
    failures.append(f"section 10 not in table of contents, got {s10['toc']!r}")
else:
    print(f"  ok  {'table of contents':<24} lists section 10")

print("\n=== section 11: load collapse ===")
s11 = payload["s11"]
n_tokens11 = int(s11["n_tokens"])
d_model11 = int(s11["d_model"])
n_experts11 = int(s11["n_experts"])
top_k11 = int(s11["top_k"])
X11, Wr11, logits11 = s11["X"], s11["W_router"], s11["logits"]
bias11, score11, sel1_11, load11 = s11["bias"], s11["score"], s11["sel1"], s11["load"]
entropy11 = s11["entropy"]

print("shapes")
shape("X", X11, d_model11, n_tokens11)
shape("W_router", Wr11, n_experts11, d_model11)
shape("logits", logits11, n_experts11, n_tokens11)
shape("bias", bias11, n_experts11, n_tokens11)
shape("score", score11, n_experts11, n_tokens11)

print("\nrouter")
close("logits = W_r X", logits11, matmul(Wr11, X11))
close("score = logits + bias", score11, [
    [logits11[i][j] + bias11[i][j] for j in range(n_tokens11)]
    for i in range(n_experts11)
])

print("\nload collapse")
checks += 1
for j in range(n_tokens11):
    col = [score11[i][j] for i in range(n_experts11)]
    want = max(range(n_experts11), key=lambda i: col[i]) + 1
    if int(sel1_11[j]) != want:
        failures.append(f"token {j}: top expert {sel1_11[j]}, expected {want}")
        break
else:
    print(f"  ok  {'top-1 picks match':<24} argmax of selection score")

checks += 1
want_load = [0] * n_experts11
for j in range(n_tokens11):
    want_load[int(sel1_11[j]) - 1] += 1
if load11 != want_load:
    failures.append(f"load {load11} != expected {want_load}")
else:
    print(f"  ok  {'load counts match':<24} {load11}")

checks += 1
total = sum(load11)
probs = [c / total for c in load11]
nz = [p for p in probs if p > 0]
ent_want = -sum(p * math.log(p) for p in nz) / math.log(n_experts11)
if abs(entropy11 - ent_want) > 1e-9:
    failures.append(f"entropy {entropy11:.4f} != {ent_want:.4f}")
else:
    print(f"  ok  {'normalised entropy':<24} {entropy11:.4f}")

checks += 1
top3 = sum(sorted(load11, reverse=True)[:3])
if entropy11 > 0.75:
    failures.append(f"entropy {entropy11:.4f} too high; the load did not collapse")
elif top3 < n_tokens11 * 0.6:
    failures.append(f"top-3 experts only took {top3}/{n_tokens11} tokens; collapse too weak")
else:
    print(f"  ok  {'collapse is visible':<24} entropy {entropy11:.4f}, top-3 hold {top3}/{n_tokens11}")

checks += 1
if s11["toc"].strip() != "# Load Collapse":
    failures.append(f"section 11 not in table of contents, got {s11['toc']!r}")
else:
    print(f"  ok  {'table of contents':<24} lists section 11")

print("\n=== section 12: auxiliary-loss-free load balancing ===")
s12 = payload["s12"]
n_tokens12 = int(s12["n_tokens"])
d_model12 = int(s12["d_model"])
n_experts12 = int(s12["n_experts"])
top_k12 = int(s12["top_k"])
rate12 = float(s12["update_rate"])
X12, Wr12, logits12 = s12["X"], s12["W_router"], s12["logits"]
b0_12, c0 = s12["b0"], s12["count0"]
b1_12, c1 = s12["b1"], s12["count1"]
b2_12, c2 = s12["b2"], s12["count2"]
b3_12, wdiff12 = s12["b3"], s12["w_diff"]

print("shapes")
shape("X", X12, d_model12, n_tokens12)
shape("W_router", Wr12, n_experts12, d_model12)
shape("logits", logits12, n_experts12, n_tokens12)
shape("b0", [b0_12], 1, n_experts12)
shape("count0", [c0], 1, n_experts12)
shape("b1", [b1_12], 1, n_experts12)
shape("w_diff", wdiff12, n_experts12, d_model12)

adv12 = s12["advantage"]
shape("advantage", adv12, n_experts12, n_tokens12)

print("\nrouter")
# Section 12 inherits section 11's engineered collapse, so the logits carry a
# fixed per-expert advantage on top of the projection.
base12 = matmul(Wr12, X12)
close("logits = W_r X + advantage", logits12, [
    [base12[i][j] + adv12[i][j] for j in range(n_tokens12)] for i in range(n_experts12)
])


def load_counts(bias, logits, n_tok, n_exp):
    """Recompute top-1 load counts from logits + broadcast bias."""
    counts = [0] * n_exp
    for j in range(n_tok):
        col = [logits[i][j] + bias[i] for i in range(n_exp)]
        best = max(range(n_exp), key=lambda i: col[i])
        counts[best] += 1
    return counts


print("\nthe update rule: b += rate * sign(target - load)")
target12 = n_tokens12 / n_experts12

checks += 1
c0_want = load_counts(b0_12, logits12, n_tokens12, n_experts12)
if c0 != c0_want:
    failures.append(f"count0 {c0} != {c0_want}")
else:
    print(f"  ok  {'count0 from first principles':<24}")

checks += 1
b1_want = [b0_12[i] + rate12 * (1 if c0_want[i] < target12 else -1) for i in range(n_experts12)]
if any(abs(b1_12[i] - b1_want[i]) > 1e-9 for i in range(n_experts12)):
    failures.append(f"b1 {b1_12} != {b1_want}")
else:
    print(f"  ok  {'b1 = b0 + rate*sign':<24} target {target12:.0f}")

checks += 1
c1_want = load_counts(b1_want, logits12, n_tokens12, n_experts12)
if c1 != c1_want:
    failures.append(f"count1 {c1} != {c1_want}")
else:
    print(f"  ok  {'count1 from first principles':<24}")

checks += 1
b2_want = [b1_12[i] + rate12 * (1 if c1_want[i] < target12 else -1) for i in range(n_experts12)]
if any(abs(b2_12[i] - b2_want[i]) > 1e-9 for i in range(n_experts12)):
    failures.append(f"b2 {b2_12} != {b2_want}")
else:
    print(f"  ok  {'b2 = b1 + rate*sign':<24}")

checks += 1
c2_want = load_counts(b2_want, logits12, n_tokens12, n_experts12)
if c2 != c2_want:
    failures.append(f"count2 {c2} != {c2_want}")
else:
    print(f"  ok  {'count2 from first principles':<24}")

checks += 1
b3_want = [b2_12[i] + rate12 * (1 if c2_want[i] < target12 else -1) for i in range(n_experts12)]
if any(abs(b3_12[i] - b3_want[i]) > 1e-9 for i in range(n_experts12)):
    failures.append(f"b3 {b3_12} != {b3_want}")
else:
    print(f"  ok  {'b3 = b2 + rate*sign':<24}")

# the load should move toward uniform: the spread of counts should shrink
checks += 1
spread0 = max(c0_want) - min(c0_want)
spread2 = max(c2_want) - min(c2_want)
if spread2 >= spread0:
    failures.append(f"load spread did not shrink: {spread0} -> {spread2}")
else:
    print(f"  ok  {'load spread shrinks':<24} {spread0} -> {spread2}")

# the proof: w_diff is all zeros
checks += 1
worst_diff = max(abs(v) for row in wdiff12 for v in row)
if worst_diff > 1e-9:
    failures.append(f"W_router changed under the bias update: max diff {worst_diff:.2e}")
else:
    print(f"  ok  {'gate weight untouched':<24} W_router - copy is all zeros")

checks += 1
if s12["toc"].strip() != "# Auxiliary-Loss-Free Load Balancing":
    failures.append(f"section 12 not in table of contents, got {s12['toc']!r}")
else:
    print(f"  ok  {'table of contents':<24} lists section 12")

print("\n=== section 13: the V3 block and the sequence-wise balance loss ===")
s13 = payload["s13"]
seq13 = int(s13["seq"])
dm13 = int(s13["d_model"])
dk13 = int(s13["d_k"])
eh13 = int(s13["expert_hidden"])
ne13 = int(s13["n_experts"])
tk13 = int(s13["top_k"])
eps13 = float(s13["rms_eps"])
coef13 = float(s13["balance_coef"])
scale13 = float(s13["route_scale"])

X13, gamma13, n13 = s13["X"], s13["gamma"], s13["n"]

print("RMSNorm")
rms_want = [
    math.sqrt(sum(X13[i][j] ** 2 for i in range(dm13)) / dm13 + eps13) for j in range(seq13)
]
close("rms = sqrt(mean(x^2)+eps)", [s13["rms"]], [rms_want])
close("n = X / rms * gamma", n13, [
    [X13[i][j] / rms_want[j] * gamma13[i] for j in range(seq13)] for i in range(dm13)
])

# RMSNorm does not centre, so the per-token mean is not zero. What does hold is
# that each column is scaled to very nearly unit RMS. Not exactly: eps sits
# inside the square root, so the mean square lands on ms/(ms+eps). Assert that
# exact value rather than 1.0, which makes the check tight and also pins down
# that eps is applied inside the root and not outside.
checks += 1
worst_unit = 0.0
drift_from_one = 0.0
for j in range(seq13):
    mean_sq = sum(X13[i][j] ** 2 for i in range(dm13)) / dm13
    got = sum((X13[i][j] / rms_want[j]) ** 2 for i in range(dm13)) / dm13
    worst_unit = max(worst_unit, abs(got - mean_sq / (mean_sq + eps13)))
    drift_from_one = max(drift_from_one, abs(got - 1.0))
if worst_unit > 1e-12:
    failures.append(f"normalised column mean square != ms/(ms+eps): worst {worst_unit:.2e}")
else:
    print(f"  ok  {'unit RMS before gamma':<24} exact to {worst_unit:.1e}; eps pulls it {drift_from_one:.1e} off 1.0")

print("\nattention sublayer")
close("Q = Wq n", s13["Q"], matmul(s13["Wq"], n13))
close("K = Wk n", s13["K"], matmul(s13["Wk"], n13))
close("V = Wv n", s13["V"], matmul(s13["Wv"], n13))
close("K^T = transpose(K)", s13["KT"], transpose(s13["K"]))
close("S = K^T Q / sqrt(dk)", s13["S"], [
    [v / math.sqrt(dk13) for v in row] for row in matmul(s13["KT"], s13["Q"])
])

A13 = s13["A"]
checks += 1
leak13 = [(i, j) for i in range(seq13) for j in range(seq13) if i > j and abs(A13[i][j]) > 0]
if leak13:
    failures.append(f"block attention leaks to future keys at {leak13[:5]}")
else:
    print(f"  ok  {'A is causal':<24} no query attends to a later key")

S13 = s13["S"]
expected_A13 = [[0.0] * seq13 for _ in range(seq13)]
for j in range(seq13):
    total = sum(math.exp(S13[i][j]) for i in range(j + 1))
    for i in range(j + 1):
        expected_A13[i][j] = math.exp(S13[i][j]) / total
close("A = softmax over prefix", A13, expected_A13)
close("O_attn = V A", s13["O_attn"], matmul(s13["V"], A13))
close("attn_out = W_O O_attn", s13["attn_out"], matmul(s13["W_O"], s13["O_attn"]))

# the residual: the sublayer output is added to the raw X, not the normalised n
print("\nresidual structure")
close("x1 = X + attn_out", s13["x1"], [
    [X13[i][j] + s13["attn_out"][i][j] for j in range(seq13)] for i in range(dm13)
])
checks += 1
# if the residual had wrongly used n instead of X, x1 - attn_out would equal n
worst_wrong = max(
    abs(s13["x1"][i][j] - s13["attn_out"][i][j] - n13[i][j])
    for i in range(dm13) for j in range(seq13)
)
if worst_wrong < 1e-9:
    failures.append("residual added the sublayer to the normalised input, not the raw one")
else:
    print(f"  ok  {'residual carries raw X':<24} differs from the normalised path by {worst_wrong:.2f}")

print("\nFFN sublayer, layer 0 dense")
x1_13 = s13["x1"]
rms2_want = [
    math.sqrt(sum(x1_13[i][j] ** 2 for i in range(dm13)) / dm13 + eps13) for j in range(seq13)
]
close("rms2", [s13["rms2"]], [rms2_want])
close("n2 = x1 / rms2 * gamma2", s13["n2"], [
    [x1_13[i][j] / rms2_want[j] * s13["gamma2"][i] for j in range(seq13)] for i in range(dm13)
])
gate_pre = matmul(s13["W_gate"], s13["n2"])
close("g = silu(W_g n2)", s13["g"], [
    [z / (1 + math.exp(-z)) for z in row] for row in gate_pre
])
close("v = W_v n2", s13["ffn_v"], matmul(s13["W_value"], s13["n2"]))
hidden13 = [
    [s13["g"][i][j] * s13["ffn_v"][i][j] for j in range(seq13)] for i in range(eh13)
]
close("ffn_out = W_o (g*v)", s13["ffn_out"], matmul(s13["W_out"], hidden13))
close("y = x1 + ffn_out", s13["y"], [
    [x1_13[i][j] + s13["ffn_out"][i][j] for j in range(seq13)] for i in range(dm13)
])

print("\nMoE router, layer 1 and up")
close("logits = W_r n2", s13["logits"], matmul(s13["W_router"], s13["n2"]))
aff13 = s13["affinity"]
close("affinity = sigmoid(logits)", aff13, [
    [1 / (1 + math.exp(-z)) for z in row] for row in s13["logits"]
])

checks += 1
bad_sel = []
for j in range(seq13):
    order = sorted(range(ne13), key=lambda i: s13["logits"][i][j], reverse=True)
    if int(s13["sel1"][j]) != order[0] + 1 or int(s13["sel2"][j]) != order[1] + 1:
        bad_sel.append(j + 1)
if bad_sel:
    failures.append(f"top-2 selection wrong at tokens {bad_sel}")
else:
    print(f"  ok  {'top-2 picks match':<24} two largest logits per token")

checks += 1
worst_w = 0.0
for j in range(seq13):
    a1 = aff13[int(s13["sel1"][j]) - 1][j]
    a2 = aff13[int(s13["sel2"][j]) - 1][j]
    worst_w = max(worst_w, abs(s13["w1"][j] - a1 / (a1 + a2) * scale13))
    worst_w = max(worst_w, abs(s13["w2"][j] - a2 / (a1 + a2) * scale13))
if worst_w > 1e-9:
    failures.append(f"gate weights wrong: max diff {worst_w:.2e}")
else:
    print(f"  ok  {'w1,w2 renormalised':<24} over the pair, times route_scale {scale13}")

checks += 1
pair_sum = [s13["w1"][j] + s13["w2"][j] for j in range(seq13)]
if max(abs(p - scale13) for p in pair_sum) > 1e-9:
    failures.append(f"gate weights do not sum to route_scale: {pair_sum}")
else:
    print(f"  ok  {'w1+w2 = route_scale':<24} every token")

print("\nsequence-wise balance loss")
count_want = [0] * ne13
for j in range(seq13):
    count_want[int(s13["sel1"][j]) - 1] += 1
    count_want[int(s13["sel2"][j]) - 1] += 1
close("count per expert", [s13["count"]], [[float(c) for c in count_want]])
freq_want = [c * ne13 / (seq13 * tk13) for c in count_want]
close("f = count*n_e/(tok*top_k)", [s13["freq"]], [freq_want])
mean_want = [sum(aff13[i][j] for j in range(seq13)) / seq13 for i in range(ne13)]
close("P = mean affinity", [s13["meanaff"]], [mean_want])
close("f * P", [s13["fP"]], [[freq_want[i] * mean_want[i] for i in range(ne13)]])
close("loss = coef * sum(f*P)", [[s13["balance_loss"]]], [
    [coef13 * sum(freq_want[i] * mean_want[i] for i in range(ne13))]
])

checks += 1
if sum(count_want) != seq13 * tk13:
    failures.append(f"assignments {sum(count_want)} != tokens*top_k {seq13 * tk13}")
else:
    print(f"  ok  {'assignments add up':<24} {sum(count_want)} = {seq13} tokens x top-{tk13}")

print("\nparameter ledger")
expert_want = 2 * eh13 * dm13 + dm13 * eh13
close("attention params", [[s13["p_attention"]]], [[3 * dk13 * dm13 + dm13 * dk13]])
close("one expert", [[s13["p_expert"]]], [[expert_want]])
close("MoE total", [[s13["p_moe_total"]]], [[(ne13 + 1) * expert_want + ne13 * dm13]])
close("MoE active", [[s13["p_moe_active"]]], [[(tk13 + 1) * expert_want + ne13 * dm13]])
close("block total", [[s13["p_block_total"]]], [
    [3 * dk13 * dm13 + dm13 * dk13 + (ne13 + 1) * expert_want + ne13 * dm13 + 2 * dm13]
])
close("active fraction", [[s13["p_active_frac"]]], [
    [s13["p_block_active"] / s13["p_block_total"]]
])

checks += 1
if s13["toc"].strip() != "# The V3 Block and the Sequence-Wise Balance Loss":
    failures.append(f"section 13 not in table of contents, got {s13['toc']!r}")
else:
    print(f"  ok  {'table of contents':<24} lists section 13")

print("\n=== section 14: multi-token prediction ===")
s14 = payload["s14"]
seq14 = int(s14["seq"])
dm14 = int(s14["d_model"])
vocab14 = int(s14["vocab"])
hor14 = int(s14["horizon"])
al14 = int(s14["aligned"])
dk14 = int(s14["d_k"])
eh14 = int(s14["expert_hidden"])
eps14 = float(s14["rms_eps"])
lam14 = float(s14["lambda"])


def rmsnorm(block, gamma, rows, cols, eps):
    rms = [math.sqrt(sum(block[i][j] ** 2 for i in range(rows)) / rows + eps) for j in range(cols)]
    normed = [[block[i][j] / rms[j] * gamma[i] for j in range(cols)] for i in range(rows)]
    return rms, normed


print("the alignment (this is the section)")
tokens14 = [int(t) for t in s14["tokens"]]
checks += 1
if al14 != seq14 - hor14:
    failures.append(f"aligned {al14} != seq - horizon {seq14 - hor14}")
else:
    print(f"  ok  {'aligned = seq - horizon':<24} {al14} = {seq14} - {hor14}")

checks += 1
want_input = tokens14[hor14 - 1 : -1]
if [int(t) for t in s14["input_tok"]] != want_input:
    failures.append(f"input tokens {s14['input_tok']} != t({hor14}..{seq14 - 1}) {want_input}")
else:
    print(f"  ok  {'input = t(i+1)':<24} {want_input}  (make_mtp_input_tokens)")

checks += 1
want_target = tokens14[hor14:]
if [int(t) for t in s14["target_tok"]] != want_target:
    failures.append(f"targets {s14['target_tok']} != t({hor14 + 1}..{seq14}) {want_target}")
else:
    print(f"  ok  {'target = t(i+2)':<24} {want_target}  (make_future_targets)")

# The trap: input must be one step before the target, never the target itself.
checks += 1
if [int(t) for t in s14["input_tok"]] == [int(t) for t in s14["target_tok"]]:
    failures.append("input tokens equal the targets: the objective is degenerate (GATE_U)")
else:
    offsets = [want_target[i] for i in range(al14)]
    print(f"  ok  {'input is not the target':<24} offset by one, objective is not degenerate")

checks += 1
if s14["h_aligned"] != [row[:al14] for row in s14["h_full"]]:
    failures.append("h aligned is not the first (seq - horizon) columns of h")
else:
    print(f"  ok  {'h aligned':<24} first {al14} of {seq14} columns (align_hidden_states)")

print("\nembedding lookup and merge")
E14 = s14["E"]

# Unit columns are what make the degenerate-objective panel deterministic.
checks += 1
worst_norm = max(
    abs(math.sqrt(sum(E14[r][c] ** 2 for r in range(dm14))) - 1.0) for c in range(vocab14)
)
if worst_norm > 1e-12:
    failures.append(f"embedding columns are not unit length: worst {worst_norm:.2e}")
else:
    print(f"  ok  {'E columns are unit norm':<24} worst deviation {worst_norm:.2e}")

close("Emb(t(i+1)) = E[:, t]", s14["emb"], [
    [E14[i][int(s14["input_tok"][j]) - 1] for j in range(al14)] for i in range(dm14)
])

rms_h_want, norm_h_want = rmsnorm(s14["h_aligned"], s14["gamma_h"], dm14, al14, eps14)
rms_e_want, norm_e_want = rmsnorm(s14["emb"], s14["gamma_e"], dm14, al14, eps14)
close("rms_h", [s14["rms_h"]], [rms_h_want])
close("rms_e", [s14["rms_e"]], [rms_e_want])
close("concat = norm_h over norm_e", s14["concat"], norm_h_want + norm_e_want)
close("merged = M concat", s14["merged"], matmul(s14["M"], s14["concat"]))

checks += 1
if len(s14["concat"]) != 2 * dm14:
    failures.append(f"concat has {len(s14['concat'])} rows, expected {2 * dm14}")
else:
    print(f"  ok  {'concat is 2*d_model tall':<24} {2 * dm14} rows into a {dm14}-wide merge")

print("\nthe transformer block")
rms3_want, n3_want = rmsnorm(s14["merged"], s14["gamma3"], dm14, al14, eps14)
close("rms3", [s14["rms3"]], [rms3_want])
close("n3", s14["n3"], n3_want)
close("Q = Wq n3", s14["Q"], matmul(s14["Wq"], s14["n3"]))
close("K = Wk n3", s14["K"], matmul(s14["Wk"], s14["n3"]))
close("V = Wv n3", s14["Vmat"], matmul(s14["Wv"], s14["n3"]))
close("K^T", s14["KT"], transpose(s14["K"]))
close("S = K^T Q / sqrt(dk)", s14["S"], [
    [v / math.sqrt(dk14) for v in row] for row in matmul(s14["KT"], s14["Q"])
])

A14 = s14["A"]
checks += 1
leak14 = [(i, j) for i in range(al14) for j in range(al14) if i > j and abs(A14[i][j]) > 0]
if leak14:
    failures.append(f"MTP block attention leaks to future positions at {leak14[:5]}")
else:
    print(f"  ok  {'MTP block is causal':<24} no position attends forward")

S14 = s14["S"]
expected_A14 = [[0.0] * al14 for _ in range(al14)]
for j in range(al14):
    total = sum(math.exp(S14[i][j]) for i in range(j + 1))
    for i in range(j + 1):
        expected_A14[i][j] = math.exp(S14[i][j]) / total
close("A = softmax over prefix", A14, expected_A14)
close("O_attn = V A", s14["O_attn"], matmul(s14["Vmat"], A14))
close("attn_out = W_O O_attn", s14["attn_out"], matmul(s14["W_O"], s14["O_attn"]))
close("r1 = merged + attn_out", s14["r1"], [
    [s14["merged"][i][j] + s14["attn_out"][i][j] for j in range(al14)] for i in range(dm14)
])

rms4_want, n4_want = rmsnorm(s14["r1"], s14["gamma4"], dm14, al14, eps14)
close("rms4", [s14["rms4"]], [rms4_want])
close("n4", s14["n4"], n4_want)
close("g = silu(W_g n4)", s14["g"], [
    [z / (1 + math.exp(-z)) for z in row] for row in matmul(s14["W_gate"], s14["n4"])
])
close("v = W_v n4", s14["ffn_v"], matmul(s14["W_value"], s14["n4"]))
close("ffn_out = W_o (g*v)", s14["ffn_out"], matmul(s14["W_out"], [
    [s14["g"][i][j] * s14["ffn_v"][i][j] for j in range(al14)] for i in range(eh14)
]))
close("refined = r1 + ffn_out", s14["refined"], [
    [s14["r1"][i][j] + s14["ffn_out"][i][j] for j in range(al14)] for i in range(dm14)
])

print("\ntied output head and loss")
rms5_want, nf_want = rmsnorm(s14["refined"], s14["gamma5"], dm14, al14, eps14)
close("rms5", [s14["rms5"]], [rms5_want])
close("nf", s14["nf"], nf_want)
# tied: the output head IS the embedding transposed
close("logits = E^T nf", s14["logits"], matmul(transpose(E14), s14["nf"]))

probs14 = s14["probs"]
expected_p14 = []
for i in range(vocab14):
    expected_p14.append([0.0] * al14)
for j in range(al14):
    col = [s14["logits"][i][j] for i in range(vocab14)]
    total = sum(math.exp(v) for v in col)
    for i in range(vocab14):
        expected_p14[i][j] = math.exp(col[i]) / total
close("p = softmax(logits)", probs14, expected_p14)

checks += 1
worst_p = max(abs(sum(probs14[i][j] for i in range(vocab14)) - 1.0) for j in range(al14))
if worst_p > 1e-12:
    failures.append(f"probability columns do not sum to 1: worst {worst_p:.2e}")
else:
    print(f"  ok  {'p sums to 1 per position':<24} worst deviation {worst_p:.2e}")

close("p at target", [s14["p_target"]], [
    [probs14[int(s14["target_tok"][j]) - 1][j] for j in range(al14)]
])
close("loss = -ln(p)", [s14["loss_pos"]], [[-math.log(p) for p in s14["p_target"]]])
close("mtp loss = mean", [[s14["mtp_loss"]]], [[sum(s14["loss_pos"]) / al14]])
close("combined = main + lam*mtp", [[s14["combined"]]], [
    [s14["main_loss"] + lam14 * s14["mtp_loss"]]
])

checks += 1
if abs(s14["lambda_50"] - lam14) > 1e-12 or abs(s14["lambda_80"] - float(s14["lambda_final"])) > 1e-12:
    failures.append(f"lambda schedule wrong: 50% -> {s14['lambda_50']}, 80% -> {s14['lambda_80']}")
else:
    print(f"  ok  {'lambda schedule':<24} {lam14} before {s14['decay_fraction']}, {s14['lambda_final']} after")

print("\nspeculative draft head")
checks += 1
prop_want = []
for j in range(al14):
    col = [s14["logits"][i][j] for i in range(vocab14)]
    prop_want.append(max(range(vocab14), key=lambda i: col[i]) + 1)
if [int(p) for p in s14["proposal"]] != prop_want:
    failures.append(f"proposals {s14['proposal']} != argmax {prop_want}")
else:
    print(f"  ok  {'proposal = argmax':<24} {prop_want}")

close("accepted = proposal==target", [s14["accepted"]], [
    [1.0 if int(s14["proposal"][j]) == int(s14["target_tok"][j]) else 0.0 for j in range(al14)]
])
close("acceptance rate", [[s14["accept_rate"]]], [[sum(s14["accepted"]) / al14]])

print("\nthe degenerate-objective demonstration")
# E^T e_target must peak at the target index: that is exactly why feeding the
# target as input lets a tied head score it without learning anything.
checks += 1
target1 = int(s14["target_tok"][0])
dots = s14["tied_dots"]
peak = max(range(vocab14), key=lambda i: dots[i]) + 1
runner_up = max(dots[i] for i in range(vocab14) if i != target1 - 1)
if peak != target1:
    failures.append(f"E^T e_target peaks at {peak}, expected the target {target1}")
elif abs(dots[target1 - 1] - 1.0) > 1e-12:
    failures.append(f"self dot is {dots[target1 - 1]}, expected exactly 1.0 for unit columns")
else:
    print(f"  ok  {'E^T e_target peaks at target':<24} exactly 1.00 vs {runner_up:+.2f} runner-up")

close("E^T e_target", [dots], [
    [sum(E14[r][i] * E14[r][target1 - 1] for r in range(dm14)) for i in range(vocab14)]
])

checks += 1
if s14["toc"].strip() != "# Multi-Token Prediction":
    failures.append(f"section 14 not in table of contents, got {s14['toc']!r}")
else:
    print(f"  ok  {'table of contents':<24} lists section 14")

print()
if failures:
    print(f"FAILED {len(failures)} of {checks} checks")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"PASSED all {checks} checks")
