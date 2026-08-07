# Builds "DeepSeek-V3 by Hand" via Excel COM.
#
# Excel is the only thing that writes dynamic-array formulas correctly: a
# spilling formula needs <f t="array" ref="..."> plus cm="1" cell metadata and
# an xl/metadata.xml part. Generating that by hand is not worth it, so this
# script drives Excel and lets it emit the file.
#
# This Excel build has no LAMBDA family (LAMBDA, BYCOL, MAKEARRAY, TOCOL,
# VSTACK, HSTACK, TAKE, CHOOSEROWS all return #NAME?). Every formula below
# stays inside the supported set. See excel/CAPABILITIES.md.
#
# Usage: pwsh -File excel/build_sheet.ps1

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$out = Join-Path $root 'excel\deepseekv3_by_hand.xlsx'
$dump = Join-Path $root 'excel\_verify_dump.json'

# ---------------------------------------------------------------- constants
$xlCenter = -4108
$xlRight = -4152
$xlContinuous = 1
$xlThin = 2
$xlThick = 4
$xlEdgeBottom = 9
$xlCalculationManual = -4135
$xlCalculationAutomatic = -4105
$orangeFill = 14083579   # Orange Accent 2, Lighter 80%  (BGR of 251,229,214)
$headerColor = 6970692   # 44546A dark slate            (BGR of 68,84,106)
$barColor = 5920255      # FF555A                       (BGR of 255,85,90)

# ---------------------------------------------------------------- launch
$xl = New-Object -ComObject Excel.Application
$xl.Visible = $false
$xl.DisplayAlerts = $false
$xl.ScreenUpdating = $false

$wb = $xl.Workbooks.Add()
while ($wb.Worksheets.Count -gt 1) { $wb.Worksheets.Item($wb.Worksheets.Count).Delete() }
$ws = $wb.Worksheets.Item(1)
$ws.Name = 'deepseekv3'

# ---------------------------------------------------------------- canvas
# His grid: 4.875-wide columns, 24-high rows, Aptos Narrow 12, gridlines on,
# General number format. Narrow columns are what round the display; there is
# no number formatting anywhere in his sheet.
$ws.StandardWidth = 4.875
$ws.Cells.RowHeight = 24
$ws.Cells.Font.Name = 'Aptos Narrow'
$ws.Cells.Font.Size = 12

# Writing the same cell twice is silent: the second write wins, no Excel error
# appears, and a label just vanishes. That happened at V286, where a heading
# landed on top of the shared "theta = pos" label. Track every target and
# report collisions at the end of the build.
$script:written = @{}
function Note-Write([string]$addr) {
  if ($script:written.ContainsKey($addr)) { $script:written[$addr] += 1 }
  else { $script:written[$addr] = 1 }
}

# PowerShell caches the COM member binding per call site, so a single helper
# cannot set both strings and numbers: whichever type is passed first wins and
# the other throws an InvalidCastException. Hence two setters.
function Set-Text([string]$addr, [string]$text) { Note-Write $addr; $ws.Range($addr).Value2 = $text }
function Set-Num([string]$addr, [double]$n) { Note-Write $addr; $ws.Range($addr).Value2 = $n }
function Set-Formula([string]$addr, [string]$f) { Note-Write $addr; $ws.Range($addr).Formula2 = $f }
function Set-Center($addr) { $ws.Range($addr).HorizontalAlignment = $xlCenter }
function Set-Right($addr) { $ws.Range($addr).HorizontalAlignment = $xlRight }
function Set-Bold($addr) { $ws.Range($addr).Font.Bold = $true }
function Set-Weights($addr) {
  $r = $ws.Range($addr)
  $r.Interior.Color = $orangeFill
  $r.HorizontalAlignment = $xlCenter
}
function Set-Box($addr) {
  $r = $ws.Range($addr)
  $r.BorderAround($xlContinuous, $xlThin) | Out-Null
  $r.HorizontalAlignment = $xlCenter
}
# The only number formatting in the workbook. His sheet uses General
# everywhere and lets the narrow column do the rounding, which works right up
# until a cell holds 1e-6: General renders it as a bare "0" and the reader
# cannot see the value at all. Used only for the genuinely small constants.
function Set-Sci([string]$addr) { $ws.Range($addr).NumberFormat = '0.0E+00' }

# ---------------------------------------------------------------- title
Set-Text 'A1' 'DeepSeek-V3 by Hand'
$ws.Range('A1').Font.Size = 36
$ws.Range('A1').Font.Bold = $true
$ws.Range('A1').Font.Color = $headerColor
$ws.Rows.Item(1).RowHeight = 63.75

Set-Text 'A2' 'Every matrix computed in the grid. Companion to src/compact_v3.'

Set-Text 'A4' 'Table of Content'
Set-Bold 'A4'
# Same navigation trick he uses: find every cell in column A starting with "#",
# build the anchor, link to it, and indent by how many hashes it carries. B5
# spills one row per section on its own; the link and label columns need one
# formula each, so they are written per section. Writing more rows than there
# are sections would make INDIRECT fail on an empty anchor and show as an
# error cell, so this count has to track $sectionCount.
$sectionCount = 14
Set-Formula 'B5' '=FILTER("A"&ROW(A:A), LEFT(A:A,1)="#")'
for ($i = 0; $i -lt $sectionCount; $i++) {
  $r = 5 + $i
  Set-Formula "C$r" ('=HYPERLINK("#"&B' + $r + ', UNICHAR(128279))')
  Set-Formula "D$r" ('=LET(n, INDIRECT(B' + $r + '), indent, 9, k, LEN(n) - LEN(SUBSTITUTE(n,"#","")), IF(k>=2, REPT(" ", indent*(k-1)) & n, n))')
  Set-Center "C$r"
}

# ================================================================ SECTION 1
Set-Text 'A21' '# Attention'
$ws.Range('A21').Font.Size = 36
$ws.Range('A21').Font.Bold = $true
$ws.Range('A21').Font.Color = $headerColor
$ws.Rows.Item(21).RowHeight = 48
$b = $ws.Range('A21:BB21').Borders.Item($xlEdgeBottom)
$b.LineStyle = $xlContinuous
$b.Weight = $xlThick

# --- shape parameters, referenced by every formula below so one edit resizes
#     the whole section
Set-Text 'E24' 'seq';     Set-Num 'F24' 6
Set-Text 'E25' 'd_model'; Set-Num 'F25' 8
Set-Text 'E26' 'd_k';     Set-Num 'F26' 4
Set-Text 'E27' 'd_v';     Set-Num 'F27' 3
Set-Right 'E24:E27'
Set-Center 'F24:F27'

# --- input
Set-Text 'N24' 'pos'
Set-Right 'N24'
Set-Formula 'O24' '=SEQUENCE(1,F24,1)'
Set-Center 'O24:T24'

Set-Formula 'O26' '="x"&O24#'
Set-Center 'O26:T26'
# X is centred for the same reason the weights are. A non-negative X gives
# every token a large shared mean component, so every q and k points partly in
# one common direction and every dot product inherits a positive offset. The
# scores then sit in [-0.1, +2.2] instead of straddling zero. Real hidden
# states arrive here after RMSNorm, so centred is also the honest picture.
Set-Text 'N27' 'X'
Set-Right 'N27'
Set-Formula 'O27' '=(RANDARRAY(F25,F24)-0.5)*2'
Set-Box 'O27:T34'

# --- QKV projection
Set-Text 'D36' 'QKV Projection'
Set-Bold 'D36'

Set-Formula 'O37' '="q"&O24#'
Set-Center 'O37:T37'
# Projection weights are uniform on [-1,1], not [0,1]. Two failure modes sit
# either side of this. Uniform [0,1] weights against a non-negative X make
# every score large and positive, softmax saturates, and one key takes ~0.9 of
# every column. Centring but leaving the width at 1 (RANDARRAY-0.5) drives the
# logits down to ~0.2 and softmax goes uniform. Both collapse O to identical
# columns and hide what attention does. [-1,1] puts the logits near unit
# variance. He centres the same way in Dynamic Gating (=RANDARRAY(4,2)-0.5).
Set-Text 'D38' 'Wq'
Set-Right 'D38'
Set-Formula 'E38' '=(RANDARRAY(F26,F25)-0.5)*2'
Set-Weights 'E38:L41'
Set-Text 'N38' 'Q'
Set-Right 'N38'
Set-Formula 'O38' '=MMULT(E38#,O27#)'
Set-Box 'O38:T41'

Set-Formula 'O43' '="k"&O24#'
Set-Center 'O43:T43'
Set-Text 'D44' 'Wk'
Set-Right 'D44'
Set-Formula 'E44' '=(RANDARRAY(F26,F25)-0.5)*2'
Set-Weights 'E44:L47'
Set-Text 'N44' 'K'
Set-Right 'N44'
Set-Formula 'O44' '=MMULT(E44#,O27#)'
Set-Box 'O44:T47'

Set-Formula 'O49' '="v"&O24#'
Set-Center 'O49:T49'
Set-Text 'D50' 'Wv'
Set-Right 'D50'
Set-Formula 'E50' '=(RANDARRAY(F27,F25)-0.5)*2'
Set-Weights 'E50:L52'
Set-Text 'N50' 'V'
Set-Right 'N50'
Set-Formula 'O50' '=MMULT(E50#,O27#)'
Set-Box 'O50:T52'

# --- attention
Set-Text 'V36' 'K^T'
Set-Formula 'V37' '=TRANSPOSE("k"&O24#)'
Set-Right 'V37:V42'
Set-Formula 'W37' '=TRANSPOSE(O44#)'
Set-Box 'W37:Z42'

Set-Text 'AB36' 'D = K^T Q'
Set-Formula 'AB35' '="q"&O24#'
Set-Center 'AB35:AG35'
Set-Formula 'AA37' '=TRANSPOSE("k"&O24#)'
Set-Right 'AA37:AA42'
Set-Formula 'AB37' '=MMULT(W37#,O38#)'
Set-Box 'AB37:AG42'

Set-Text 'AI36' 'S = D / sqrt(dk)'
Set-Formula 'AI35' '="q"&O24#'
Set-Center 'AI35:AN35'
Set-Formula 'AH37' '=TRANSPOSE("k"&O24#)'
Set-Right 'AH37:AH42'
Set-Formula 'AI37' '=AB37#/SQRT(ROWS(O38#))'
Set-Box 'AI37:AN42'

# Softmax down each column. His section 1 does exactly this, one LET per
# column, because the whole-block BYCOL form needs LAMBDA.
Set-Text 'AP36' 'A = Softmax(S)'
Set-Formula 'AP35' '="q"&O24#'
Set-Center 'AP35:AU35'
Set-Formula 'AO37' '=TRANSPOSE("k"&O24#)'
Set-Right 'AO37:AO42'
$softmaxCols = @(
  @('AP37', 'AI37:AI42'), @('AQ37', 'AJ37:AJ42'), @('AR37', 'AK37:AK42'),
  @('AS37', 'AL37:AL42'), @('AT37', 'AM37:AM42'), @('AU37', 'AN37:AN42')
)
foreach ($pair in $softmaxCols) {
  Set-Formula $pair[0] ('=LET(z,' + $pair[1] + ',EXP(z)/SUM(EXP(z)))')
}
Set-Box 'AP37:AU42'

Set-Text 'AW43' 'O = V A'
Set-Formula 'AW44' '="o"&O24#'
Set-Center 'AW44:BB44'
# AP37# is only the first softmax column, so the whole block goes in by range.
Set-Formula 'AW45' '=MMULT(O50#,AP37:AU42)'
Set-Box 'AW45:BB47'

# Attention weights get his red data bar, fixed 0..1 scale.
try {
  $fc = $ws.Range('AP37:AU42').FormatConditions.AddDatabar()
  $fc.MinPoint.Modify(0, 0) | Out-Null
  $fc.MaxPoint.Modify(0, 1) | Out-Null
  $fc.BarColor.Color = $barColor
} catch {
  Write-Host "data bar skipped: $($_.Exception.Message)"
}

# ================================================================ SECTION 2
# Minimal delta on section 1: same X, same projections, same D and S. The only
# new machinery is the causal mask and what it costs you at decode time. Each
# section re-rolls its own inputs so it can be read standalone, which is how
# his sheet is organised.
Set-Text 'A60' '# Causal Attention and the KV Cache'
$ws.Range('A60').Font.Size = 36
$ws.Range('A60').Font.Bold = $true
$ws.Range('A60').Font.Color = $headerColor
$ws.Rows.Item(60).RowHeight = 48
$b2 = $ws.Range('A60:BP60').Borders.Item($xlEdgeBottom)
$b2.LineStyle = $xlContinuous
$b2.Weight = $xlThick

Set-Text 'E63' 'seq';     Set-Num 'F63' 6
Set-Text 'E64' 'd_model'; Set-Num 'F64' 8
Set-Text 'E65' 'd_k';     Set-Num 'F65' 4
Set-Text 'E66' 'd_v';     Set-Num 'F66' 3
Set-Right 'E63:E66'
Set-Center 'F63:F66'

Set-Text 'N63' 'pos'
Set-Right 'N63'
Set-Formula 'O63' '=SEQUENCE(1,F63,1)'
Set-Center 'O63:T63'

Set-Formula 'O65' '="x"&O63#'
Set-Center 'O65:T65'
Set-Text 'N66' 'X'
Set-Right 'N66'
Set-Formula 'O66' '=(RANDARRAY(F64,F63)-0.5)*2'
Set-Box 'O66:T73'

Set-Text 'D75' 'QKV Projection'
Set-Bold 'D75'

Set-Formula 'O76' '="q"&O63#'
Set-Center 'O76:T76'
Set-Text 'D77' 'Wq'
Set-Right 'D77'
Set-Formula 'E77' '=(RANDARRAY(F65,F64)-0.5)*2'
Set-Weights 'E77:L80'
Set-Text 'N77' 'Q'
Set-Right 'N77'
Set-Formula 'O77' '=MMULT(E77#,O66#)'
Set-Box 'O77:T80'

Set-Formula 'O82' '="k"&O63#'
Set-Center 'O82:T82'
Set-Text 'D83' 'Wk'
Set-Right 'D83'
Set-Formula 'E83' '=(RANDARRAY(F65,F64)-0.5)*2'
Set-Weights 'E83:L86'
Set-Text 'N83' 'K'
Set-Right 'N83'
Set-Formula 'O83' '=MMULT(E83#,O66#)'
Set-Box 'O83:T86'

Set-Formula 'O88' '="v"&O63#'
Set-Center 'O88:T88'
Set-Text 'D89' 'Wv'
Set-Right 'D89'
Set-Formula 'E89' '=(RANDARRAY(F66,F64)-0.5)*2'
Set-Weights 'E89:L91'
Set-Text 'N89' 'V'
Set-Right 'N89'
Set-Formula 'O89' '=MMULT(E89#,O66#)'
Set-Box 'O89:T91'

# --- unchanged from section 1 up to S
Set-Text 'V75' 'K^T'
Set-Formula 'V76' '=TRANSPOSE("k"&O63#)'
Set-Right 'V76:V81'
Set-Formula 'W76' '=TRANSPOSE(O83#)'
Set-Box 'W76:Z81'

Set-Text 'AB75' 'D = K^T Q'
Set-Formula 'AB74' '="q"&O63#'
Set-Center 'AB74:AG74'
Set-Formula 'AA76' '=TRANSPOSE("k"&O63#)'
Set-Right 'AA76:AA81'
Set-Formula 'AB76' '=MMULT(W76#,O77#)'
Set-Box 'AB76:AG81'

Set-Text 'AI75' 'S = D / sqrt(dk)'
Set-Formula 'AI74' '="q"&O63#'
Set-Center 'AI74:AN74'
Set-Formula 'AH76' '=TRANSPOSE("k"&O63#)'
Set-Right 'AH76:AH81'
Set-Formula 'AI76' '=AB76#/SQRT(ROWS(O77#))'
Set-Box 'AI76:AN81'

# --- the delta starts here
# Rows are keys i, columns are queries j, so query j may read key i only when
# i <= j. No MAKEARRAY needed: two SEQUENCEs broadcast against each other.
Set-Text 'AP75' 'M = Causal Mask'
Set-Formula 'AP74' '="q"&O63#'
Set-Center 'AP74:AU74'
Set-Formula 'AO76' '=TRANSPOSE("k"&O63#)'
Set-Right 'AO76:AO81'
Set-Formula 'AP76' '=IF(SEQUENCE(F63,1)<=SEQUENCE(1,F63),1,0)'
Set-Box 'AP76:AU81'

# The triangle made visible. Blanking the masked half is the whole point of
# this block: it is the clearest single picture of causality on the sheet.
Set-Text 'AW75' "S' = S masked"
Set-Formula 'AW74' '="q"&O63#'
Set-Center 'AW74:BB74'
Set-Formula 'AV76' '=TRANSPOSE("k"&O63#)'
Set-Right 'AV76:AV81'
Set-Formula 'AW76' '=IF(AP76#=1,AI76#,"")'
Set-Box 'AW76:BB81'

# Softmax over the allowed keys only. Excel has no -inf, so instead of adding
# a large negative before EXP, the mask multiplies the exponentials. Same
# result, no overflow, and the zeros are exact rather than 1e-300.
Set-Text 'BD75' "A = Softmax(S')"
Set-Formula 'BD74' '="q"&O63#'
Set-Center 'BD74:BI74'
Set-Formula 'BC76' '=TRANSPOSE("k"&O63#)'
Set-Right 'BC76:BC81'
$causalCols = @(
  @('BD76', 'AI76:AI81', 'AP76:AP81'), @('BE76', 'AJ76:AJ81', 'AQ76:AQ81'),
  @('BF76', 'AK76:AK81', 'AR76:AR81'), @('BG76', 'AL76:AL81', 'AS76:AS81'),
  @('BH76', 'AM76:AM81', 'AT76:AT81'), @('BI76', 'AN76:AN81', 'AU76:AU81')
)
foreach ($t in $causalCols) {
  Set-Formula $t[0] ('=LET(z,' + $t[1] + ',m,' + $t[2] + ',m*EXP(z)/SUM(m*EXP(z)))')
}
Set-Box 'BD76:BI81'
try {
  $fc2 = $ws.Range('BD76:BI81').FormatConditions.AddDatabar()
  $fc2.MinPoint.Modify(0, 0) | Out-Null
  $fc2.MaxPoint.Modify(0, 1) | Out-Null
  $fc2.BarColor.Color = $barColor
} catch { Write-Host "section 2 data bar skipped: $($_.Exception.Message)" }

Set-Text 'BK74' 'O = V A'
Set-Formula 'BK75' '="o"&O63#'
Set-Center 'BK75:BP75'
Set-Formula 'BK76' '=MMULT(O89#,BD76:BI81)'
Set-Box 'BK76:BP78'

# --- what causality costs at decode time
Set-Text 'E94' 'KV Cache'
Set-Bold 'E94'
Set-Text 'N95' 't'
Set-Right 'N95'
Set-Formula 'O95' '=SEQUENCE(1,F63,1)'
Set-Center 'O95:T95'
Set-Text 'N96' 'K floats'
Set-Right 'N96'
Set-Formula 'O96' '=O95#*F65'
Set-Center 'O96:T96'
Set-Text 'N97' 'V floats'
Set-Right 'N97'
Set-Formula 'O97' '=O95#*F66'
Set-Center 'O97:T97'
Set-Text 'N98' 'cached'
Set-Right 'N98'
Set-Formula 'O98' '=O96#+O97#'
Set-Box 'O98:T98'

Set-Text 'E100' 'floats per token'
Set-Right 'E100'
Set-Formula 'F100' '=F65+F66'
Set-Text 'E101' 'context'
Set-Right 'E101'
Set-Num 'F101' 512
Set-Text 'E102' 'cache at context'
Set-Right 'E102'
Set-Formula 'F102' '=F100*F101'
Set-Center 'F100:F102'

Set-Text 'V94' 'Decode t = 6: only k6 and v6 are new. k1..k5 and v1..v5 are read back'
Set-Text 'V95' 'from cache, never recomputed. That is the row above growing linearly.'

# ================================================================ SECTION 3
# Low-Rank Compression. The smallest section: no attention, just
# c = W_down x, x_hat = W_up c, and the reconstruction error beside the
# original. Stands alone, exactly like every other section, and sets up MLA:
# this is the W_UK / W_UV pair from mla.py at rank kv_lora_rank = 3.
Set-Text 'A108' '# Low-Rank Compression'
$ws.Range('A108').Font.Size = 36
$ws.Range('A108').Font.Bold = $true
$ws.Range('A108').Font.Color = $headerColor
$ws.Rows.Item(108).RowHeight = 48
$b3 = $ws.Range('A108:BV108').Borders.Item($xlEdgeBottom)
$b3.LineStyle = $xlContinuous
$b3.Weight = $xlThick

# --- shape parameters: same numbers the handoff fixes for all 14 sections
Set-Text 'E111' 'seq';          Set-Num 'F111' 6
Set-Text 'E112' 'd_model';      Set-Num 'F112' 8
Set-Text 'E113' 'd_compressed'; Set-Num 'F113' 3
Set-Right 'E111:E113'
Set-Center 'F111:F113'

# --- input: X is centred on [-1,1], the same rule as sections 1 and 2
Set-Text 'N111' 'pos'
Set-Right 'N111'
Set-Formula 'O111' '=SEQUENCE(1,F111,1)'
Set-Center 'O111:T111'

Set-Formula 'O113' '="x"&O111#'
Set-Center 'O113:T113'
Set-Text 'N114' 'X'
Set-Right 'N114'
Set-Formula 'O114' '=(RANDARRAY(F112,F111)-0.5)*2'
Set-Box 'O114:T121'

# --- compression: c = W_down x
Set-Text 'D123' 'Compress'
Set-Bold 'D123'
Set-Text 'D124' 'W_down'
Set-Right 'D124'
Set-Formula 'E124' '=(RANDARRAY(F113,F112)-0.5)*2'
Set-Weights 'E124:L126'
Set-Text 'N124' 'c'
Set-Right 'N124'
Set-Formula 'O124' '=MMULT(E124#,O114#)'
Set-Box 'O124:T126'

# --- reconstruction: x_hat = W_up c
Set-Text 'D129' 'Reconstruct'
Set-Bold 'D129'
Set-Text 'D130' 'W_up'
Set-Right 'D130'
Set-Formula 'E130' '=(RANDARRAY(F112,F113)-0.5)*2'
Set-Weights 'E130:G137'
Set-Text 'N130' 'x_hat'
Set-Right 'N130'
Set-Formula 'O130' '=MMULT(E130#,O124#)'
Set-Box 'O130:T137'

# --- reconstruction error, beside the original X so the loss is visible
Set-Text 'W112' 'x_hat - X'
Set-Bold 'W112'
Set-Formula 'W113' '="x"&O111#'
Set-Center 'W113:AB113'
Set-Formula 'W114' '=O130#-O114#'
Set-Box 'W114:AB121'

# ================================================================ SECTION 4
# Multi-head vs Latent Attention. Two panels on one X so the difference is
# literal. Left, plain multi-head attention: per-head K and V are projected
# straight from x, and the cache has to hold all 2x3 rows of each, 12 floats
# per token. Right, the latent version (MLA): x is compressed once to c_KV
# (rank 3), the cache holds only that, and K and V are rebuilt per head at
# read time. Mirrors MultiHeadLatentAttention in src/compact_v3/mla.py:
# kv_down (d_model -> kv_lora_rank), then k_content_up and v_up
# (kv_lora_rank -> n_heads * head_dim). Same shapes, 12 vs 3 floats cached.
Set-Text 'A145' '# Multi-head vs Latent Attention'
$ws.Range('A145').Font.Size = 36
$ws.Range('A145').Font.Bold = $true
$ws.Range('A145').Font.Color = $headerColor
$ws.Rows.Item(145).RowHeight = 48
$b4 = $ws.Range('A145:CD145').Borders.Item($xlEdgeBottom)
$b4.LineStyle = $xlContinuous
$b4.Weight = $xlThick

# --- shape parameters: the hand-scale config, all 14 sections
Set-Text 'E148' 'seq';       Set-Num 'F148' 6
Set-Text 'E149' 'd_model';   Set-Num 'F149' 8
Set-Text 'E150' 'kv_lora_rank'; Set-Num 'F150' 3
Set-Text 'E151' 'n_heads';   Set-Num 'F151' 2
Set-Text 'E152' 'head_dim';  Set-Num 'F152' 3
Set-Right 'E148:E152'
Set-Center 'F148:F152'

# --- input, re-rolled and centred like every section
Set-Text 'N148' 'pos'
Set-Right 'N148'
Set-Formula 'O148' '=SEQUENCE(1,F148,1)'
Set-Center 'O148:T148'

Set-Formula 'O150' '="x"&O148#'
Set-Center 'O150:T150'
Set-Text 'N151' 'X'
Set-Right 'N151'
Set-Formula 'O151' '=(RANDARRAY(F149,F148)-0.5)*2'
Set-Box 'O151:T158'

# --- left panel: multi-head attention, per-head K and V straight from x
Set-Text 'D160' 'Multi-head Attention'
Set-Bold 'D160'
Set-Text 'D161' 'W_K'
Set-Right 'D161'
Set-Formula 'E161' '=(RANDARRAY(F151*F152,F149)-0.5)*2'
Set-Weights 'E161:L166'
Set-Text 'N161' 'K'
Set-Right 'N161'
Set-Formula 'O161' '=MMULT(E161#,O151#)'
Set-Box 'O161:T166'
Set-Text 'U160' 'heads'
Set-Bold 'U160'
Set-Formula 'U161' '=SEQUENCE(F151*F152,1)'
Set-Right 'U161:U166'

Set-Text 'D168' 'W_V'
Set-Right 'D168'
Set-Formula 'E168' '=(RANDARRAY(F151*F152,F149)-0.5)*2'
Set-Weights 'E168:L173'
Set-Text 'N168' 'V'
Set-Right 'N168'
Set-Formula 'O168' '=MMULT(E168#,O151#)'
Set-Box 'O168:T173'
Set-Formula 'U168' '=SEQUENCE(F151*F152,1)'
Set-Right 'U168:U173'

# --- right panel: latent attention, compress once then rebuild per head
Set-Text 'W160' 'Latent Attention'
Set-Bold 'W160'
Set-Text 'V161' 'W_DKV'
Set-Right 'V161'
Set-Formula 'W161' '=(RANDARRAY(F150,F149)-0.5)*2'
Set-Weights 'W161:AD163'
Set-Text 'AF161' 'c_KV'
Set-Right 'AF161'
Set-Formula 'AG161' '=MMULT(W161#,O151#)'
Set-Box 'AG161:AL163'

Set-Text 'V168' 'W_UK'
Set-Right 'V168'
Set-Formula 'W168' '=(RANDARRAY(F151*F152,F150)-0.5)*2'
Set-Weights 'W168:Y173'
Set-Text 'AF168' 'K'
Set-Right 'AF168'
Set-Formula 'AG168' '=MMULT(W168#,AG161#)'
Set-Box 'AG168:AL173'
Set-Text 'AM167' 'heads'
Set-Bold 'AM167'
Set-Formula 'AM168' '=SEQUENCE(F151*F152,1)'
Set-Right 'AM168:AM173'

Set-Text 'V175' 'W_UV'
Set-Right 'V175'
Set-Formula 'W175' '=(RANDARRAY(F151*F152,F150)-0.5)*2'
Set-Weights 'W175:Y180'
Set-Text 'AF175' 'V'
Set-Right 'AF175'
Set-Formula 'AG175' '=MMULT(W175#,AG161#)'
Set-Box 'AG175:AL180'
Set-Formula 'AM175' '=SEQUENCE(F151*F152,1)'
Set-Right 'AM175:AM180'

# --- cache ledger: what a KV cache costs per token
Set-Text 'E184' 'cache per token'
Set-Bold 'E184'
Set-Text 'E186' 'MHA'
Set-Right 'E186'
Set-Text 'E187' 'MLA'
Set-Right 'E187'
Set-Text 'F185' 'floats'
Set-Right 'F185'
Set-Formula 'F186' '=F151*(F152+F152)'
Set-Formula 'F187' '=F150'
Set-Center 'F186:F187'

# ================================================================ SECTION 5
# The RoPE Conflict. The point: RoPE is applied to the per-head K in real
# MLA, but MLA only has the compressed c_KV. If you try RoPE(W_UK c) instead
# of W_UK RoPE(c), the two disagree. Two summary cells that should match and
# do not. This is why section 6 decouples RoPE into its own key head.
# Mirrors rotate_half in src/compact_v3/rope.py: a head is [x0,x1,x2], the
# rotary pair is dims 0-1, angle at position p is p radians (base 10000 with
# a 2-dim rotary pair), and x2 stays put.
Set-Text 'A195' '# The RoPE Conflict'
$ws.Range('A195').Font.Size = 36
$ws.Range('A195').Font.Bold = $true
$ws.Range('A195').Font.Color = $headerColor
$ws.Rows.Item(195).RowHeight = 48
$b5 = $ws.Range('A195:DA195').Borders.Item($xlEdgeBottom)
$b5.LineStyle = $xlContinuous
$b5.Weight = $xlThick

# --- shape parameters
Set-Text 'E198' 'seq';       Set-Num 'F198' 6
Set-Text 'E199' 'd_model';   Set-Num 'F199' 8
Set-Text 'E200' 'kv_lora_rank'; Set-Num 'F200' 3
Set-Text 'E201' 'n_heads';   Set-Num 'F201' 2
Set-Text 'E202' 'head_dim';  Set-Num 'F202' 3
Set-Text 'E203' 'rope_dim';  Set-Num 'F203' 2
Set-Right 'E198:E203'
Set-Center 'F198:F203'

# --- input, re-rolled and centred like every section
Set-Text 'N198' 'pos'
Set-Right 'N198'
Set-Formula 'O198' '=SEQUENCE(1,F198,1)'
Set-Center 'O198:T198'

Set-Formula 'O200' '="x"&O198#'
Set-Center 'O200:T200'
Set-Text 'N201' 'X'
Set-Right 'N201'
Set-Formula 'O201' '=(RANDARRAY(F199,F198)-0.5)*2'
Set-Box 'O201:T208'

# --- compress to the latent code
Set-Text 'D210' 'Compress'
Set-Bold 'D210'
Set-Text 'D211' 'W_DKV'
Set-Right 'D211'
Set-Formula 'E211' '=(RANDARRAY(F200,F199)-0.5)*2'
Set-Weights 'E211:L213'
Set-Text 'N211' 'c_KV'
Set-Right 'N211'
Set-Formula 'O211' '=MMULT(E211#,O201#)'
Set-Box 'O211:T213'

# --- the two orders
Set-Text 'D216' 'Order 1: rotate after projecting'
Set-Bold 'D216'
Set-Text 'D217' 'W_UK'
Set-Right 'D217'
Set-Formula 'E217' '=(RANDARRAY(F201*F202,F200)-0.5)*2'
Set-Weights 'E217:G222'
Set-Text 'N217' 'K'
Set-Right 'N217'
Set-Formula 'O217' '=MMULT(E217#,O211#)'
Set-Box 'O217:T222'

Set-Text 'D225' 'RoPE(K)'
Set-Bold 'D225'
Set-Formula 'O224' '="k"&O198#'
Set-Center 'O224:T224'
# theta values, one per token, as a visible helper row
Set-Text 'V224' 'theta = pos'
Set-Right 'V224'
Set-Formula 'W224' '=SEQUENCE(1,F198,1)'
Set-Center 'W224:AB224'
# The rotation is a 2x2 per head per token: [x0,x1] -> [x0*c - x1*s, x0*s + x1*c],
# dim 2 unchanged. Head 1 uses K rows 1-2, head 2 rows 4-5. Each head is one
# 2-row spill via a 2x1 SEQUENCE selector, so the head's own rows are used.
Set-Formula 'O225' '=IF(SEQUENCE(2,1)=1,COS(W224#)*O217:T217-SIN(W224#)*O218:T218,SIN(W224#)*O217:T217+COS(W224#)*O218:T218)'
Set-Center 'O225:T225'
Set-Formula 'O227' '=O219:T219'
Set-Center 'O227:T227'
Set-Formula 'O228' '=IF(SEQUENCE(2,1)=1,COS(W224#)*O220:T220-SIN(W224#)*O221:T221,SIN(W224#)*O220:T220+COS(W224#)*O221:T221)'
Set-Center 'O228:T228'
Set-Formula 'O230' '=O222:T222'
Set-Center 'O230:T230'
Set-Box 'O225:T230'

Set-Text 'D233' 'Order 2: rotate the code, then project'
Set-Bold 'D233'
Set-Text 'D234' 'W_UK (same)'
Set-Right 'D234'
Set-Text 'N234' 'RoPE(c_KV)'
Set-Right 'N234'
# c_KV has 3 rows; its rotary pair is rows 1-2, row 3 unchanged. One formula
# must spill the whole 3x6 block so O234# is 3x6 for the MMULT below; a 3x1
# SEQUENCE selector broadcasts against the three 1x6 row expressions.
Set-Formula 'O234' '=IF(SEQUENCE(F200,1)=1,COS(W224#)*O211:T211-SIN(W224#)*O212:T212,IF(SEQUENCE(F200,1)=2,SIN(W224#)*O211:T211+COS(W224#)*O212:T212,O213:T213))'
Set-Box 'O234:T236'
Set-Text 'N238' 'W_UK RoPE(c_KV)'
Set-Right 'N238'
Set-Formula 'O238' '=MMULT(E217#,O234#)'
Set-Box 'O238:T243'

# --- the two summary cells that should match and do not. If RoPE commuted
#     with W_UK, cell AA196 would equal cell AE196. They do not, and their
#     difference is the conflict section 6 resolves. The pair is spaced three
#     columns apart: each bold label needs about four columns and an adjacent
#     cell would clip it.
Set-Text 'AA195' 'RoPE(K)[h1,d0,t1]'
Set-Bold 'AA195'
Set-Text 'AE195' 'W_UK RoPE(c)[h1,d0,t1]'
Set-Bold 'AE195'
Set-Formula 'AA196' '=INDEX(O225#,1,1)'
Set-Box 'AA196'
Set-Formula 'AE196' '=INDEX(O238#,1,1)'
Set-Box 'AE196'

# ================================================================ SECTION 6
# Decoupled RoPE. Section 5 showed RoPE(W_UK c) != W_UK RoPE(c). The fix:
# keep RoPE on a key that comes from x, not from the latent code. q splits
# into q_c (content, from the latent) and q_r (rope, from the latent, then
# rotated); k splits into k_c (content, from the latent, never rotated) and
# k_r (one shared rope head fed from x, rotated). Score = q_c.k_c + q_r.k_r.
# Mirrors MultiHeadLatentAttention.reference_manual.
Set-Text 'A250' '# Decoupled RoPE'
$ws.Range('A250').Font.Size = 36
$ws.Range('A250').Font.Bold = $true
$ws.Range('A250').Font.Color = $headerColor
$ws.Rows.Item(250).RowHeight = 48
$b6 = $ws.Range('A250:EA250').Borders.Item($xlEdgeBottom)
$b6.LineStyle = $xlContinuous
$b6.Weight = $xlThick

# --- shape parameters
Set-Text 'E253' 'seq';          Set-Num 'F253' 6
Set-Text 'E254' 'd_model';      Set-Num 'F254' 8
Set-Text 'E255' 'q_lora_rank';  Set-Num 'F255' 4
Set-Text 'E256' 'kv_lora_rank'; Set-Num 'F256' 3
Set-Text 'E257' 'n_heads';      Set-Num 'F257' 2
Set-Text 'E258' 'qk_nope';      Set-Num 'F258' 3
Set-Text 'E259' 'qk_rope';      Set-Num 'F259' 2
Set-Right 'E253:E259'
Set-Center 'F253:F259'

# --- input, re-rolled and centred
Set-Text 'N253' 'pos'
Set-Right 'N253'
Set-Formula 'O253' '=SEQUENCE(1,F253,1)'
Set-Center 'O253:T253'
Set-Formula 'O255' '="x"&O253#'
Set-Center 'O255:T255'
Set-Text 'N256' 'X'
Set-Right 'N256'
Set-Formula 'O256' '=(RANDARRAY(F254,F253)-0.5)*2'
Set-Box 'O256:T263'

# --- query panel: split the head into content and rope. Per-head weights are
# separate RANDARRAYs (Excel 2021: MMULT/TRANSPOSE cannot slice a spill).
Set-Text 'D265' 'Split the query head'
Set-Bold 'D265'
Set-Text 'D266' 'W_DQ'
Set-Right 'D266'
Set-Formula 'E266' '=(RANDARRAY(F255,F254)-0.5)*2'
Set-Weights 'E266:L269'
Set-Text 'N266' 'c_Q'
Set-Right 'N266'
Set-Formula 'O266' '=MMULT(E266#,O256#)'
Set-Box 'O266:T269'

Set-Text 'D271' 'W_UQ1'
Set-Right 'D271'
Set-Formula 'E271' '=(RANDARRAY(F258,F255)-0.5)*2'
Set-Weights 'E271:H273'
Set-Text 'N271' 'q_c1'
Set-Right 'N271'
Set-Formula 'O271' '=MMULT(E271#,O266#)'
Set-Box 'O271:T273'

Set-Text 'D275' 'W_UQ2'
Set-Right 'D275'
Set-Formula 'E275' '=(RANDARRAY(F258,F255)-0.5)*2'
Set-Weights 'E275:H277'
Set-Text 'N275' 'q_c2'
Set-Right 'N275'
Set-Formula 'O275' '=MMULT(E275#,O266#)'
Set-Box 'O275:T277'

Set-Text 'D279' 'W_QR1'
Set-Right 'D279'
Set-Formula 'E279' '=(RANDARRAY(F259,F255)-0.5)*2'
Set-Weights 'E279:H280'
Set-Text 'N279' 'q_r1'
Set-Right 'N279'
Set-Formula 'O279' '=MMULT(E279#,O266#)'
Set-Box 'O279:T280'

Set-Text 'D282' 'W_QR2'
Set-Right 'D282'
Set-Formula 'E282' '=(RANDARRAY(F259,F255)-0.5)*2'
Set-Weights 'E282:H283'
Set-Text 'N282' 'q_r2'
Set-Right 'N282'
Set-Formula 'O282' '=MMULT(E282#,O266#)'
Set-Box 'O282:T283'

Set-Text 'V286' 'theta = pos'
Set-Right 'V286'
Set-Formula 'W286' '=SEQUENCE(1,F253,1)'
Set-Center 'W286:AB286'
Set-Text 'D286' 'RoPE(q_r)'
Set-Bold 'D286'
# q_r1 is O279:O280 (head 1), q_r2 is O282:O283 (head 2)
Set-Formula 'O286' '=IF(SEQUENCE(2,1)=1,COS(W286#)*O279:T279-SIN(W286#)*O280:T280,SIN(W286#)*O279:T279+COS(W286#)*O280:T280)'
Set-Formula 'O288' '=IF(SEQUENCE(2,1)=1,COS(W286#)*O282:T282-SIN(W286#)*O283:T283,SIN(W286#)*O282:T282+COS(W286#)*O283:T283)'
Set-Box 'O286:T289'

# --- key panel: content from the latent, rope from x
Set-Text 'V265' 'Split the key head'
Set-Bold 'V265'
Set-Text 'V266' 'W_DKV'
Set-Right 'V266'
Set-Formula 'W266' '=(RANDARRAY(F256,F254)-0.5)*2'
Set-Weights 'W266:AD268'
Set-Text 'AF266' 'c_KV'
Set-Right 'AF266'
Set-Formula 'AG266' '=MMULT(W266#,O256#)'
Set-Box 'AG266:AL268'

Set-Text 'V271' 'W_UK1'
Set-Right 'V271'
Set-Formula 'W271' '=(RANDARRAY(F258,F256)-0.5)*2'
Set-Weights 'W271:Y273'
Set-Text 'AF271' 'k_c1'
Set-Right 'AF271'
Set-Formula 'AG271' '=MMULT(W271#,AG266#)'
Set-Box 'AG271:AL273'

Set-Text 'V275' 'W_UK2'
Set-Right 'V275'
Set-Formula 'W275' '=(RANDARRAY(F258,F256)-0.5)*2'
Set-Weights 'W275:Y277'
Set-Text 'AF275' 'k_c2'
Set-Right 'AF275'
Set-Formula 'AG275' '=MMULT(W275#,AG266#)'
Set-Box 'AG275:AL277'

Set-Text 'V279' 'W_KR'
Set-Right 'V279'
Set-Formula 'W279' '=(RANDARRAY(F259,F254)-0.5)*2'
Set-Weights 'W279:AD280'
Set-Text 'AF279' 'k_r'
Set-Right 'AF279'
Set-Formula 'AG279' '=MMULT(W279#,O256#)'
Set-Box 'AG279:AL280'

# This heading used to be written to V286, which already held the shared
# "theta = pos" label for the rotation angles. The second write silently won,
# so the theta row lost its label and this block was labelled in the wrong
# column. AF is where the rest of the key panel's labels live (AF279 'k_r').
Set-Text 'AF286' 'RoPE(k_r)'
Set-Bold 'AF286'
Set-Right 'AF286'
Set-Formula 'AG286' '=IF(SEQUENCE(2,1)=1,COS(W286#)*AG279:AL279-SIN(W286#)*AG280:AL280,SIN(W286#)*AG279:AL279+COS(W286#)*AG280:AL280)'
Set-Box 'AG286:AL287'

# --- score, head 1: S = K_c1^T Q_c1 + K_r^T Q_r1. All operands raw spills;
# transposes are standalone cells.
Set-Text 'D291' 'Score, head 1'
Set-Bold 'D291'
Set-Text 'D292' 'k_c1^T'
Set-Right 'D292'
Set-Formula 'E292' '=TRANSPOSE(AG271#)'
Set-Box 'E292:G297'
Set-Text 'D299' 'q_r1^T'
Set-Right 'D299'
Set-Formula 'E299' '=TRANSPOSE(O279#)'
Set-Box 'E299:G304'
Set-Text 'D306' 'k_r^T'
Set-Right 'D306'
Set-Formula 'E306' '=TRANSPOSE(AG279#)'
Set-Box 'E306:G311'
Set-Text 'D313' 'S_c = K_c1^T Q_c1'
Set-Right 'D313'
Set-Formula 'O313' '=MMULT(E292#,O271#)'
Set-Box 'O313:T318'
Set-Text 'D320' 'S_r = K_r^T Q_r1'
Set-Right 'D320'
Set-Formula 'O320' '=MMULT(E306#,O279#)'
Set-Box 'O320:T325'
Set-Text 'D327' 'S = S_c + S_r'
Set-Right 'D327'
Set-Formula 'O327' '=O313#+O320#'
Set-Box 'O327:T332'
Set-Text 'D334' 'RoPE now hits k_r, a key straight from x, not the latent code.'
Set-Text 'D335' 'Section 5 conflict gone: content never rotates, rope never projects.'

# ================================================================ SECTION 7
# Weight Absorption and the Cache Ledger. Two ways to compute the same head-1
# output. Naive (mirrors decode_naive): rebuild k_c = W_UK c_KV and v = W_UV
# c_KV per head at read time. Absorbed (mirrors decode): fold W_UK into the
# query, score directly against c_KV, fold W_UV out after attention. S_A = S_B
# and O_A = O_B. The ledger shows why: MHA 16 floats per token, GQA 8, MLA 5.
Set-Text 'A340' '# Weight Absorption and the Cache Ledger'
$ws.Range('A340').Font.Size = 36
$ws.Range('A340').Font.Bold = $true
$ws.Range('A340').Font.Color = $headerColor
$ws.Rows.Item(340).RowHeight = 48
$b7 = $ws.Range('A340:EA340').Borders.Item($xlEdgeBottom)
$b7.LineStyle = $xlContinuous
$b7.Weight = $xlThick

# --- shape parameters
Set-Text 'E343' 'seq';          Set-Num 'F343' 6
Set-Text 'E344' 'd_model';      Set-Num 'F344' 8
Set-Text 'E345' 'q_lora_rank';  Set-Num 'F345' 4
Set-Text 'E346' 'kv_lora_rank'; Set-Num 'F346' 3
Set-Text 'E347' 'n_heads';      Set-Num 'F347' 2
Set-Text 'E348' 'qk_nope';      Set-Num 'F348' 3
Set-Text 'E349' 'qk_rope';      Set-Num 'F349' 2
Set-Text 'E350' 'v_head';       Set-Num 'F350' 3
Set-Right 'E343:E350'
Set-Center 'F343:F350'

# --- input, re-rolled and centred
Set-Text 'N343' 'pos'
Set-Right 'N343'
Set-Formula 'O343' '=SEQUENCE(1,F343,1)'
Set-Center 'O343:T343'
Set-Formula 'O345' '="x"&O343#'
Set-Center 'O345:T345'
Set-Text 'N346' 'X'
Set-Right 'N346'
Set-Formula 'O346' '=(RANDARRAY(F344,F343)-0.5)*2'
Set-Box 'O346:T353'

# --- shared preparation: query parts and latent code. Per-head query weights
# are separate RANDARRAYs (Excel 2021: MMULT/TRANSPOSE cannot slice a spill).
Set-Text 'D355' 'Shared preparation'
Set-Bold 'D355'
Set-Text 'D356' 'W_DQ'
Set-Right 'D356'
Set-Formula 'E356' '=(RANDARRAY(F345,F344)-0.5)*2'
Set-Weights 'E356:L359'
Set-Text 'N356' 'c_Q'
Set-Right 'N356'
Set-Formula 'O356' '=MMULT(E356#,O346#)'
Set-Box 'O356:T359'

Set-Text 'D361' 'W_UQ1'
Set-Right 'D361'
Set-Formula 'E361' '=(RANDARRAY(F348,F345)-0.5)*2'
Set-Weights 'E361:H363'
Set-Text 'N361' 'q_c1'
Set-Right 'N361'
Set-Formula 'O361' '=MMULT(E361#,O356#)'
Set-Box 'O361:T363'

# Head 2 removed. W_UQ2/q_c2, W_QR2/q_r2, W_UK2/k_c2 and W_UV2/v2 were each
# computed and boxed but never read by S_A, S_B, O_A or O_B: the section only
# ever scores head 1. Eight blocks of scaffolding that a reader has to inspect
# before discovering they lead nowhere. One head, shown fully, plus a line of
# text saying head 2 is identical.
Set-Text 'D369' 'W_QR1'
Set-Right 'D369'
Set-Formula 'E369' '=(RANDARRAY(F349,F345)-0.5)*2'
Set-Weights 'E369:H370'
Set-Text 'N369' 'q_r1'
Set-Right 'N369'
Set-Formula 'O369' '=MMULT(E369#,O356#)'
Set-Box 'O369:T370'

Set-Text 'V361' 'W_DKV'
Set-Right 'V361'
Set-Formula 'W361' '=(RANDARRAY(F346,F344)-0.5)*2'
Set-Weights 'W361:AD363'
Set-Text 'AF361' 'c_KV'
Set-Right 'AF361'
Set-Formula 'AG361' '=MMULT(W361#,O346#)'
Set-Box 'AG361:AL363'

Set-Text 'V368' 'W_KR'
Set-Right 'V368'
Set-Formula 'W368' '=(RANDARRAY(F349,F344)-0.5)*2'
Set-Weights 'W368:X369'
Set-Text 'AF368' 'k_r'
Set-Right 'AF368'
Set-Formula 'AG368' '=MMULT(W368#,O346#)'
Set-Box 'AG368:AL369'

Set-Text 'V370' 'theta = pos'
Set-Right 'V370'
Set-Formula 'W370' '=SEQUENCE(1,F343,1)'
Set-Center 'W370:AB370'
Set-Text 'V373' 'RoPE(k_r)'
Set-Bold 'V373'
Set-Formula 'AG373' '=IF(SEQUENCE(2,1)=1,COS(W370#)*AG368:AL368-SIN(W370#)*AG369:AL369,SIN(W370#)*AG368:AL368+COS(W370#)*AG369:AL369)'
Set-Box 'AG373:AL374'
Set-Text 'D376' 'RoPE(q_r)'
Set-Bold 'D376'
Set-Formula 'O376' '=IF(SEQUENCE(2,1)=1,COS(W370#)*O369:T369-SIN(W370#)*O370:T370,SIN(W370#)*O369:T369+COS(W370#)*O370:T370)'
Set-Box 'O376:T377'

# --- path A (naive): rebuild K and V per head. Per-head weights are separate
# RANDARRAYs. Excel 2021 quirk (probed 2026-08-06): MMULT and TRANSPOSE cannot
# consume a slice (plain range or INDEX) of any spill, so every matrix below is
# a raw RANDARRAY or a MMULT of raw spills, and every transpose is a standalone
# cell of a raw spill, exactly like section 1's D = K^T Q.
Set-Text 'D380' 'Naive: rebuild K and V per head'
Set-Bold 'D380'
Set-Text 'D381' 'W_UK1'
Set-Right 'D381'
Set-Formula 'E381' '=(RANDARRAY(F348,F346)-0.5)*2'
Set-Weights 'E381:G383'
Set-Text 'N381' 'k_c1'
Set-Right 'N381'
Set-Formula 'O381' '=MMULT(E381#,AG361#)'
Set-Box 'O381:T383'

Set-Text 'D385' 'W_UV1'
Set-Right 'D385'
Set-Formula 'E385' '=(RANDARRAY(F350,F346)-0.5)*2'
Set-Weights 'E385:G387'
Set-Text 'N385' 'v1'
Set-Right 'N385'
Set-Formula 'O385' '=MMULT(E385#,AG361#)'
Set-Box 'O385:T387'

# Transposes of raw spills (TRANSPOSE cannot consume a slice of a spill here).
# q_r1^T was computed and boxed but referenced by nothing, so it is gone.
# k_r^T now transposes RoPE(k_r), not the raw k_r: the rope half of the score
# has to use the rotated vectors, which is the whole point of section 6. The
# section previously computed RoPE(q_r) and RoPE(k_r), displayed them, and then
# scored with the unrotated values, so those two blocks were decorative and the
# score was wrong on that term. Both paths use the same rope term, so S_A = S_B
# held either way and no check caught it.
Set-Text 'D390' 'k_c1^T'
Set-Right 'D390'
Set-Formula 'E390' '=TRANSPOSE(O381#)'
Set-Box 'E390:G395'
Set-Text 'D397' 'q_c1^T'
Set-Right 'D397'
Set-Formula 'E397' '=TRANSPOSE(O361#)'
Set-Box 'E397:G402'
Set-Text 'D404' 'RoPE(k_r)^T'
Set-Right 'D404'
Set-Formula 'E404' '=TRANSPOSE(AG373#)'
Set-Box 'E404:G409'

# Both paths converge here. S_A, A_A and O_A share their rows with S_B, A_B and
# O_B in the column band to the right, so a horizontal line at any row is a
# valid ruler between the two paths. That is what makes them comparable.
# Headings go above their blocks. These equations are too long to sit
# right-aligned in the label column: they overflow left into the neighbouring
# path's block and get clipped.
Set-Text 'O411' 'S_A = K_c1^T Q_c1 + RoPE(K_r)^T RoPE(Q_r)'
Set-Formula 'O412' '=MMULT(E390#,O361#)+MMULT(E404#,O376#)'
Set-Box 'O412:T417'

Set-Text 'O418' 'A_A = softmax(S_A)'
Set-Bold 'O418'
$sAcols = @(
  @('O419','O412:O417'), @('P419','P412:P417'), @('Q419','Q412:Q417'),
  @('R419','R412:R417'), @('S419','S412:S417'), @('T419','T412:T417')
)
foreach ($p in $sAcols) {
  Set-Formula $p[0] ('=LET(z,' + $p[1] + ',EXP(z)/SUM(EXP(z)))')
}
Set-Box 'O419:T424'

Set-Text 'O425' 'O_A = V1 A_A'
Set-Formula 'O426' '=MMULT(O385#,O419:T424)'
Set-Box 'O426:T428'

# --- path B (absorbed): fold W_UK into Q, W_UV out
Set-Text 'V380' 'Absorbed: fold W_UK into Q, W_UV out'
Set-Bold 'V380'
Set-Text 'V381' 'W_UK1'
Set-Right 'V381'
Set-Formula 'W381' '=E381#'
Set-Box 'W381:Y383'
Set-Text 'W384' "q'_c = q_c1^T W_UK1"
# q_c1 is 3x6 (head 1), W_UK1 is 3x3. q'_c = q_c1^T W_UK1 is 6x3:
# tokens x latent, the query that scores directly against c_KV.
Set-Formula 'W385' '=MMULT(E397#,W381#)'
Set-Box 'W385:Y390'

Set-Text 'V392' 'c_KV^T'
Set-Right 'V392'
Set-Formula 'W392' '=TRANSPOSE(AG361#)'
Set-Box 'W392:Y397'
Set-Text 'V399' "q'_c^T"
Set-Right 'V399'
Set-Formula 'W399' '=TRANSPOSE(W385#)'
Set-Box 'W399:Y401'

Set-Text 'V403' 'W_UV1'
Set-Right 'V403'
Set-Formula 'W403' '=E385#'
Set-Box 'W403:Y405'

Set-Text 'W411' "S_B = c_KV^T q'_c + RoPE(K_r)^T RoPE(Q_r)"
# c_KV^T is 6x3, q'_c^T is 3x6; 6x3 * 3x6 = 6x6.
Set-Formula 'W412' '=MMULT(W392#,W399#)+MMULT(E404#,O376#)'
Set-Box 'W412:AB417'

Set-Text 'W418' 'A_B = softmax(S_B)'
Set-Bold 'W418'
$sBcols = @(
  @('W419','W412:W417'), @('X419','X412:X417'), @('Y419','Y412:Y417'),
  @('Z419','Z412:Z417'), @('AA419','AA412:AA417'), @('AB419','AB412:AB417')
)
foreach ($p in $sBcols) {
  Set-Formula $p[0] ('=LET(z,' + $p[1] + ',EXP(z)/SUM(EXP(z)))')
}
Set-Box 'W419:AB424'

Set-Text 'AD418' 'c_KV A_B (weighted latent)'
Set-Formula 'AD419' '=MMULT(AG361#,W419:AB424)'
Set-Box 'AD419:AI421'

Set-Text 'W425' 'O_B = W_UV1 (c_KV A_B)'
Set-Formula 'W426' '=MMULT(W403#,AD419#)'
Set-Box 'W426:AB428'

# Conclusion sits below the two outputs it describes, not above them.
Set-Text 'D431' 'S_A = S_B and O_A = O_B: identical output, but path B never materialises a per-head K or V.'
Set-Text 'D432' 'It stays in the shared latent code c_KV the whole way. Head 2 is identical and is not shown.'

# --- the punchline, in its own block rather than squeezed into the D-F gutter
Set-Text 'D435' 'Cache per token: what each scheme must store'
Set-Bold 'D435'
Set-Text 'N436' 'MHA'
Set-Right 'N436'
Set-Formula 'O436' '=F347*(F348+F350+F349)'
Set-Text 'N437' 'GQA (1 group)'
Set-Right 'N437'
Set-Formula 'O437' '=F348+F350+F349'
Set-Text 'N438' 'MLA (c_KV + k_r)'
Set-Right 'N438'
Set-Formula 'O438' '=F346+F349'
Set-Box 'O436:O438'
Set-Text 'N440' 'MLA is smaller than MHA by'
Set-Right 'N440'
Set-Formula 'O440' '=O436/O438'
Set-Box 'O440'
Set-Text 'Q440' 'times'

# ================================================================ SECTION 8
# SwiGLU and Top-k MoE. Two new mechanisms, one screen. First the dense expert
# from SwiGLUExpert in src/compact_v3/experts.py: out = W_o (silu(W_g x) * W_v x).
# Then the router from TopKRouter: logits = W_r x, softmax down each token
# column, and each token is sent to its argmax expert. Top-k 1 here; section 9
# raises it.
Set-Text 'A455' '# SwiGLU and Top-k MoE'
$ws.Range('A455').Font.Size = 36
$ws.Range('A455').Font.Bold = $true
$ws.Range('A455').Font.Color = $headerColor
$ws.Rows.Item(455).RowHeight = 48
$b8 = $ws.Range('A455:EA455').Borders.Item($xlEdgeBottom)
$b8.LineStyle = $xlContinuous
$b8.Weight = $xlThick

# --- shape parameters
Set-Text 'E458' 'seq';           Set-Num 'F458' 6
Set-Text 'E459' 'd_model';       Set-Num 'F459' 8
Set-Text 'E460' 'expert_hidden'; Set-Num 'F460' 6
Set-Text 'E461' 'n_experts';     Set-Num 'F461' 8
Set-Text 'E462' 'top_k';         Set-Num 'F462' 1
Set-Right 'E458:E462'
Set-Center 'F458:F462'

# --- input
Set-Text 'N458' 'pos'
Set-Right 'N458'
Set-Formula 'O458' '=SEQUENCE(1,F458,1)'
Set-Center 'O458:T458'
Set-Formula 'O460' '="x"&O458#'
Set-Center 'O460:T460'
Set-Text 'N461' 'X'
Set-Right 'N461'
Set-Formula 'O461' '=(RANDARRAY(F459,F458)-0.5)*2'
Set-Box 'O461:T468'

# --- the dense SwiGLU expert
Set-Text 'D470' 'SwiGLU expert'
Set-Bold 'D470'
Set-Text 'D471' 'W_gate'
Set-Right 'D471'
Set-Formula 'E471' '=(RANDARRAY(F460,F459)-0.5)*2'
Set-Weights 'E471:L476'
Set-Text 'O470' 'g = silu(W_g X)'
Set-Formula 'O471' '=MMULT(E471#,O461#)/(1+EXP(-MMULT(E471#,O461#)))'
Set-Box 'O471:T476'

Set-Text 'D478' 'W_value'
Set-Right 'D478'
Set-Formula 'E478' '=(RANDARRAY(F460,F459)-0.5)*2'
Set-Weights 'E478:L483'
Set-Text 'N478' 'v = W_v X'
Set-Right 'N478'
Set-Formula 'O478' '=MMULT(E478#,O461#)'
Set-Box 'O478:T483'

Set-Text 'D485' 'h = g * v'
Set-Right 'D485'
Set-Formula 'O485' '=O471#*O478#'
Set-Box 'O485:T490'

Set-Text 'D492' 'W_out'
Set-Right 'D492'
Set-Formula 'E492' '=(RANDARRAY(F459,F460)-0.5)*2'
Set-Weights 'E492:J499'
Set-Text 'N492' 'out = W_o h'
Set-Right 'N492'
Set-Formula 'O492' '=MMULT(E492#,O485#)'
Set-Box 'O492:T499'

# --- the router
Set-Text 'V470' 'Router'
Set-Bold 'V470'
Set-Text 'V471' 'W_router'
Set-Right 'V471'
Set-Formula 'W471' '=(RANDARRAY(F461,F459)-0.5)*2'
Set-Weights 'W471:AD478'
Set-Text 'AG470' 'logits = W_r X'
Set-Formula 'AG471' '=MMULT(W471#,O461#)'
Set-Box 'AG471:AL478'

Set-Text 'W479' 'A = softmax(logits) by token'
Set-Bold 'W479'
# 8 experts x 6 tokens, softmax down each token column
$r8cols = @(
  @('W480','AG471:AG478'), @('X480','AH471:AH478'), @('Y480','AI471:AI478'),
  @('Z480','AJ471:AJ478'), @('AA480','AK471:AK478'), @('AB480','AL471:AL478')
)
foreach ($p in $r8cols) {
  Set-Formula $p[0] ('=LET(z,' + $p[1] + ',EXP(z)/SUM(EXP(z)))')
}
Set-Box 'W480:AB487'

Set-Text 'W488' 'argmax expert per token'
Set-Bold 'W488'
# top_k=1: LARGE gives the max, MATCH its expert index (1..8)
$argmax = @(
  @('W489','W480:W487'), @('X489','X480:X487'), @('Y489','Y480:Y487'),
  @('Z489','Z480:Z487'), @('AA489','AA480:AA487'), @('AB489','AB480:AB487')
)
foreach ($p in $argmax) {
  Set-Formula $p[0] ('=MATCH(LARGE(' + $p[1] + ',1),' + $p[1] + ',0)')
}
Set-Center 'W489:AB489'

Set-Text 'D501' 'Each token goes to its argmax expert; that expert runs the SwiGLU above.'
Set-Text 'D502' 'top_k = 1 here. Section 9 sends each token to 2 experts and adds a shared one.'

# ================================================================ SECTION 9
# DeepSeekMoE: Fine-Grained and Shared Experts. The delta on section 8: experts
# are quartered (expert_hidden 6 -> 3), top_k goes to 2, and one always-on
# shared expert joins the routed ones. Mirrors DeepSeekMoE in src/compact_v3/moe.py.
Set-Text 'A510' '# DeepSeekMoE: Fine-Grained and Shared Experts'
$ws.Range('A510').Font.Size = 36
$ws.Range('A510').Font.Bold = $true
$ws.Range('A510').Font.Color = $headerColor
$ws.Rows.Item(510).RowHeight = 48
$b9 = $ws.Range('A510:EA510').Borders.Item($xlEdgeBottom)
$b9.LineStyle = $xlContinuous
$b9.Weight = $xlThick

Set-Text 'E513' 'seq';           Set-Num 'F513' 6
Set-Text 'E514' 'd_model';       Set-Num 'F514' 8
Set-Text 'E515' 'expert_hidden'; Set-Num 'F515' 3
Set-Text 'E516' 'n_experts';     Set-Num 'F516' 8
Set-Text 'E517' 'top_k';         Set-Num 'F517' 2
Set-Text 'E518' 'n_shared';      Set-Num 'F518' 1
Set-Right 'E513:E518'
Set-Center 'F513:F518'

Set-Text 'N513' 'pos'
Set-Right 'N513'
Set-Formula 'O513' '=SEQUENCE(1,F513,1)'
Set-Center 'O513:T513'
Set-Formula 'O515' '="x"&O513#'
Set-Center 'O515:T515'
Set-Text 'N516' 'X'
Set-Right 'N516'
Set-Formula 'O516' '=(RANDARRAY(F514,F513)-0.5)*2'
Set-Box 'O516:T523'

# --- router: logits and softmax (8 experts x 6 tokens)
Set-Text 'D525' 'Router'
Set-Bold 'D525'
Set-Text 'D526' 'W_router'
Set-Right 'D526'
Set-Formula 'E526' '=(RANDARRAY(F516,F514)-0.5)*2'
Set-Weights 'E526:L533'
Set-Text 'N526' 'logits'
Set-Right 'N526'
Set-Formula 'O526' '=MMULT(E526#,O516#)'
Set-Box 'O526:T533'
Set-Text 'N535' 'A = softmax'
Set-Right 'N535'
# One formula spills the whole 8x6 softmax. Per-column LET spills cannot be
# consumed by MATCH/INDEX in this Excel (probed 2026-08-06); a single
# IF(SEQUENCE) broadcast spill can. Top-2 selection uses the logits (raw
# MMULT spill), which MATCH/LARGE can read.
Set-Formula 'O535' '=IF(SEQUENCE(1,6)=1,LET(z,O526:O533,EXP(z)/SUM(EXP(z))),IF(SEQUENCE(1,6)=2,LET(z,P526:P533,EXP(z)/SUM(EXP(z))),IF(SEQUENCE(1,6)=3,LET(z,Q526:Q533,EXP(z)/SUM(EXP(z))),IF(SEQUENCE(1,6)=4,LET(z,R526:R533,EXP(z)/SUM(EXP(z))),IF(SEQUENCE(1,6)=5,LET(z,S526:S533,EXP(z)/SUM(EXP(z))),LET(z,T526:T533,EXP(z)/SUM(EXP(z))))))))'
Set-Box 'O535:T542'

# --- top-2 selection per token on the LOGITS (monotone in softmax)
Set-Text 'D544' 'top-2 experts per token'
Set-Bold 'D544'
Set-Text 'N544' '1st'
Set-Right 'N544'
$t9a = @(
  @('O544','O526:O533'), @('P544','P526:P533'), @('Q544','Q526:Q533'),
  @('R544','R526:R533'), @('S544','S526:S533'), @('T544','T526:T533')
)
foreach ($p in $t9a) {
  Set-Formula $p[0] ('=MATCH(LARGE(' + $p[1] + ',1),' + $p[1] + ',0)')
}
Set-Center 'O544:T544'
Set-Text 'N545' '2nd'
Set-Right 'N545'
$t9b = @(
  @('O545','O526:O533'), @('P545','P526:P533'), @('Q545','Q526:Q533'),
  @('R545','R526:R533'), @('S545','S526:S533'), @('T545','T526:T533')
)
foreach ($p in $t9b) {
  Set-Formula $p[0] ('=MATCH(LARGE(' + $p[1] + ',2),' + $p[1] + ',0)')
}
Set-Center 'O545:T545'

# --- renormalised weights over the selected two. Computed from the LOGITS via
# per-column slice INDEX (plain ranges, reliable in this Excel). w1 uses the
# 1st and 2nd selectors explicitly - they are NOT adjacent rows.
Set-Text 'N547' 'w1'
Set-Right 'N547'
$w9a = @(
  @('O547','O526:O533','O544','O545'), @('P547','P526:P533','P544','P545'), @('Q547','Q526:Q533','Q544','Q545'),
  @('R547','R526:R533','R544','R545'), @('S547','S526:S533','S544','S545'), @('T547','T526:T533','T544','T545')
)
foreach ($p in $w9a) {
  Set-Formula $p[0] ('=EXP(INDEX(' + $p[1] + ',' + $p[2] + '))/(EXP(INDEX(' + $p[1] + ',' + $p[2] + '))+EXP(INDEX(' + $p[1] + ',' + $p[3] + ')))')
}
Set-Text 'N548' 'w2'
Set-Right 'N548'
$w9b = @(
  @('O548','O526:O533','O545','O544'), @('P548','P526:P533','P545','P544'), @('Q548','Q526:Q533','Q545','Q544'),
  @('R548','R526:R533','R545','R544'), @('S548','S526:S533','S545','S544'), @('T548','T526:T533','T545','T544')
)
foreach ($p in $w9b) {
  Set-Formula $p[0] ('=EXP(INDEX(' + $p[1] + ',' + $p[2] + '))/(EXP(INDEX(' + $p[1] + ',' + $p[2] + '))+EXP(INDEX(' + $p[1] + ',' + $p[3] + ')))')
}
Set-Center 'O547:T548'

# --- the shared expert: always on, one SwiGLU over all tokens
Set-Text 'D550' 'Shared expert (always on)'
Set-Bold 'D550'
Set-Text 'D551' 'W_sg'
Set-Right 'D551'
Set-Formula 'E551' '=(RANDARRAY(F515,F514)-0.5)*2'
Set-Weights 'E551:L553'
Set-Text 'D555' 'W_sv'
Set-Right 'D555'
Set-Formula 'E555' '=(RANDARRAY(F515,F514)-0.5)*2'
Set-Weights 'E555:L557'
Set-Text 'D559' 'W_so'
Set-Right 'D559'
Set-Formula 'E559' '=(RANDARRAY(F514,F515)-0.5)*2'
Set-Weights 'E559:G566'
Set-Text 'N559' 'shared out'
Set-Right 'N559'
Set-Formula 'O559' '=MMULT(E559#,(MMULT(E551#,O516#)/(1+EXP(-MMULT(E551#,O516#))))*MMULT(E555#,O516#))'
Set-Box 'O559:T566'

Set-Text 'D568' 'A routed expert would compute out = W_o (silu(W_g x)*W_v x) the same way.'
Set-Text 'D569' 'top-2 weights are renormalised over the selected pair, then each routed expert'
Set-Text 'D570' 'contributes weight * its SwiGLU output; the shared expert adds its full output.'

# ================================================================ SECTION 10
# Sigmoid Affinity and Route Scale. V3's change from V2. The router's logits
# become per-expert sigmoid affinities (no softmax competition), top-2 picked,
# renormalised over the selected pair only, then scaled by route_scale 0.75.
# Mirrors TopKRouter in src/compact_v3/routing.py.
Set-Text 'A580' '# Sigmoid Affinity and Route Scale'
$ws.Range('A580').Font.Size = 36
$ws.Range('A580').Font.Bold = $true
$ws.Range('A580').Font.Color = $headerColor
$ws.Rows.Item(580).RowHeight = 48
$b10 = $ws.Range('A580:EA580').Borders.Item($xlEdgeBottom)
$b10.LineStyle = $xlContinuous
$b10.Weight = $xlThick

Set-Text 'E583' 'seq';           Set-Num 'F583' 6
Set-Text 'E584' 'd_model';       Set-Num 'F584' 8
Set-Text 'E585' 'n_experts';     Set-Num 'F585' 8
Set-Text 'E586' 'top_k';         Set-Num 'F586' 2
Set-Text 'E587' 'route_scale';   Set-Num 'F587' 0.75
Set-Right 'E583:E587'
Set-Center 'F583:F587'

Set-Text 'N583' 'pos'
Set-Right 'N583'
Set-Formula 'O583' '=SEQUENCE(1,F583,1)'
Set-Center 'O583:T583'
Set-Formula 'O585' '="x"&O583#'
Set-Center 'O585:T585'
Set-Text 'N586' 'X'
Set-Right 'N586'
Set-Formula 'O586' '=(RANDARRAY(F584,F583)-0.5)*2'
Set-Box 'O586:T593'

Set-Text 'D595' 'Router'
Set-Bold 'D595'
Set-Text 'D596' 'W_router'
Set-Right 'D596'
Set-Formula 'E596' '=(RANDARRAY(F585,F584)-0.5)*2'
Set-Weights 'E596:L603'
Set-Text 'N596' 'logits'
Set-Right 'N596'
Set-Formula 'O596' '=MMULT(E596#,O586#)'
Set-Box 'O596:T603'
Set-Text 'N605' 'affinity = sigmoid(logits)'
Set-Right 'N605'
Set-Formula 'O605' '=1/(1+EXP(-O596#))'
Set-Box 'O605:T612'

# --- top-2 by affinity, renormalised, scaled
Set-Text 'D614' 'top-2 by affinity'
Set-Bold 'D614'
Set-Text 'N614' '1st'
Set-Right 'N614'
$a10a = @(
  @('O614','O596:O603'), @('P614','P596:P603'), @('Q614','Q596:Q603'),
  @('R614','R596:R603'), @('S614','S596:S603'), @('T614','T596:T603')
)
foreach ($p in $a10a) {
  Set-Formula $p[0] ('=MATCH(LARGE(' + $p[1] + ',1),' + $p[1] + ',0)')
}
Set-Center 'O614:T614'
Set-Text 'N615' '2nd'
Set-Right 'N615'
$a10b = @(
  @('O615','O596:O603'), @('P615','P596:P603'), @('Q615','Q596:Q603'),
  @('R615','R596:R603'), @('S615','S596:S603'), @('T615','T596:T603')
)
foreach ($p in $a10b) {
  Set-Formula $p[0] ('=MATCH(LARGE(' + $p[1] + ',2),' + $p[1] + ',0)')
}
Set-Center 'O615:T615'

Set-Text 'N617' 'w1'
Set-Right 'N617'
# From the affinity (sigmoid) via per-column slice INDEX. Unlike softmax, the
# sigmoid ratio is not the exp(logit) ratio, so the affinity values are used.
$w10a = @(
  @('O617','O605:O612','O614','O615'), @('P617','P605:P612','P614','P615'), @('Q617','Q605:Q612','Q614','Q615'),
  @('R617','R605:R612','R614','R615'), @('S617','S605:S612','S614','S615'), @('T617','T605:T612','T614','T615')
)
foreach ($p in $w10a) {
  Set-Formula $p[0] ('=INDEX(' + $p[1] + ',' + $p[2] + ')/(INDEX(' + $p[1] + ',' + $p[2] + ')+INDEX(' + $p[1] + ',' + $p[3] + '))')
}
Set-Text 'N618' 'w2'
Set-Right 'N618'
$w10b = @(
  @('O618','O605:O612','O615','O614'), @('P618','P605:P612','P615','P614'), @('Q618','Q605:Q612','Q615','Q614'),
  @('R618','R605:R612','R615','R614'), @('S618','S605:S612','S615','S614'), @('T618','T605:T612','T615','T614')
)
foreach ($p in $w10b) {
  Set-Formula $p[0] ('=INDEX(' + $p[1] + ',' + $p[2] + ')/(INDEX(' + $p[1] + ',' + $p[2] + ')+INDEX(' + $p[1] + ',' + $p[3] + '))')
}
Set-Center 'O617:T618'

Set-Text 'D620' 'route scale'
Set-Right 'D620'
Set-Formula 'F620' '=F587'
Set-Center 'F620'
Set-Text 'N620' 'w1 * route_scale'
Set-Right 'N620'
Set-Text 'N621' 'w2 * route_scale'
Set-Right 'N621'
# w1/w2 are per-token single cells (INDEX returns scalars), so scale per cell
$scaled10 = @(
  @('O620','O617','O618'), @('P620','P617','P618'), @('Q620','Q617','Q618'),
  @('R620','R617','R618'), @('S620','S617','S618'), @('T620','T617','T618')
)
foreach ($p in $scaled10) {
  Set-Formula $p[0] ('=' + $p[1] + '*F587')
  Set-Formula ([string]($p[0][0]) + '621') ('=' + $p[2] + '*F587')
}
Set-Center 'O620:T621'
Set-Box 'O620:T621'

Set-Text 'D623' 'Sigmoid lets several experts be good at once; softmax forces a zero-sum.'
Set-Text 'D624' 'Renormalise over the selected pair only, then scale by route_scale.'

# ================================================================ SECTION 11
# Load Collapse. A failure, no new mechanism: route 24 tokens through 8
# experts (top-2) and watch the load concentrate. A biased router (affinity
# plus a fixed per-expert bias) makes experts 2, 5, 8 dominate. Normalised
# entropy of the load distribution measures how collapsed it is: 1.0 is
# uniform, 0.0 is everything on one expert.
Set-Text 'A632' '# Load Collapse'
$ws.Range('A632').Font.Size = 36
$ws.Range('A632').Font.Bold = $true
$ws.Range('A632').Font.Color = $headerColor
$ws.Rows.Item(632).RowHeight = 48
$b11 = $ws.Range('A632:EA632').Borders.Item($xlEdgeBottom)
$b11.LineStyle = $xlContinuous
$b11.Weight = $xlThick

Set-Text 'E635' 'n_tokens'; Set-Num 'F635' 24
Set-Text 'E636' 'd_model';   Set-Num 'F636' 8
Set-Text 'E637' 'n_experts'; Set-Num 'F637' 8
Set-Text 'E638' 'top_k';     Set-Num 'F638' 2
# The forced advantage that makes three experts win everything. It was hard
# coded at 1.5, which the random logits could overcome: the section's own
# "collapse is visible" check failed on roughly 9 draws in 100. At 3.0 the
# collapse is decisive on every draw. Section 12 reads the same number so the
# two sections tell one continuous story.
Set-Text 'E639' 'advantage'; Set-Num 'F639' 3
Set-Right 'E635:E639'
Set-Center 'F635:F639'

# 24 tokens, 8 features each
Set-Text 'N635' 'pos'
Set-Right 'N635'
Set-Formula 'O635' '=SEQUENCE(1,F635,1)'
Set-Center 'O635:T635'
Set-Text 'N636' 'X (24 tokens)'
Set-Right 'N636'
Set-Formula 'O636' '=(RANDARRAY(F636,F635)-0.5)*2'
Set-Box 'O636:AL643'

# router logits and affinity
Set-Text 'D661' 'Router'
Set-Bold 'D661'
Set-Text 'D662' 'W_router'
Set-Right 'D662'
Set-Formula 'E662' '=(RANDARRAY(F637,F636)-0.5)*2'
Set-Weights 'E662:L669'
Set-Text 'N662' 'logits'
Set-Right 'N662'
Set-Formula 'O662' '=MMULT(E662#,O636#)'
Set-Box 'O662:AL669'

# bias makes experts 2, 5, 8 dominate: +1.5 to those rows. The 8x1 row mask
# uses arithmetic (not OR, which collapses to a scalar) and is broadcast to
# 8x24 via the inner SEQUENCE.
Set-Text 'N683' 'bias'
Set-Right 'N683'
Set-Formula 'O683' '=IF((SEQUENCE(F637,1)=2)+(SEQUENCE(F637,1)=5)+(SEQUENCE(F637,1)=8)>0,IF(SEQUENCE(1,F635)>0,F639,0),0)'
Set-Center 'O683:AL690'

Set-Text 'N692' 'selection score = logits + bias'
Set-Right 'N692'
Set-Formula 'O692' '=O662#+O683#'
Set-Box 'O692:AL699'

# top-2 per token on the selection score
Set-Text 'N714' 'top-2 expert per token'
Set-Right 'N714'
$s11a = @(
  @('O714','O692:O699'), @('P714','P692:P699'), @('Q714','Q692:Q699'),
  @('R714','R692:R699'), @('S714','S692:S699'), @('T714','T692:T699'),
  @('U714','U692:U699'), @('V714','V692:V699'), @('W714','W692:W699'),
  @('X714','X692:X699'), @('Y714','Y692:Y699'), @('Z714','Z692:Z699'),
  @('AA714','AA692:AA699'), @('AB714','AB692:AB699'), @('AC714','AC692:AC699'),
  @('AD714','AD692:AD699'), @('AE714','AE692:AE699'), @('AF714','AF692:AF699'),
  @('AG714','AG692:AG699'), @('AH714','AH692:AH699'), @('AI714','AI692:AI699'),
  @('AJ714','AJ692:AJ699'), @('AK714','AK692:AK699'), @('AL714','AL692:AL699')
)
foreach ($p in $s11a) {
  Set-Formula $p[0] ('=MATCH(LARGE(' + $p[1] + ',1),' + $p[1] + ',0)')
}
Set-Center 'O714:AL714'

# expert load: count how many of the 24 top-1 picks hit each expert.
# COUNTIF with an array criterion returns #NAME? in this Excel, so one
# COUNTIF per expert, spilling 8 results.
Set-Text 'N716' 'load per expert'
Set-Right 'N716'
Set-Formula 'O716' '=COUNTIF(O714:AL714,1)'
Set-Formula 'P716' '=COUNTIF(O714:AL714,2)'
Set-Formula 'Q716' '=COUNTIF(O714:AL714,3)'
Set-Formula 'R716' '=COUNTIF(O714:AL714,4)'
Set-Formula 'S716' '=COUNTIF(O714:AL714,5)'
Set-Formula 'T716' '=COUNTIF(O714:AL714,6)'
Set-Formula 'U716' '=COUNTIF(O714:AL714,7)'
Set-Formula 'V716' '=COUNTIF(O714:AL714,8)'
Set-Center 'O716:V716'
Set-Box 'O716:V716'

# normalised entropy of the load
Set-Text 'N718' 'normalised entropy'
Set-Right 'N718'
Set-Formula 'O718' '=LET(p,O716:V716,n,F637,t,SUM(p),-SUM(IF(p>0,p/t*LN(p/t),0))/LN(n))'
Set-Center 'O718'

Set-Text 'D720' 'Uniform load would give entropy 1.0. Three experts taking everything'
Set-Text 'D721' 'drives it toward 0. This is the failure the bias term in section 12 fixes.'

# ================================================================ SECTION 12
# Auxiliary-Loss-Free Load Balancing. Mirrors LoadBalancer in
# src/compact_v3/routing.py. A bias b is added to the selection score but
# never to the gate weight. Each update step: target = mean load, and
# b_i += rate if expert i is underloaded, b_i -= rate if overloaded. Three
# steps move the load toward uniform. The proof cell: W_gate is untouched by
# b, so the router's output is the same function of x.
Set-Text 'A730' '# Auxiliary-Loss-Free Load Balancing'
$ws.Range('A730').Font.Size = 36
$ws.Range('A730').Font.Bold = $true
$ws.Range('A730').Font.Color = $headerColor
$ws.Rows.Item(730).RowHeight = 48
$b12 = $ws.Range('A730:EA730').Borders.Item($xlEdgeBottom)
$b12.LineStyle = $xlContinuous
$b12.Weight = $xlThick

Set-Text 'E733' 'n_tokens'; Set-Num 'F733' 24
Set-Text 'E734' 'd_model';   Set-Num 'F734' 8
Set-Text 'E735' 'n_experts'; Set-Num 'F735' 8
Set-Text 'E736' 'top_k';     Set-Num 'F736' 2
# update_rate has to be large enough that three steps can undo the advantage.
# After 3 steps the bias gap between an overloaded and an underloaded expert is
# 6*rate, so it must clear the advantage of 3.0 with headroom: 0.75 gives 4.5.
# At the old 0.05 the gap was 0.3 and the load barely moved; at 0.5 the gap is
# exactly 3.0, which only ties the advantage and stalls. Measured over 30
# draws, 0.75 shrinks the spread every time and never widens it.
Set-Text 'E737' 'update_rate'; Set-Num 'F737' 0.75
Set-Formula 'F738' '=F639'
Set-Text 'E738' 'advantage'
Set-Right 'E733:E738'
Set-Center 'F733:F738'

# input and router, same as section 11 but the bias is learned
Set-Text 'N733' 'pos'
Set-Right 'N733'
Set-Formula 'O733' '=SEQUENCE(1,F733,1)'
Set-Center 'O733:T733'
Set-Text 'N734' 'X (24 tokens)'
Set-Right 'N734'
Set-Formula 'O734' '=(RANDARRAY(F734,F733)-0.5)*2'
Set-Box 'O734:AL741'

Set-Text 'D743' 'Router'
Set-Bold 'D743'
Set-Text 'D744' 'W_router'
Set-Right 'D744'
Set-Formula 'E744' '=(RANDARRAY(F735,F734)-0.5)*2'
Set-Weights 'E744:L751'
# This section has to start from a collapsed router, or there is nothing for
# the bias to correct. Previously it used a plain random router, whose load is
# already near uniform, so three small updates moved the spread by noise and
# the "load spread shrinks" claim failed on 3 draws in 10. Carrying section
# 11's engineered advantage (+1.5 to experts 2, 5 and 8) forward gives the
# bias a real imbalance to undo, which is also the section chain working as
# intended: 11 breaks it, 12 fixes it.
Set-Text 'AO743' 'advantage (experts 2, 5, 8)'
Set-Formula 'AO744' '=IF((SEQUENCE(F735,1)=2)+(SEQUENCE(F735,1)=5)+(SEQUENCE(F735,1)=8)>0,IF(SEQUENCE(1,F733)>0,F738,0),0)'
Set-Box 'AO744:BL751'

Set-Text 'N744' 'logits'
Set-Right 'N744'
Set-Formula 'O744' '=MMULT(E744#,O734#)+AO744#'
Set-Box 'O744:AL751'

# b0 = 0, then three updates. Each step: b + rate*sign(target - load).
# The bias is an 8x1 column in E; b1..b3 follow in F, G, H. Counts are 1x8
# rows; they are transposed to 8x1 via TRANSPOSE of the plain range.
Set-Text 'D752' 'bias b (3 update steps)'
Set-Bold 'D752'
Set-Text 'N753' 'b0'
Set-Right 'N753'
Set-Num 'E753' 0; Set-Num 'E754' 0; Set-Num 'E755' 0; Set-Num 'E756' 0
Set-Num 'E757' 0; Set-Num 'E758' 0; Set-Num 'E759' 0; Set-Num 'E760' 0
Set-Center 'E753:E760'

# step 1: score0 = logits + broadcast(b0)
Set-Text 'N762' 'score0 = logits + b0'
Set-Right 'N762'
Set-Formula 'O762' '=O744#+IF(SEQUENCE(1,F733)>0,E753:E760,0)'
Set-Box 'O762:AL769'
Set-Text 'N771' 'load0'
Set-Right 'N771'
$load0 = @(
  @('O771','O762:O769'), @('P771','P762:P769'), @('Q771','Q762:Q769'),
  @('R771','R762:R769'), @('S771','S762:S769'), @('T771','T762:T769'),
  @('U771','U762:U769'), @('V771','V762:V769'), @('W771','W762:W769'),
  @('X771','X762:X769'), @('Y771','Y762:Y769'), @('Z771','Z762:Z769'),
  @('AA771','AA762:AA769'), @('AB771','AB762:AB769'), @('AC771','AC762:AC769'),
  @('AD771','AD762:AD769'), @('AE771','AE762:AE769'), @('AF771','AF762:AF769'),
  @('AG771','AG762:AG769'), @('AH771','AH762:AH769'), @('AI771','AI762:AI769'),
  @('AJ771','AJ762:AJ769'), @('AK771','AK762:AK769'), @('AL771','AL762:AL769')
)
foreach ($p in $load0) {
  Set-Formula $p[0] ('=MATCH(LARGE(' + $p[1] + ',1),' + $p[1] + ',0)')
}
Set-Center 'O771:AL771'
Set-Text 'N773' 'count0'
Set-Right 'N773'
Set-Formula 'O773' '=COUNTIF(O771:AL771,1)'
Set-Formula 'P773' '=COUNTIF(O771:AL771,2)'
Set-Formula 'Q773' '=COUNTIF(O771:AL771,3)'
Set-Formula 'R773' '=COUNTIF(O771:AL771,4)'
Set-Formula 'S773' '=COUNTIF(O771:AL771,5)'
Set-Formula 'T773' '=COUNTIF(O771:AL771,6)'
Set-Formula 'U773' '=COUNTIF(O771:AL771,7)'
Set-Formula 'V773' '=COUNTIF(O771:AL771,8)'
Set-Center 'O773:V773'

Set-Text 'N775' 'b1 = b0 + rate*sign(target - count0)'
Set-Right 'N775'
Set-Formula 'E775' '=E753:E760+F737*IF(TRANSPOSE(O773:V773)<F733/F735,1,-1)'
Set-Center 'E775:E782'

# step 2
Set-Text 'N784' 'score1 = logits + b1'
Set-Right 'N784'
Set-Formula 'O784' '=O744#+IF(SEQUENCE(1,F733)>0,E775:E782,0)'
Set-Box 'O784:AL791'
Set-Text 'N793' 'load1'
Set-Right 'N793'
$load1 = @(
  @('O793','O784:O791'), @('P793','P784:P791'), @('Q793','Q784:Q791'),
  @('R793','R784:R791'), @('S793','S784:S791'), @('T793','T784:T791'),
  @('U793','U784:U791'), @('V793','V784:V791'), @('W793','W784:W791'),
  @('X793','X784:X791'), @('Y793','Y784:Y791'), @('Z793','Z784:Z791'),
  @('AA793','AA784:AA791'), @('AB793','AB784:AB791'), @('AC793','AC784:AC791'),
  @('AD793','AD784:AD791'), @('AE793','AE784:AE791'), @('AF793','AF784:AF791'),
  @('AG793','AG784:AG791'), @('AH793','AH784:AH791'), @('AI793','AI784:AI791'),
  @('AJ793','AJ784:AJ791'), @('AK793','AK784:AK791'), @('AL793','AL784:AL791')
)
foreach ($p in $load1) {
  Set-Formula $p[0] ('=MATCH(LARGE(' + $p[1] + ',1),' + $p[1] + ',0)')
}
Set-Center 'O793:AL793'
Set-Text 'N795' 'count1'
Set-Right 'N795'
Set-Formula 'O795' '=COUNTIF(O793:AL793,1)'
Set-Formula 'P795' '=COUNTIF(O793:AL793,2)'
Set-Formula 'Q795' '=COUNTIF(O793:AL793,3)'
Set-Formula 'R795' '=COUNTIF(O793:AL793,4)'
Set-Formula 'S795' '=COUNTIF(O793:AL793,5)'
Set-Formula 'T795' '=COUNTIF(O793:AL793,6)'
Set-Formula 'U795' '=COUNTIF(O793:AL793,7)'
Set-Formula 'V795' '=COUNTIF(O793:AL793,8)'
Set-Center 'O795:V795'

Set-Text 'N797' 'b2 = b1 + rate*sign(target - count1)'
Set-Right 'N797'
Set-Formula 'E797' '=E775:E782+F737*IF(TRANSPOSE(O795:V795)<F733/F735,1,-1)'
Set-Center 'E797:E804'

# step 3
Set-Text 'N806' 'score2 = logits + b2'
Set-Right 'N806'
Set-Formula 'O806' '=O744#+IF(SEQUENCE(1,F733)>0,E797:E804,0)'
Set-Box 'O806:AL813'
Set-Text 'N815' 'load2'
Set-Right 'N815'
$load2 = @(
  @('O815','O806:O813'), @('P815','P806:P813'), @('Q815','Q806:Q813'),
  @('R815','R806:R813'), @('S815','S806:S813'), @('T815','T806:T813'),
  @('U815','U806:U813'), @('V815','V806:V813'), @('W815','W806:W813'),
  @('X815','X806:X813'), @('Y815','Y806:Y813'), @('Z815','Z806:Z813'),
  @('AA815','AA806:AA813'), @('AB815','AB806:AB813'), @('AC815','AC806:AC813'),
  @('AD815','AD806:AD813'), @('AE815','AE806:AE813'), @('AF815','AF806:AF813'),
  @('AG815','AG806:AG813'), @('AH815','AH806:AH813'), @('AI815','AI806:AI813'),
  @('AJ815','AJ806:AJ813'), @('AK815','AK806:AK813'), @('AL815','AL806:AL813')
)
foreach ($p in $load2) {
  Set-Formula $p[0] ('=MATCH(LARGE(' + $p[1] + ',1),' + $p[1] + ',0)')
}
Set-Center 'O815:AL815'
Set-Text 'N817' 'count2'
Set-Right 'N817'
Set-Formula 'O817' '=COUNTIF(O815:AL815,1)'
Set-Formula 'P817' '=COUNTIF(O815:AL815,2)'
Set-Formula 'Q817' '=COUNTIF(O815:AL815,3)'
Set-Formula 'R817' '=COUNTIF(O815:AL815,4)'
Set-Formula 'S817' '=COUNTIF(O815:AL815,5)'
Set-Formula 'T817' '=COUNTIF(O815:AL815,6)'
Set-Formula 'U817' '=COUNTIF(O815:AL815,7)'
Set-Formula 'V817' '=COUNTIF(O815:AL815,8)'
Set-Center 'O817:V817'

Set-Text 'N819' 'b3 = b2 + rate*sign(target - count2)'
Set-Right 'N819'
Set-Formula 'E819' '=E797:E804+F737*IF(TRANSPOSE(O817:V817)<F733/F735,1,-1)'
Set-Center 'E819:E826'

# Proof: W_router (the gate weight) is untouched by b.
#
# This block sits at row 830, below everything else in the section. It used to
# start at row 800 and collided with two spills: the 8-row b2 column occupies
# E797:E804, and the 8x24 score2 block occupies O806:AL813. Writing E801 and
# O801 into those ranges blocked both spills, which cascaded into #SPILL! at
# E797, O801, O806, load2 and b3. Any block placed here must clear row 826,
# where the b3 column ends.
Set-Text 'D830' 'Proof: the gate weight does not change'
Set-Bold 'D830'
Set-Text 'D831' 'W_router copy'
Set-Right 'D831'
Set-Formula 'E831' '=E744#'
Set-Weights 'E831:L838'
# Heading goes above the block, not to its left: a right-aligned label in N831
# overflows leftward and gets clipped by the orange weight block ending at L.
Set-Text 'O830' 'W_router - W_router copy'
Set-Formula 'O831' '=E744#-E831#'
Set-Center 'O831:V838'
Set-Text 'D840' 'All zeros: b changes the selection, never the gate weight.'
Set-Text 'D841' 'The router output is the same function of x at every step.'

# ================================================================ SECTION 13
# The V3 Block and the Sequence-Wise Balance Loss. The assembly section: the
# first time a norm, an attention sublayer, a residual and an FFN sublayer sit
# in one stack. Mirrors CompactV3Block.forward in src/compact_v3/block.py:
#     x = x + attention(attn_norm(x))
#     x = x + ffn(ffn_norm(x))
# Layer 0 uses the dense SwiGLU (n_dense_layers = 1 in config.py); layers 1 and
# up swap it for the MoE. The balance loss mirrors
# RoutingResult.sequence_balance_loss in routing.py.
#
# The attention here is a single causal head, not full MLA. Sections 4 to 7
# already build MLA; repeating it would bury the thing this section is for,
# which is the residual structure around the sublayers.
Set-Text 'A860' '# The V3 Block and the Sequence-Wise Balance Loss'
$ws.Range('A860').Font.Size = 36
$ws.Range('A860').Font.Bold = $true
$ws.Range('A860').Font.Color = $headerColor
$ws.Rows.Item(860).RowHeight = 48
$b13 = $ws.Range('A860:BB860').Borders.Item($xlEdgeBottom)
$b13.LineStyle = $xlContinuous
$b13.Weight = $xlThick

Set-Text 'E863' 'seq';           Set-Num 'F863' 6
Set-Text 'E864' 'd_model';       Set-Num 'F864' 8
Set-Text 'E865' 'd_k';           Set-Num 'F865' 4
Set-Text 'E866' 'expert_hidden'; Set-Num 'F866' 6
Set-Text 'E867' 'n_experts';     Set-Num 'F867' 8
Set-Text 'E868' 'top_k';         Set-Num 'F868' 2
Set-Text 'E869' 'rms_eps';       Set-Num 'F869' 0.000001
Set-Text 'E870' 'balance_coef';  Set-Num 'F870' 0.0001
Set-Text 'E871' 'route_scale';   Set-Num 'F871' 0.75
Set-Right 'E863:E871'
Set-Center 'F863:F871'
Set-Sci 'F869'
Set-Sci 'F870'

Set-Text 'N863' 'pos'
Set-Right 'N863'
Set-Formula 'O863' '=SEQUENCE(1,F863,1)'
Set-Center 'O863:T863'
Set-Formula 'O865' '="x"&O863#'
Set-Center 'O865:T865'
Set-Text 'N866' 'X'
Set-Right 'N866'
Set-Formula 'O866' '=(RANDARRAY(F864,F863)-0.5)*2'
Set-Box 'O866:T873'

# --- RMSNorm. The only normalisation with no mean subtraction: divide by the
# root mean square down the feature axis, then scale by a learned gamma.
# The ones-vector trick gives a per-token sum of squares without BYCOL.
Set-Text 'D875' 'RMSNorm (no mean subtraction)'
Set-Bold 'D875'
Set-Text 'N876' 'rms'
Set-Right 'N876'
Set-Formula 'O876' '=SQRT(MMULT(TRANSPOSE(SEQUENCE(F864,1,1,0)),O866#*O866#)/F864+F869)'
Set-Box 'O876:T876'

Set-Text 'D879' 'gamma'
Set-Right 'D879'
# Initialised to ones in norms.py; shown trained so the scaling is visible.
Set-Formula 'E879' '=RANDARRAY(F864,1)*0.5+0.75'
Set-Weights 'E879:E886'
Set-Text 'N879' 'n = X / rms * gamma'
Set-Right 'N879'
Set-Formula 'O879' '=O866#/O876#*E879#'
Set-Box 'O879:T886'

# --- attention sublayer
Set-Text 'D889' 'Attention sublayer'
Set-Bold 'D889'
Set-Text 'D890' 'Wq'
Set-Right 'D890'
Set-Formula 'E890' '=(RANDARRAY(F865,F864)-0.5)*2'
Set-Weights 'E890:L893'
Set-Text 'N890' 'Q = Wq n'
Set-Right 'N890'
Set-Formula 'O890' '=MMULT(E890#,O879#)'
Set-Box 'O890:T893'

Set-Text 'D895' 'Wk'
Set-Right 'D895'
Set-Formula 'E895' '=(RANDARRAY(F865,F864)-0.5)*2'
Set-Weights 'E895:L898'
Set-Text 'N895' 'K = Wk n'
Set-Right 'N895'
Set-Formula 'O895' '=MMULT(E895#,O879#)'
Set-Box 'O895:T898'

Set-Text 'D900' 'Wv'
Set-Right 'D900'
Set-Formula 'E900' '=(RANDARRAY(F865,F864)-0.5)*2'
Set-Weights 'E900:L903'
Set-Text 'N900' 'V = Wv n'
Set-Right 'N900'
Set-Formula 'O900' '=MMULT(E900#,O879#)'
Set-Box 'O900:T903'

Set-Text 'V889' 'K^T'
Set-Formula 'W890' '=TRANSPOSE(O895#)'
Set-Box 'W890:Z895'

Set-Text 'AB889' 'S = K^T Q / sqrt(dk)'
Set-Formula 'AB890' '=MMULT(W890#,O890#)/SQRT(F865)'
Set-Box 'AB890:AG895'

# Causal softmax, one LET per query column. The mask is inline: query j may
# read key i only when i <= j, so column j carries IF(SEQUENCE(seq,1)<=j,1,0).
Set-Text 'AI889' 'A = causal softmax'
$s13a = @(
  @('AI890', 'AB890:AB895', 1), @('AJ890', 'AC890:AC895', 2), @('AK890', 'AD890:AD895', 3),
  @('AL890', 'AE890:AE895', 4), @('AM890', 'AF890:AF895', 5), @('AN890', 'AG890:AG895', 6)
)
foreach ($c in $s13a) {
  Set-Formula $c[0] ('=LET(z,' + $c[1] + ',m,IF(SEQUENCE(F863,1)<=' + $c[2] + ',1,0),m*EXP(z)/SUM(m*EXP(z)))')
}
Set-Box 'AI890:AN895'

Set-Text 'AP889' 'O_attn = V A'
Set-Formula 'AP890' '=MMULT(O900#,AI890:AN895)'
Set-Box 'AP890:AU893'

Set-Text 'D905' 'W_O'
Set-Right 'D905'
Set-Formula 'E905' '=(RANDARRAY(F864,F865)-0.5)*2'
Set-Weights 'E905:H912'
Set-Text 'N905' 'attn_out = W_O O_attn'
Set-Right 'N905'
Set-Formula 'O905' '=MMULT(E905#,AP890#)'
Set-Box 'O905:T912'

# The residual add. X arrives here unnormalised: the norm feeds the sublayer,
# never the highway. That is what "pre-norm" means and it is the single most
# copied-wrong line in transformer implementations.
Set-Text 'AW889' 'x1 = X + attn_out   (residual, X not normalised)'
Set-Formula 'AW890' '=O866#+O905#'
Set-Box 'AW890:BB897'

# --- FFN sublayer, layer 0 form: dense SwiGLU
Set-Text 'D915' 'FFN sublayer, layer 0: dense SwiGLU'
Set-Bold 'D915'
Set-Text 'D916' 'gamma2'
Set-Right 'D916'
Set-Formula 'E916' '=RANDARRAY(F864,1)*0.5+0.75'
Set-Weights 'E916:E923'
Set-Text 'N916' 'rms2'
Set-Right 'N916'
Set-Formula 'O916' '=SQRT(MMULT(TRANSPOSE(SEQUENCE(F864,1,1,0)),AW890#*AW890#)/F864+F869)'
Set-Box 'O916:T916'
Set-Text 'N918' 'n2 = x1 / rms2 * gamma2'
Set-Right 'N918'
Set-Formula 'O918' '=AW890#/O916#*E916#'
Set-Box 'O918:T925'

Set-Text 'D927' 'W_gate'
Set-Right 'D927'
Set-Formula 'E927' '=(RANDARRAY(F866,F864)-0.5)*2'
Set-Weights 'E927:L932'
Set-Text 'O926' 'g = silu(W_g n2)'
Set-Formula 'O927' '=MMULT(E927#,O918#)/(1+EXP(-MMULT(E927#,O918#)))'
Set-Box 'O927:T932'

Set-Text 'D934' 'W_value'
Set-Right 'D934'
Set-Formula 'E934' '=(RANDARRAY(F866,F864)-0.5)*2'
Set-Weights 'E934:L939'
Set-Text 'N934' 'v = W_v n2'
Set-Right 'N934'
Set-Formula 'O934' '=MMULT(E934#,O918#)'
Set-Box 'O934:T939'

Set-Text 'D941' 'W_out'
Set-Right 'D941'
Set-Formula 'E941' '=(RANDARRAY(F864,F866)-0.5)*2'
Set-Weights 'E941:J948'
Set-Text 'N941' 'ffn_out = W_o (g * v)'
Set-Right 'N941'
Set-Formula 'O941' '=MMULT(E941#,O927#*O934#)'
Set-Box 'O941:T948'

Set-Text 'W940' 'y = x1 + ffn_out   (block output)'
Set-Formula 'W941' '=AW890#+O941#'
Set-Box 'W941:AB948'

# --- layer 1 and up: the dense FFN becomes MoE
Set-Text 'D950' 'FFN sublayer, layer 1 and up: MoE replaces the dense FFN'
Set-Bold 'D950'
Set-Text 'D952' 'W_router'
Set-Right 'D952'
Set-Formula 'E952' '=(RANDARRAY(F867,F864)-0.5)*2'
Set-Weights 'E952:L959'
Set-Text 'O951' 'logits = W_r n2'
Set-Formula 'O952' '=MMULT(E952#,O918#)'
Set-Box 'O952:T959'

# V3 scores experts with a sigmoid per expert, not a softmax over all of them.
Set-Text 'N961' 'affinity = sigmoid(logits)'
Set-Right 'N961'
Set-Formula 'O961' '=1/(1+EXP(-O952#))'
Set-Box 'O961:T968'

Set-Text 'N970' 'top-1 expert'
Set-Right 'N970'
$s13t1 = @(
  @('O970', 'O952:O959'), @('P970', 'P952:P959'), @('Q970', 'Q952:Q959'),
  @('R970', 'R952:R959'), @('S970', 'S952:S959'), @('T970', 'T952:T959')
)
foreach ($p in $s13t1) { Set-Formula $p[0] ('=MATCH(LARGE(' + $p[1] + ',1),' + $p[1] + ',0)') }
Set-Center 'O970:T970'
Set-Text 'N971' 'top-2 expert'
Set-Right 'N971'
$s13t2 = @(
  @('O971', 'O952:O959'), @('P971', 'P952:P959'), @('Q971', 'Q952:Q959'),
  @('R971', 'R952:R959'), @('S971', 'S952:S959'), @('T971', 'T952:T959')
)
foreach ($p in $s13t2) { Set-Formula $p[0] ('=MATCH(LARGE(' + $p[1] + ',2),' + $p[1] + ',0)') }
Set-Center 'O971:T971'

# Weights renormalise over the selected pair only, then take route_scale.
Set-Text 'N973' 'w1'
Set-Right 'N973'
$s13w1 = @(
  @('O973', 'O961:O968', 'O970', 'O971'), @('P973', 'P961:P968', 'P970', 'P971'),
  @('Q973', 'Q961:Q968', 'Q970', 'Q971'), @('R973', 'R961:R968', 'R970', 'R971'),
  @('S973', 'S961:S968', 'S970', 'S971'), @('T973', 'T961:T968', 'T970', 'T971')
)
foreach ($p in $s13w1) {
  Set-Formula $p[0] ('=INDEX(' + $p[1] + ',' + $p[2] + ')/(INDEX(' + $p[1] + ',' + $p[2] + ')+INDEX(' + $p[1] + ',' + $p[3] + '))*$F$871')
}
Set-Text 'N974' 'w2'
Set-Right 'N974'
$s13w2 = @(
  @('O974', 'O961:O968', 'O971', 'O970'), @('P974', 'P961:P968', 'P971', 'P970'),
  @('Q974', 'Q961:Q968', 'Q971', 'Q970'), @('R974', 'R961:R968', 'R971', 'R970'),
  @('S974', 'S961:S968', 'S971', 'S970'), @('T974', 'T961:T968', 'T971', 'T970')
)
foreach ($p in $s13w2) {
  Set-Formula $p[0] ('=INDEX(' + $p[1] + ',' + $p[2] + ')/(INDEX(' + $p[1] + ',' + $p[2] + ')+INDEX(' + $p[1] + ',' + $p[3] + '))*$F$871')
}
Set-Center 'O973:T974'

# --- the sequence-wise balance loss
# routing.py: frequency = count / (tokens*top_k) * n_experts, P = mean affinity
# per expert over the sequence, loss = coefficient * sum(frequency * P).
Set-Text 'D977' 'Sequence-wise balance loss'
Set-Bold 'D977'
Set-Formula 'O977' '="e"&SEQUENCE(1,F867,1)'
Set-Center 'O977:V977'
Set-Text 'N978' 'count'
Set-Right 'N978'
for ($e = 1; $e -le 8; $e++) {
  $col = [char](79 + $e - 1)   # O is 79
  Set-Formula "$col`978" ('=COUNTIF($O$970:$T$971,' + $e + ')')
}
Set-Center 'O978:V978'
Set-Text 'N979' 'f = count*n_experts/(tokens*top_k)'
Set-Right 'N979'
Set-Formula 'O979' '=O978:V978*F867/(F863*F868)'
Set-Center 'O979:V979'
Set-Text 'N980' 'P = mean affinity per expert'
Set-Right 'N980'
Set-Formula 'O980' '=TRANSPOSE(MMULT(O961#,SEQUENCE(F863,1,1,0))/F863)'
Set-Center 'O980:V980'
Set-Text 'N981' 'f * P'
Set-Right 'N981'
Set-Formula 'O981' '=O979#*O980#'
Set-Box 'O981:V981'
Set-Text 'N983' 'balance loss = coef * sum(f*P)'
Set-Right 'N983'
Set-Formula 'O983' '=F870*SUM(O981#)'
Set-Box 'O983'
Set-Sci 'O983'
Set-Text 'D985' 'Tiny by design. The bias in section 12 does the real balancing; this term'
Set-Text 'D986' 'is a gradient nudge, and at coefficient 1e-4 it barely moves the loss.'

# --- parameter ledger
Set-Text 'D989' 'Parameters: total against active per token'
Set-Bold 'D989'
Set-Text 'N990' 'attention'
Set-Right 'N990'
Set-Formula 'P990' '=3*F865*F864+F864*F865'
Set-Text 'N991' 'one SwiGLU expert'
Set-Right 'N991'
Set-Formula 'P991' '=2*F866*F864+F864*F866'
Set-Text 'N992' 'MoE total (8 routed + 1 shared + router)'
Set-Right 'N992'
Set-Formula 'P992' '=(F867+1)*P991+F867*F864'
Set-Text 'N993' 'MoE active (top_k routed + 1 shared + router)'
Set-Right 'N993'
Set-Formula 'P993' '=(F868+1)*P991+F867*F864'
Set-Text 'N994' 'block total'
Set-Right 'N994'
Set-Formula 'P994' '=P990+P992+2*F864'
Set-Text 'N995' 'block active'
Set-Right 'N995'
Set-Formula 'P995' '=P990+P993+2*F864'
Set-Text 'N996' 'active fraction'
Set-Right 'N996'
Set-Formula 'P996' '=P995/P994'
Set-Center 'P990:P996'

# ================================================================ SECTION 14
# Multi-Token Prediction. Mirrors MTPObjective in src/compact_v3/mtp.py.
# One module hangs off the main model's hidden states: merge h_i with the
# embedding of t_{i+1}, run one causal transformer block, and read t_{i+2}
# through the output head, which is tied to the token embedding.
#
# The alignment is the whole section. The module is fed t_{i+1} and predicts
# t_{i+2}. Feed it t_{i+2} instead and it reads the answer off its own input
# through the tied head and learns the identity map. See
# experiments/GATE_U_MTP_OBJECTIVE.md. The panel at the bottom shows why.
Set-Text 'A1010' '# Multi-Token Prediction'
$ws.Range('A1010').Font.Size = 36
$ws.Range('A1010').Font.Bold = $true
$ws.Range('A1010').Font.Color = $headerColor
$ws.Rows.Item(1010).RowHeight = 48
$b14 = $ws.Range('A1010:AT1010').Borders.Item($xlEdgeBottom)
$b14.LineStyle = $xlContinuous
$b14.Weight = $xlThick

Set-Text 'E1013' 'seq';            Set-Num 'F1013' 6
Set-Text 'E1014' 'd_model';        Set-Num 'F1014' 8
Set-Text 'E1015' 'vocab';          Set-Num 'F1015' 6
Set-Text 'E1016' 'horizon';        Set-Num 'F1016' 2
Set-Text 'E1017' 'aligned';        Set-Formula 'F1017' '=F1013-F1016'
Set-Text 'E1018' 'd_k';            Set-Num 'F1018' 4
Set-Text 'E1019' 'expert_hidden';  Set-Num 'F1019' 6
Set-Text 'E1020' 'rms_eps';        Set-Num 'F1020' 0.000001
Set-Text 'E1021' 'lambda';         Set-Num 'F1021' 0.3
Set-Text 'E1022' 'lambda_final';   Set-Num 'F1022' 0.1
Set-Text 'E1023' 'decay_fraction'; Set-Num 'F1023' 0.6757
Set-Right 'E1013:E1023'
Set-Center 'F1013:F1023'
Set-Sci 'F1020'

# --- the alignment. Three offset rows, which is the clearest way to show an
# off-by-one that is otherwise only arguable.
# Heading sits above the panel, not in column D: D1013 is flush against the
# config label in E1013 and gets clipped to "The al".
Set-Text 'N1012' 'The alignment'
Set-Bold 'N1012'
Set-Text 'N1013' 'pos'
Set-Right 'N1013'
Set-Formula 'O1013' '=SEQUENCE(1,F1013,1)'
Set-Center 'O1013:T1013'
Set-Text 'N1014' 'token t'
Set-Right 'N1014'
# Fixed, not random: the alignment has to be readable and stable across recalcs.
Set-Num 'O1014' 3; Set-Num 'P1014' 1; Set-Num 'Q1014' 4
Set-Num 'R1014' 1; Set-Num 'S1014' 5; Set-Num 'T1014' 2
Set-Box 'O1014:T1014'

Set-Text 'N1016' 'i (aligned)'
Set-Right 'N1016'
Set-Formula 'O1016' '=SEQUENCE(1,F1017,1)'
Set-Center 'O1016:R1016'
Set-Text 'N1017' 'hidden state used'
Set-Right 'N1017'
Set-Formula 'O1017' '="h"&O1016#'
Set-Center 'O1017:R1017'
Set-Text 'N1018' 'input token t(i+1)'
Set-Right 'N1018'
Set-Formula 'O1018' '=P1014:S1014'
Set-Box 'O1018:R1018'
Set-Text 'N1019' 'target t(i+2)'
Set-Right 'N1019'
Set-Formula 'O1019' '=Q1014:T1014'
Set-Box 'O1019:R1019'
Set-Text 'V1017' 'align_hidden_states drops the last 2 positions;'
Set-Text 'V1018' 'make_mtp_input_tokens takes t(2..5);'
Set-Text 'V1019' 'make_future_targets takes t(3..6).'

# --- token embedding, shared with the output head
Set-Text 'D1024' 'Token embedding, tied to the output head'
Set-Bold 'D1024'
Set-Text 'D1025' 'E'
Set-Right 'D1025'
# Columns are normalised to unit length. With raw random columns, E^T e_target
# peaked somewhere other than the target on about 1 draw in 40, because a
# longer off-target column can out-dot the target's own squared norm. The
# degenerate-objective panel below would then visibly contradict itself on
# recalc. Unit columns make its diagonal exactly 1.0 and every off-diagonal a
# cosine strictly below it, so the demonstration holds on every draw. LET binds
# RANDARRAY once, so the numerator and the norms come from the same draw.
Set-Formula 'E1025' '=LET(r,(RANDARRAY(F1014,F1015)-0.5)*2,r/SQRT(MMULT(TRANSPOSE(SEQUENCE(F1014,1,1,0)),r*r)))'
Set-Weights 'E1025:J1032'
# Lookup as a one-hot matmul, which is what an embedding table is.
Set-Text 'N1025' 'Emb(t(i+1))'
Set-Right 'N1025'
$s14emb = @(@('O1025', 'O1018'), @('P1025', 'P1018'), @('Q1025', 'Q1018'), @('R1025', 'R1018'))
foreach ($p in $s14emb) {
  Set-Formula $p[0] ('=MMULT($E$1025#,IF(SEQUENCE($F$1015,1)=' + $p[1] + ',1,0))')
}
Set-Box 'O1025:R1032'

Set-Text 'D1034' 'h from the main model (y in section 13)'
Set-Bold 'D1034'
Set-Text 'N1034' 'h'
Set-Right 'N1034'
Set-Formula 'O1034' '=(RANDARRAY(F1014,F1013)-0.5)*2'
Set-Box 'O1034:T1041'
Set-Text 'N1043' 'h aligned (drop last 2)'
Set-Right 'N1043'
Set-Formula 'O1043' '=O1034:R1041'
Set-Box 'O1043:R1050'

# --- normalise both, stack them, merge
Set-Text 'D1052' 'RMSNorm both inputs, concatenate, then merge'
Set-Bold 'D1052'
Set-Text 'D1053' 'gamma_h'
Set-Right 'D1053'
Set-Formula 'E1053' '=RANDARRAY(F1014,1)*0.5+0.75'
Set-Weights 'E1053:E1060'
Set-Text 'N1053' 'rms_h'
Set-Right 'N1053'
Set-Formula 'O1053' '=SQRT(MMULT(TRANSPOSE(SEQUENCE(F1014,1,1,0)),O1043#*O1043#)/F1014+F1020)'
Set-Box 'O1053:R1053'

Set-Text 'D1062' 'gamma_e'
Set-Right 'D1062'
Set-Formula 'E1062' '=RANDARRAY(F1014,1)*0.5+0.75'
Set-Weights 'E1062:E1069'
Set-Text 'N1062' 'rms_e'
Set-Right 'N1062'
# O1025# would be only the first of the four per-column embedding formulas,
# i.e. 8x1 rather than 8x4. Multi-column blocks built one column at a time have
# to be read as a plain range.
Set-Formula 'O1062' '=SQRT(MMULT(TRANSPOSE(SEQUENCE(F1014,1,1,0)),O1025:R1032*O1025:R1032)/F1014+F1020)'
Set-Box 'O1062:R1062'

# Excel has no VSTACK here, so the concatenation is physical: the two normed
# blocks sit directly on top of each other and the merge reads one 16x4 range.
Set-Text 'AB1054' 'concat: RMSNorm(h) stacked on RMSNorm(Emb)'
Set-Formula 'AB1055' '=O1043#/O1053#*E1053#'
Set-Formula 'AB1063' '=O1025:R1032/O1062#*E1062#'
Set-Box 'AB1055:AE1070'

Set-Text 'E1071' 'M (d_model x 2*d_model)'
Set-Formula 'E1072' '=(RANDARRAY(F1014,2*F1014)-0.5)*2'
Set-Weights 'E1072:T1079'
Set-Text 'AG1072' 'merged = M concat'
Set-Formula 'AG1073' '=MMULT(E1072#,AB1055:AE1070)'
Set-Box 'AG1073:AJ1080'

# --- one causal transformer block over the merged states
Set-Text 'D1083' 'One causal transformer block'
Set-Bold 'D1083'
Set-Text 'D1084' 'gamma3'
Set-Right 'D1084'
Set-Formula 'E1084' '=RANDARRAY(F1014,1)*0.5+0.75'
Set-Weights 'E1084:E1091'
Set-Text 'N1084' 'rms3'
Set-Right 'N1084'
Set-Formula 'O1084' '=SQRT(MMULT(TRANSPOSE(SEQUENCE(F1014,1,1,0)),AG1073#*AG1073#)/F1014+F1020)'
Set-Box 'O1084:R1084'
Set-Text 'N1086' 'n3'
Set-Right 'N1086'
Set-Formula 'O1086' '=AG1073#/O1084#*E1084#'
Set-Box 'O1086:R1093'

Set-Text 'D1095' 'Wq'
Set-Right 'D1095'
Set-Formula 'E1095' '=(RANDARRAY(F1018,F1014)-0.5)*2'
Set-Weights 'E1095:L1098'
Set-Text 'N1095' 'Q'
Set-Right 'N1095'
Set-Formula 'O1095' '=MMULT(E1095#,O1086#)'
Set-Box 'O1095:R1098'
Set-Text 'D1100' 'Wk'
Set-Right 'D1100'
Set-Formula 'E1100' '=(RANDARRAY(F1018,F1014)-0.5)*2'
Set-Weights 'E1100:L1103'
Set-Text 'N1100' 'K'
Set-Right 'N1100'
Set-Formula 'O1100' '=MMULT(E1100#,O1086#)'
Set-Box 'O1100:R1103'
Set-Text 'D1105' 'Wv'
Set-Right 'D1105'
Set-Formula 'E1105' '=(RANDARRAY(F1018,F1014)-0.5)*2'
Set-Weights 'E1105:L1108'
Set-Text 'N1105' 'V'
Set-Right 'N1105'
Set-Formula 'O1105' '=MMULT(E1105#,O1086#)'
Set-Box 'O1105:R1108'

Set-Text 'V1094' 'K^T'
Set-Formula 'W1095' '=TRANSPOSE(O1100#)'
Set-Box 'W1095:Z1098'
Set-Text 'AB1094' 'S = K^T Q / sqrt(dk)'
Set-Formula 'AB1095' '=MMULT(W1095#,O1095#)/SQRT(F1018)'
Set-Box 'AB1095:AE1098'
# The MTP block is causal too: it may not read a position it is meant to predict.
Set-Text 'AG1094' 'A = causal softmax'
$s14a = @(
  @('AG1095', 'AB1095:AB1098', 1), @('AH1095', 'AC1095:AC1098', 2),
  @('AI1095', 'AD1095:AD1098', 3), @('AJ1095', 'AE1095:AE1098', 4)
)
foreach ($c in $s14a) {
  Set-Formula $c[0] ('=LET(z,' + $c[1] + ',m,IF(SEQUENCE(F1017,1)<=' + $c[2] + ',1,0),m*EXP(z)/SUM(m*EXP(z)))')
}
Set-Box 'AG1095:AJ1098'
Set-Text 'AL1094' 'O_attn = V A'
Set-Formula 'AL1095' '=MMULT(O1105#,AG1095:AJ1098)'
Set-Box 'AL1095:AO1098'

Set-Text 'D1110' 'W_O'
Set-Right 'D1110'
Set-Formula 'E1110' '=(RANDARRAY(F1014,F1018)-0.5)*2'
Set-Weights 'E1110:H1117'
Set-Text 'N1110' 'attn_out'
Set-Right 'N1110'
Set-Formula 'O1110' '=MMULT(E1110#,AL1095#)'
Set-Box 'O1110:R1117'
Set-Text 'AL1109' 'r1 = merged + attn_out'
Set-Formula 'AL1110' '=AG1073#+O1110#'
Set-Box 'AL1110:AO1117'

Set-Text 'D1119' 'FFN sublayer'
Set-Bold 'D1119'
Set-Text 'D1120' 'gamma4'
Set-Right 'D1120'
Set-Formula 'E1120' '=RANDARRAY(F1014,1)*0.5+0.75'
Set-Weights 'E1120:E1127'
Set-Text 'N1120' 'rms4'
Set-Right 'N1120'
Set-Formula 'O1120' '=SQRT(MMULT(TRANSPOSE(SEQUENCE(F1014,1,1,0)),AL1110#*AL1110#)/F1014+F1020)'
Set-Box 'O1120:R1120'
Set-Text 'N1122' 'n4'
Set-Right 'N1122'
Set-Formula 'O1122' '=AL1110#/O1120#*E1120#'
Set-Box 'O1122:R1129'

Set-Text 'D1131' 'W_gate'
Set-Right 'D1131'
Set-Formula 'E1131' '=(RANDARRAY(F1019,F1014)-0.5)*2'
Set-Weights 'E1131:L1136'
Set-Text 'O1130' 'g = silu(W_g n4)'
Set-Formula 'O1131' '=MMULT(E1131#,O1122#)/(1+EXP(-MMULT(E1131#,O1122#)))'
Set-Box 'O1131:R1136'
Set-Text 'D1138' 'W_value'
Set-Right 'D1138'
Set-Formula 'E1138' '=(RANDARRAY(F1019,F1014)-0.5)*2'
Set-Weights 'E1138:L1143'
Set-Text 'N1138' 'v = W_v n4'
Set-Right 'N1138'
Set-Formula 'O1138' '=MMULT(E1138#,O1122#)'
Set-Box 'O1138:R1143'
Set-Text 'D1145' 'W_out'
Set-Right 'D1145'
Set-Formula 'E1145' '=(RANDARRAY(F1014,F1019)-0.5)*2'
Set-Weights 'E1145:J1152'
Set-Text 'N1145' 'ffn_out'
Set-Right 'N1145'
Set-Formula 'O1145' '=MMULT(E1145#,O1131#*O1138#)'
Set-Box 'O1145:R1152'
Set-Text 'V1144' 'refined = r1 + ffn_out'
Set-Formula 'V1145' '=AL1110#+O1145#'
Set-Box 'V1145:Y1152'

# --- output head, tied to the embedding
Set-Text 'D1155' 'Output head, tied: logits = E^T final_norm(refined)'
Set-Bold 'D1155'
Set-Text 'D1156' 'gamma5'
Set-Right 'D1156'
Set-Formula 'E1156' '=RANDARRAY(F1014,1)*0.5+0.75'
Set-Weights 'E1156:E1163'
Set-Text 'N1156' 'rms5'
Set-Right 'N1156'
Set-Formula 'O1156' '=SQRT(MMULT(TRANSPOSE(SEQUENCE(F1014,1,1,0)),V1145#*V1145#)/F1014+F1020)'
Set-Box 'O1156:R1156'
Set-Text 'N1158' 'nf'
Set-Right 'N1158'
Set-Formula 'O1158' '=V1145#/O1156#*E1156#'
Set-Box 'O1158:R1165'

Set-Formula 'N1167' '="v"&SEQUENCE(F1015,1,1,1)'
Set-Right 'N1167:N1172'
Set-Text 'O1166' 'logits (vocab x aligned)'
Set-Formula 'O1167' '=MMULT(TRANSPOSE(E1025#),O1158#)'
Set-Box 'O1167:R1172'

Set-Text 'D1174' 'p = softmax(logits)'
Set-Right 'D1174'
$s14p = @(
  @('O1174', 'O1167:O1172'), @('P1174', 'P1167:P1172'),
  @('Q1174', 'Q1167:Q1172'), @('R1174', 'R1167:R1172')
)
foreach ($p in $s14p) { Set-Formula $p[0] ('=LET(z,' + $p[1] + ',EXP(z)/SUM(EXP(z)))') }
Set-Box 'O1174:R1179'

Set-Text 'N1181' 'p at target'
Set-Right 'N1181'
$s14pt = @(
  @('O1181', 'O1174:O1179', 'O1019'), @('P1181', 'P1174:P1179', 'P1019'),
  @('Q1181', 'Q1174:Q1179', 'Q1019'), @('R1181', 'R1174:R1179', 'R1019')
)
foreach ($p in $s14pt) { Set-Formula $p[0] ('=INDEX(' + $p[1] + ',' + $p[2] + ')') }
Set-Center 'O1181:R1181'
Set-Text 'N1182' 'loss = -ln(p)'
Set-Right 'N1182'
Set-Formula 'O1182' '=-LN(O1181:R1181)'
Set-Box 'O1182:R1182'
Set-Text 'N1184' 'mtp loss (mean)'
Set-Right 'N1184'
Set-Formula 'O1184' '=AVERAGE(O1182#)'
Set-Box 'O1184'

Set-Text 'N1186' 'main loss (from the model head)'
Set-Right 'N1186'
Set-Num 'P1186' 2.1
Set-Text 'N1187' 'combined = main + lambda * mtp'
Set-Right 'N1187'
Set-Formula 'P1187' '=P1186+F1021*O1184'
Set-Center 'P1186:P1187'
Set-Text 'N1189' 'lambda at 50% of training'
Set-Right 'N1189'
Set-Formula 'P1189' '=IF(0.5<F1023,F1021,F1022)'
Set-Text 'N1190' 'lambda at 80% of training'
Set-Right 'N1190'
Set-Formula 'P1190' '=IF(0.8<F1023,F1021,F1022)'
Set-Center 'P1189:P1190'

# --- the same module as a speculative draft head
Set-Text 'D1193' 'The same module as a speculative draft head'
Set-Bold 'D1193'
Set-Text 'N1194' 'proposal = argmax logits'
Set-Right 'N1194'
$s14prop = @(
  @('O1194', 'O1167:O1172'), @('P1194', 'P1167:P1172'),
  @('Q1194', 'Q1167:Q1172'), @('R1194', 'R1167:R1172')
)
foreach ($p in $s14prop) { Set-Formula $p[0] ('=MATCH(LARGE(' + $p[1] + ',1),' + $p[1] + ',0)') }
Set-Center 'O1194:R1194'
Set-Text 'N1195' 'actual t(i+2)'
Set-Right 'N1195'
Set-Formula 'O1195' '=O1019#'
Set-Center 'O1195:R1195'
Set-Text 'N1196' 'accepted'
Set-Right 'N1196'
Set-Formula 'O1196' '=IF(O1194:R1194=O1195#,1,0)'
Set-Box 'O1196:R1196'
Set-Text 'N1197' 'acceptance rate'
Set-Right 'N1197'
Set-Formula 'O1197' '=AVERAGE(O1196#)'
Set-Box 'O1197'
Set-Text 'V1195' 'Untrained weights, so the rate is chance. What matters is that the'
Set-Text 'V1196' 'module needs only t(i+1), which the main model has just emitted.'

# --- why feeding the target degenerates the objective
Set-Text 'D1200' 'Why feeding the target as input would degenerate the objective'
Set-Bold 'D1200'
Set-Formula 'N1202' '="v"&SEQUENCE(F1015,1,1,1)'
Set-Right 'N1202:N1207'
Set-Text 'D1202' 'E^T e_target at i=1'
Set-Right 'D1202'
Set-Formula 'O1202' '=MMULT(TRANSPOSE(E1025#),MMULT(E1025#,IF(SEQUENCE(F1015,1)=O1019,1,0)))'
Set-Box 'O1202:O1207'
Set-Text 'Q1202' 'The head is tied to the embedding, so a target fed as input scores'
Set-Text 'Q1203' 'itself through this dot product alone. Embedding columns are unit'
Set-Text 'Q1204' 'length, so the target index reads exactly 1.00 and every other'
Set-Text 'Q1205' 'index is a cosine below it, on every recalc. A module fed the'
Set-Text 'Q1206' 'target would learn the identity map and teach nothing. It is fed'
Set-Text 'Q1207' 't(i+1), one step before the target. See GATE_U.'

# ---------------------------------------------------------------- settle
$xl.Calculation = $xlCalculationManual
$xl.CalculateFullRebuild()
try { $xl.ActiveWindow.Zoom = 75 } catch {}
$ws.Range('A1').Select() | Out-Null

# ---------------------------------------------------------------- verify dump
# Read every block in one pass under manual calculation so RANDARRAY cannot
# reshuffle between reads.
function Dump-Range($addr) {
  $vals = $ws.Range($addr).Value2
  $rows = $ws.Range($addr).Rows.Count
  $cols = $ws.Range($addr).Columns.Count
  $matrix = @()
  for ($r = 1; $r -le $rows; $r++) {
    $line = @()
    for ($c = 1; $c -le $cols; $c++) {
      if ($rows -eq 1 -and $cols -eq 1) { $line += $vals }
      else { $line += $vals[$r, $c] }
    }
    $matrix += , $line
  }
  # Unary comma: without it PowerShell unrolls a single-row result and the
  # caller's [0] would index into the row instead of selecting it.
  return , $matrix
}

# Flatten a single-column range to a 1-D list. Section 12 writes its bias
# vectors as columns, but verify.py compares them as flat vectors.
function Dump-Col($addr) {
  $flat = @()
  foreach ($row in (Dump-Range $addr)) { $flat += $row[0] }
  return , $flat
}

$payload = [ordered]@{
  seq     = $ws.Range('F24').Value2
  d_model = $ws.Range('F25').Value2
  d_k     = $ws.Range('F26').Value2
  d_v     = $ws.Range('F27').Value2
  X       = Dump-Range 'O27:T34'
  Wq      = Dump-Range 'E38:L41'
  Q       = Dump-Range 'O38:T41'
  Wk      = Dump-Range 'E44:L47'
  K       = Dump-Range 'O44:T47'
  Wv      = Dump-Range 'E50:L52'
  V       = Dump-Range 'O50:T52'
  KT      = Dump-Range 'W37:Z42'
  D       = Dump-Range 'AB37:AG42'
  S       = Dump-Range 'AI37:AN42'
  A       = Dump-Range 'AP37:AU42'
  O       = Dump-Range 'AW45:BB47'
  labels  = @{
    x = (Dump-Range 'O26:T26')[0]
    q = (Dump-Range 'O37:T37')[0]
    k = (Dump-Range 'O43:T43')[0]
    v = (Dump-Range 'O49:T49')[0]
    o = (Dump-Range 'AW44:BB44')[0]
    toc = $ws.Range('D5').Text
  }
  s2 = [ordered]@{
    seq     = $ws.Range('F63').Value2
    d_model = $ws.Range('F64').Value2
    d_k     = $ws.Range('F65').Value2
    d_v     = $ws.Range('F66').Value2
    X   = Dump-Range 'O66:T73'
    Wq  = Dump-Range 'E77:L80'
    Q   = Dump-Range 'O77:T80'
    Wk  = Dump-Range 'E83:L86'
    K   = Dump-Range 'O83:T86'
    Wv  = Dump-Range 'E89:L91'
    V   = Dump-Range 'O89:T91'
    KT  = Dump-Range 'W76:Z81'
    D   = Dump-Range 'AB76:AG81'
    S   = Dump-Range 'AI76:AN81'
    M   = Dump-Range 'AP76:AU81'
    Smask = Dump-Range 'AW76:BB81'
    A   = Dump-Range 'BD76:BI81'
    O   = Dump-Range 'BK76:BP78'
    ledger_t      = (Dump-Range 'O95:T95')[0]
    ledger_k      = (Dump-Range 'O96:T96')[0]
    ledger_v      = (Dump-Range 'O97:T97')[0]
    ledger_cached = (Dump-Range 'O98:T98')[0]
    per_token = $ws.Range('F100').Value2
    context   = $ws.Range('F101').Value2
    at_context = $ws.Range('F102').Value2
    toc = $ws.Range('D6').Text
  }
  s3 = [ordered]@{
    seq          = $ws.Range('F111').Value2
    d_model      = $ws.Range('F112').Value2
    d_compressed = $ws.Range('F113').Value2
    X      = Dump-Range 'O114:T121'
    W_down = Dump-Range 'E124:L126'
    c      = Dump-Range 'O124:T126'
    W_up   = Dump-Range 'E130:G137'
    x_hat  = Dump-Range 'O130:T137'
    err    = Dump-Range 'W114:AB121'
    toc    = $ws.Range('D7').Text
  }
  s4 = [ordered]@{
    seq          = $ws.Range('F148').Value2
    d_model      = $ws.Range('F149').Value2
    kv_lora_rank = $ws.Range('F150').Value2
    n_heads      = $ws.Range('F151').Value2
    head_dim     = $ws.Range('F152').Value2
    X      = Dump-Range 'O151:T158'
    W_K    = Dump-Range 'E161:L166'
    K_mha  = Dump-Range 'O161:T166'
    W_V    = Dump-Range 'E168:L173'
    V_mha  = Dump-Range 'O168:T173'
    W_DKV  = Dump-Range 'W161:AD163'
    c_KV   = Dump-Range 'AG161:AL163'
    W_UK   = Dump-Range 'W168:Y173'
    K_mla  = Dump-Range 'AG168:AL173'
    W_UV   = Dump-Range 'W175:Y180'
    V_mla  = Dump-Range 'AG175:AL180'
    mha_per_token = $ws.Range('F186').Value2
    mla_per_token = $ws.Range('F187').Value2
    toc    = $ws.Range('D8').Text
  }
  s5 = [ordered]@{
    seq          = $ws.Range('F198').Value2
    d_model      = $ws.Range('F199').Value2
    kv_lora_rank = $ws.Range('F200').Value2
    n_heads      = $ws.Range('F201').Value2
    head_dim     = $ws.Range('F202').Value2
    rope_dim     = $ws.Range('F203').Value2
    X      = Dump-Range 'O201:T208'
    W_DKV  = Dump-Range 'E211:L213'
    c_KV   = Dump-Range 'O211:T213'
    W_UK   = Dump-Range 'E217:G222'
    K      = Dump-Range 'O217:T222'
    theta  = (Dump-Range 'W224:AB224')[0]
    rope_K = Dump-Range 'O225:T230'
    rope_c = Dump-Range 'O234:T236'
    WUK_rope_c = Dump-Range 'O238:T243'
    cell_a = $ws.Range('AA196').Value2
    cell_b = $ws.Range('AE196').Value2
    toc    = $ws.Range('D9').Text
  }
  s6 = [ordered]@{
    seq          = $ws.Range('F253').Value2
    d_model      = $ws.Range('F254').Value2
    q_lora_rank  = $ws.Range('F255').Value2
    kv_lora_rank = $ws.Range('F256').Value2
    n_heads      = $ws.Range('F257').Value2
    qk_nope      = $ws.Range('F258').Value2
    qk_rope      = $ws.Range('F259').Value2
    X      = Dump-Range 'O256:T263'
    W_DQ   = Dump-Range 'E266:L269'
    c_Q    = Dump-Range 'O266:T269'
    W_UQ1  = Dump-Range 'E271:H273'
    q_c1   = Dump-Range 'O271:T273'
    W_UQ2  = Dump-Range 'E275:H277'
    q_c2   = Dump-Range 'O275:T277'
    W_QR1  = Dump-Range 'E279:H280'
    q_r1   = Dump-Range 'O279:T280'
    W_QR2  = Dump-Range 'E282:H283'
    q_r2   = Dump-Range 'O282:T283'
    W_DKV  = Dump-Range 'W266:AD268'
    c_KV   = Dump-Range 'AG266:AL268'
    W_UK1  = Dump-Range 'W271:Y273'
    k_c1   = Dump-Range 'AG271:AL273'
    W_UK2  = Dump-Range 'W275:Y277'
    k_c2   = Dump-Range 'AG275:AL277'
    W_KR   = Dump-Range 'W279:AD280'
    k_r    = Dump-Range 'AG279:AL280'
    theta  = (Dump-Range 'W286:AB286')[0]
    rope_qr = Dump-Range 'O286:T287'
    rope_kr = Dump-Range 'AG286:AL287'
    S_c    = Dump-Range 'O313:T318'
    S_r    = Dump-Range 'O320:T325'
    S      = Dump-Range 'O327:T332'
    toc    = $ws.Range('D10').Text
  }
  s7 = [ordered]@{
    seq          = $ws.Range('F343').Value2
    d_model      = $ws.Range('F344').Value2
    q_lora_rank  = $ws.Range('F345').Value2
    kv_lora_rank = $ws.Range('F346').Value2
    n_heads      = $ws.Range('F347').Value2
    qk_nope      = $ws.Range('F348').Value2
    qk_rope      = $ws.Range('F349').Value2
    v_head       = $ws.Range('F350').Value2
    X      = Dump-Range 'O346:T353'
    W_DQ   = Dump-Range 'E356:L359'
    c_Q    = Dump-Range 'O356:T359'
    W_UQ1  = Dump-Range 'E361:H363'
    q_c1   = Dump-Range 'O361:T363'
    W_QR1  = Dump-Range 'E369:H370'
    q_r1   = Dump-Range 'O369:T370'
    W_DKV  = Dump-Range 'W361:AD363'
    c_KV   = Dump-Range 'AG361:AL363'
    W_KR   = Dump-Range 'W368:X369'
    k_r    = Dump-Range 'AG368:AL369'
    theta  = (Dump-Range 'W370:AB370')[0]
    rope_qr = Dump-Range 'O376:T377'
    rope_kr = Dump-Range 'AG373:AL374'
    W_UK1  = Dump-Range 'E381:G383'
    k_c1   = Dump-Range 'O381:T383'
    W_UV1  = Dump-Range 'E385:G387'
    v1     = Dump-Range 'O385:T387'
    S_A    = Dump-Range 'O412:T417'
    A_A    = Dump-Range 'O419:T424'
    O_A    = Dump-Range 'O426:T428'
    qp_c   = Dump-Range 'W385:Y390'
    S_B    = Dump-Range 'W412:AB417'
    A_B    = Dump-Range 'W419:AB424'
    cKV_A  = Dump-Range 'AD419:AI421'
    O_B    = Dump-Range 'W426:AB428'
    mha_per_token = $ws.Range('O436').Value2
    gqa_per_token = $ws.Range('O437').Value2
    mla_per_token = $ws.Range('O438').Value2
    mla_ratio     = $ws.Range('O440').Value2
    toc    = $ws.Range('D11').Text
  }
  s8 = [ordered]@{
    seq           = $ws.Range('F458').Value2
    d_model       = $ws.Range('F459').Value2
    expert_hidden = $ws.Range('F460').Value2
    n_experts     = $ws.Range('F461').Value2
    top_k         = $ws.Range('F462').Value2
    X       = Dump-Range 'O461:T468'
    W_gate  = Dump-Range 'E471:L476'
    g       = Dump-Range 'O471:T476'
    W_value = Dump-Range 'E478:L483'
    v       = Dump-Range 'O478:T483'
    h       = Dump-Range 'O485:T490'
    W_out   = Dump-Range 'E492:J499'
    out     = Dump-Range 'O492:T499'
    W_router = Dump-Range 'W471:AD478'
    logits  = Dump-Range 'AG471:AL478'
    A       = Dump-Range 'W480:AB487'
    argmax  = (Dump-Range 'W489:AB489')[0]
    toc     = $ws.Range('D12').Text
  }
  s9 = [ordered]@{
    seq           = $ws.Range('F513').Value2
    d_model       = $ws.Range('F514').Value2
    expert_hidden = $ws.Range('F515').Value2
    n_experts     = $ws.Range('F516').Value2
    top_k         = $ws.Range('F517').Value2
    n_shared      = $ws.Range('F518').Value2
    X       = Dump-Range 'O516:T523'
    W_router = Dump-Range 'E526:L533'
    logits  = Dump-Range 'O526:T533'
    A       = Dump-Range 'O535:T542'
    sel1    = (Dump-Range 'O544:T544')[0]
    sel2    = (Dump-Range 'O545:T545')[0]
    w1      = (Dump-Range 'O547:T547')[0]
    w2      = (Dump-Range 'O548:T548')[0]
    W_sg    = Dump-Range 'E551:L553'
    W_sv    = Dump-Range 'E555:L557'
    W_so    = Dump-Range 'E559:G566'
    shared_out = Dump-Range 'O559:T566'
    toc     = $ws.Range('D13').Text
  }
  s10 = [ordered]@{
    seq          = $ws.Range('F583').Value2
    d_model      = $ws.Range('F584').Value2
    n_experts    = $ws.Range('F585').Value2
    top_k        = $ws.Range('F586').Value2
    route_scale  = $ws.Range('F587').Value2
    X       = Dump-Range 'O586:T593'
    W_router = Dump-Range 'E596:L603'
    logits  = Dump-Range 'O596:T603'
    affinity = Dump-Range 'O605:T612'
    sel1    = (Dump-Range 'O614:T614')[0]
    sel2    = (Dump-Range 'O615:T615')[0]
    w1      = (Dump-Range 'O617:T617')[0]
    w2      = (Dump-Range 'O618:T618')[0]
    w_scaled = Dump-Range 'O620:T621'
    toc     = $ws.Range('D14').Text
  }
  s11 = [ordered]@{
    n_tokens  = $ws.Range('F635').Value2
    d_model   = $ws.Range('F636').Value2
    n_experts = $ws.Range('F637').Value2
    top_k     = $ws.Range('F638').Value2
    X      = Dump-Range 'O636:AL643'
    W_router = Dump-Range 'E662:L669'
    logits = Dump-Range 'O662:AL669'
    bias   = Dump-Range 'O683:AL690'
    score  = Dump-Range 'O692:AL699'
    sel1   = (Dump-Range 'O714:AL714')[0]
    load   = (Dump-Range 'O716:V716')[0]
    entropy = $ws.Range('O718').Value2
    toc    = $ws.Range('D15').Text
  }
  s12 = [ordered]@{
    n_tokens  = $ws.Range('F733').Value2
    d_model   = $ws.Range('F734').Value2
    n_experts = $ws.Range('F735').Value2
    top_k     = $ws.Range('F736').Value2
    update_rate = $ws.Range('F737').Value2
    advantage_value = $ws.Range('F738').Value2
    X      = Dump-Range 'O734:AL741'
    W_router = Dump-Range 'E744:L751'
    advantage = Dump-Range 'AO744:BL751'
    logits = Dump-Range 'O744:AL751'
    # These addresses tracked an earlier draft that laid the bias out as a row
    # in O. The builder writes it as a column in E, so every one of them read
    # empty cells and the dump handed verify.py nulls.
    b0     = Dump-Col 'E753:E760'
    count0 = (Dump-Range 'O773:V773')[0]
    b1     = Dump-Col 'E775:E782'
    count1 = (Dump-Range 'O795:V795')[0]
    b2     = Dump-Col 'E797:E804'
    count2 = (Dump-Range 'O817:V817')[0]
    b3     = Dump-Col 'E819:E826'
    w_diff = Dump-Range 'O831:V838'
    toc    = $ws.Range('D16').Text
  }
  s13 = [ordered]@{
    seq           = $ws.Range('F863').Value2
    d_model       = $ws.Range('F864').Value2
    d_k           = $ws.Range('F865').Value2
    expert_hidden = $ws.Range('F866').Value2
    n_experts     = $ws.Range('F867').Value2
    top_k         = $ws.Range('F868').Value2
    rms_eps       = $ws.Range('F869').Value2
    balance_coef  = $ws.Range('F870').Value2
    route_scale   = $ws.Range('F871').Value2
    X        = Dump-Range 'O866:T873'
    rms      = (Dump-Range 'O876:T876')[0]
    gamma    = Dump-Col  'E879:E886'
    n        = Dump-Range 'O879:T886'
    Wq       = Dump-Range 'E890:L893'
    Q        = Dump-Range 'O890:T893'
    Wk       = Dump-Range 'E895:L898'
    K        = Dump-Range 'O895:T898'
    Wv       = Dump-Range 'E900:L903'
    V        = Dump-Range 'O900:T903'
    KT       = Dump-Range 'W890:Z895'
    S        = Dump-Range 'AB890:AG895'
    A        = Dump-Range 'AI890:AN895'
    O_attn   = Dump-Range 'AP890:AU893'
    W_O      = Dump-Range 'E905:H912'
    attn_out = Dump-Range 'O905:T912'
    x1       = Dump-Range 'AW890:BB897'
    gamma2   = Dump-Col  'E916:E923'
    rms2     = (Dump-Range 'O916:T916')[0]
    n2       = Dump-Range 'O918:T925'
    W_gate   = Dump-Range 'E927:L932'
    g        = Dump-Range 'O927:T932'
    W_value  = Dump-Range 'E934:L939'
    # not "v": PowerShell hash keys are case-insensitive and V is the
    # attention value matrix above.
    ffn_v    = Dump-Range 'O934:T939'
    W_out    = Dump-Range 'E941:J948'
    ffn_out  = Dump-Range 'O941:T948'
    y        = Dump-Range 'W941:AB948'
    W_router = Dump-Range 'E952:L959'
    logits   = Dump-Range 'O952:T959'
    affinity = Dump-Range 'O961:T968'
    sel1     = (Dump-Range 'O970:T970')[0]
    sel2     = (Dump-Range 'O971:T971')[0]
    w1       = (Dump-Range 'O973:T973')[0]
    w2       = (Dump-Range 'O974:T974')[0]
    count    = (Dump-Range 'O978:V978')[0]
    freq     = (Dump-Range 'O979:V979')[0]
    meanaff  = (Dump-Range 'O980:V980')[0]
    fP       = (Dump-Range 'O981:V981')[0]
    balance_loss = $ws.Range('O983').Value2
    p_attention  = $ws.Range('P990').Value2
    p_expert     = $ws.Range('P991').Value2
    p_moe_total  = $ws.Range('P992').Value2
    p_moe_active = $ws.Range('P993').Value2
    p_block_total  = $ws.Range('P994').Value2
    p_block_active = $ws.Range('P995').Value2
    p_active_frac  = $ws.Range('P996').Value2
    toc      = $ws.Range('D17').Text
  }
  s14 = [ordered]@{
    seq        = $ws.Range('F1013').Value2
    d_model    = $ws.Range('F1014').Value2
    vocab      = $ws.Range('F1015').Value2
    horizon    = $ws.Range('F1016').Value2
    aligned    = $ws.Range('F1017').Value2
    d_k        = $ws.Range('F1018').Value2
    expert_hidden = $ws.Range('F1019').Value2
    rms_eps    = $ws.Range('F1020').Value2
    lambda     = $ws.Range('F1021').Value2
    lambda_final = $ws.Range('F1022').Value2
    decay_fraction = $ws.Range('F1023').Value2
    tokens     = (Dump-Range 'O1014:T1014')[0]
    input_tok  = (Dump-Range 'O1018:R1018')[0]
    target_tok = (Dump-Range 'O1019:R1019')[0]
    E          = Dump-Range 'E1025:J1032'
    emb        = Dump-Range 'O1025:R1032'
    h_full     = Dump-Range 'O1034:T1041'
    h_aligned  = Dump-Range 'O1043:R1050'
    gamma_h    = Dump-Col  'E1053:E1060'
    rms_h      = (Dump-Range 'O1053:R1053')[0]
    gamma_e    = Dump-Col  'E1062:E1069'
    rms_e      = (Dump-Range 'O1062:R1062')[0]
    concat     = Dump-Range 'AB1055:AE1070'
    M          = Dump-Range 'E1072:T1079'
    merged     = Dump-Range 'AG1073:AJ1080'
    gamma3     = Dump-Col  'E1084:E1091'
    rms3       = (Dump-Range 'O1084:R1084')[0]
    n3         = Dump-Range 'O1086:R1093'
    Wq         = Dump-Range 'E1095:L1098'
    Q          = Dump-Range 'O1095:R1098'
    Wk         = Dump-Range 'E1100:L1103'
    K          = Dump-Range 'O1100:R1103'
    Wv         = Dump-Range 'E1105:L1108'
    Vmat       = Dump-Range 'O1105:R1108'
    KT         = Dump-Range 'W1095:Z1098'
    S          = Dump-Range 'AB1095:AE1098'
    A          = Dump-Range 'AG1095:AJ1098'
    O_attn     = Dump-Range 'AL1095:AO1098'
    W_O        = Dump-Range 'E1110:H1117'
    attn_out   = Dump-Range 'O1110:R1117'
    r1         = Dump-Range 'AL1110:AO1117'
    gamma4     = Dump-Col  'E1120:E1127'
    rms4       = (Dump-Range 'O1120:R1120')[0]
    n4         = Dump-Range 'O1122:R1129'
    W_gate     = Dump-Range 'E1131:L1136'
    g          = Dump-Range 'O1131:R1136'
    W_value    = Dump-Range 'E1138:L1143'
    ffn_v      = Dump-Range 'O1138:R1143'
    W_out      = Dump-Range 'E1145:J1152'
    ffn_out    = Dump-Range 'O1145:R1152'
    refined    = Dump-Range 'V1145:Y1152'
    gamma5     = Dump-Col  'E1156:E1163'
    rms5       = (Dump-Range 'O1156:R1156')[0]
    nf         = Dump-Range 'O1158:R1165'
    logits     = Dump-Range 'O1167:R1172'
    probs      = Dump-Range 'O1174:R1179'
    p_target   = (Dump-Range 'O1181:R1181')[0]
    loss_pos   = (Dump-Range 'O1182:R1182')[0]
    mtp_loss   = $ws.Range('O1184').Value2
    main_loss  = $ws.Range('P1186').Value2
    combined   = $ws.Range('P1187').Value2
    lambda_50  = $ws.Range('P1189').Value2
    lambda_80  = $ws.Range('P1190').Value2
    proposal   = (Dump-Range 'O1194:R1194')[0]
    actual     = (Dump-Range 'O1195:R1195')[0]
    accepted   = (Dump-Range 'O1196:R1196')[0]
    accept_rate = $ws.Range('O1197').Value2
    tied_dots  = Dump-Col 'O1202:O1207'
    toc        = $ws.Range('D18').Text
  }
}
$payload | ConvertTo-Json -Depth 6 | Set-Content -Path $dump -Encoding UTF8

# Any cell showing an Excel error anywhere on the sheet
$errAddr = ''
try {
  $errCells = $ws.UsedRange.SpecialCells(-4123, 16)
  $errAddr = $errCells.Address($false, $false)
} catch { $errAddr = '(none)' }
Write-Host "ERROR CELLS: $errAddr"

$dupes = $script:written.Keys | Where-Object { $script:written[$_] -gt 1 } | Sort-Object
if ($dupes) { Write-Host "DUPLICATE CELL WRITES: $($dupes -join ', ')" }
else { Write-Host "DUPLICATE CELL WRITES: (none)" }

# ---------------------------------------------------------------- save
if (Test-Path $out) { Remove-Item $out -Force }
$wb.SaveAs($out, 51)   # 51 = xlOpenXMLWorkbook
# Calculation is an Application property but needs an open workbook, so it has
# to go back to automatic before the last one closes.
$xl.Calculation = $xlCalculationAutomatic
$wb.Close($false)
$xl.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($ws) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($wb) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($xl) | Out-Null
Write-Host "wrote $out"
Write-Host "wrote $dump"
