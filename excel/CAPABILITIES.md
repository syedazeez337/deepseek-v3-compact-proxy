# What this Excel can and cannot do

Probed directly on this machine on 2026-08-06 by writing each formula through
COM and reading the result back. Every formula in `build_sheet.ps1` stays
inside the supported column.

## Root cause

```
ProductReleaseIds : OneNoteFreeRetail,HomeStudent2021Retail
VersionToReport   : 16.0.20228.20124
UpdateChannel     : 492350f6-... (Current Channel)
```

Office Home & Student 2021, a perpetual licence. LAMBDA is not a missing
update, it is not licensed: Microsoft's own applies-to list for LAMBDA reads
"Excel for Microsoft 365, Excel for Microsoft 365 for Mac, Excel 2024, Excel
2024 for Mac". Excel 2021 is absent. Perpetual Office ships a frozen subset of
the Microsoft 365 feature catalogue and receives security fixes only, so
running Office updates will never add these functions.

A release-wave probe puts the boundary exactly at the Office 2021 baseline:

| Wave | Representative function | Result |
|---|---|---|
| 2019 | `TEXTJOIN`, `CONCAT` | works |
| 2021 | `SEQUENCE` `XLOOKUP` `LET` `FILTER` | works |
| 365, Mar 2022 | `LAMBDA` `BYCOL` `MAP` `REDUCE` `MAKEARRAY` | `#NAME?` |
| 365, Aug 2022 | `VSTACK` `TOCOL` `TEXTSPLIT` `TAKE` | `#NAME?` |
| 365, 2024 | `GROUPBY` | `#NAME?` |
| 365, 2025 | `REGEXTEST` | `#NAME?` |

Note that Office 2021 support ends 2026-10-13.

| Available | Missing (returns `#NAME?`) |
|---|---|
| `SEQUENCE` `RANDARRAY` `LET` `FILTER` `SORT` `SORTBY` `UNIQUE` | `LAMBDA` `BYCOL` `BYROW` `MAKEARRAY` |
| `MMULT` `TRANSPOSE` `MINVERSE` `SUMPRODUCT` | `TOCOL` `TOROW` |
| `XMATCH` `MATCH` `INDEX` `LARGE` | `VSTACK` `HSTACK` |
| `HYPERLINK` `UNICHAR` `TEXTJOIN` `REPT` `SUBSTITUTE` | `TAKE` `DROP` `CHOOSEROWS` `CHOOSECOLS` |
| Spill references (`E38#`), array broadcast in `IF` | |

`BYROW` is the one trap: it returns `0` rather than `#NAME?`, because the
function name resolves but its `LAMBDA` argument does not. Treat it as missing.

## Consequences for the build

- **Column-wise softmax** cannot use the one-formula
  `BYCOL(z, LAMBDA(c, SUM(EXP(c))))` form. Use one `LET` per column, which is
  what Tom Yeh's own section 1 does. Section 1 here writes six of them.
- **Causal masking** (section 2) does not need `MAKEARRAY`. Array broadcast
  works: `=IF(SEQUENCE(n,1)>=SEQUENCE(1,n),1,0)`. He uses the same trick in his
  chunk-wise section.
- **Top-k routing** (sections 8-12) has no native array top-k. `LARGE`,
  `MATCH`, `SORTBY` and `INDEX` are all present, so it is buildable, but it is
  the one formula worth prototyping before committing to a layout.
- **Never use** `TOCOL`, `VSTACK`, `HSTACK`, `TAKE`, `CHOOSEROWS` or
  `MAKEARRAY`, however convenient they look.

## The reference workbook does not fully open here

Opening `kimi3-release.xlsx` on this Excel leaves **87 formula cells showing
errors**, essentially all of the ShortConv section (`TOCOL`, `CHOOSEROWS`) plus
the two `BYCOL` softmax cells at `AV68` and `AV83`. His sheet was authored in
Excel Online, which has the full function set.

So the ShortConv section cannot be copied from his workbook formula for
formula. Ours has no ShortConv section, but the depth-wise convolution idea
recurs, and the sliding window he builds with `TOCOL` has to be rebuilt from
`INDEX` or explicit offset ranges.

## Reproducing the probe

`excel/build_sheet.ps1` fails loudly rather than silently: it prints
`ERROR CELLS:` with the address of every cell showing an Excel error, and
`excel/verify.py` recomputes the whole section independently. If a future
section uses an unsupported function, both will say so.
