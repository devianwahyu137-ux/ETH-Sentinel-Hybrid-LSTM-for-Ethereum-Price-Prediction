// JAI UMY SINTA 2 - Two-Column Publication Template for Typst
// Target: Journal of Accounting and Investment (JAI), Universitas Muhammadiyah Yogyakarta

#let jai_article(
  title: "",
  authors: "[Author Names Redacted for Double-Blind Peer Review]",
  target_journal: "Journal of Accounting and Investment (JAI - SINTA 2 Accredited)",
  classification: "Empirical Research Article (Computational Finance & Asset Pricing)",
  abstract: [],
  keywords: [],
  jel: [],
  body
) = [
  // Page setup: A4, 2.5cm top/bottom, 2.0cm left/right
  #set page(
    paper: "a4",
    margin: (top: 2.5cm, bottom: 2.5cm, left: 2.0cm, right: 2.0cm),
    header: context {
      set text(font: "Times New Roman", size: 8.5pt, fill: rgb("#444444"))
      grid(
        columns: (1fr, auto),
        align: (left + bottom, right + bottom),
        [*Journal of Accounting and Investment*, Vol. 27 No. 2 (2026): May–August | e-ISSN: 2622-3899],
        [#counter(page).display("1")]
      )
      v(-2pt)
      line(length: 100%, stroke: 0.5pt + rgb("#cccccc"))
    },
    footer: none
  )

  // Typography rules
  #set text(
    font: "Times New Roman",
    size: 10pt,
    lang: "en",
  )

  #set par(
    justify: true,
    leading: 0.65em,
    first-line-indent: 0.5cm,
  )

  // Suppress raw LaTeX table blocks
  #show raw.where(lang: "latex"): none

  // Float figures & tables across parent 2-column width
  #show figure: set figure(placement: auto, scope: "parent")
  #show figure.where(kind: table): set figure(supplement: [Table])
  #show figure.where(kind: image): set figure(supplement: [Figure])
  #show figure.where(kind: table): set figure.caption(position: top)
  #show figure.where(kind: image): set figure.caption(position: bottom)

  // Table styling (Booktabs open horizontal format)
  #show table: set table(
    stroke: (x, y) => if y == 0 {
      (top: 1pt + black, bottom: 0.5pt + black)
    } else {
      none
    },
    fill: none,
  )
  #show table: set text(size: 7.5pt, font: "Times New Roman")
  #show table: set par(leading: 0.45em, justify: false, first-line-indent: 0pt)
  #show table.cell: set table.cell(inset: 3.5pt)

  // Heading styles
  #show heading: it => {
    set text(font: "Times New Roman")
    if it.level == 1 {
      // Level 1: Article Title (Handled separately in banner, but if present:)
      set text(size: 13pt, weight: "bold")
      align(center)[#it.body]
      v(8pt, weak: true)
    } else if it.level == 2 {
      // Level 2: Main Sections (1. INTRODUCTION, 2. LITERATURE REVIEW, etc.)
      set text(size: 11pt, weight: "bold")
      v(14pt, weak: true)
      upper(it.body)
      v(6pt, weak: true)
    } else if it.level == 3 {
      // Level 3: Subsections (2.1., 3.1., etc.)
      set text(size: 10.5pt, weight: "bold")
      v(10pt, weak: true)
      it.body
      v(5pt, weak: true)
    } else if it.level == 4 {
      // Level 4: Sub-subsections
      set text(size: 10pt, weight: "bold", style: "italic")
      v(8pt, weak: true)
      it.body
      v(4pt, weak: true)
    } else {
      // Level 5: Panels / minor subheadings
      set text(size: 9.5pt, weight: "bold")
      v(6pt, weak: true)
      it.body
      v(3pt, weak: true)
    }
  }

  // --- TOP BANNER (Single Column) ---
  #align(center)[
    #v(4pt)
    #text(size: 14pt, weight: "bold")[#title]
    #v(8pt)
    #text(size: 10.5pt, style: "italic", fill: rgb("#222222"))[*Author(s):* #authors] \
    #text(size: 9pt, fill: rgb("#555555"))[*Target Journal:* #target_journal | *Classification:* #classification]
    #v(10pt)
  ]

  // --- ABSTRACT BOX (Single Column) ---
  #rect(
    width: 100%,
    fill: rgb("#f8f9fa"),
    stroke: (top: 0.8pt + rgb("#333333"), bottom: 0.8pt + rgb("#333333")),
    inset: (x: 14pt, y: 11pt),
    radius: 0pt,
  )[
    #set text(size: 9pt, font: "Times New Roman")
    #set par(justify: true, leading: 0.55em, first-line-indent: 0pt)
    #align(center)[#text(weight: "bold", size: 9.5pt)[ABSTRACT]]
    #v(4pt)
    #abstract
    #v(6pt)
    *Keywords:* #keywords \
    *JEL Classification:* #jel
  ]

  #v(14pt)

  // --- ARTICLE BODY (Two Columns) ---
  #columns(2, gutter: 18pt)[
    #body
  ]
]
