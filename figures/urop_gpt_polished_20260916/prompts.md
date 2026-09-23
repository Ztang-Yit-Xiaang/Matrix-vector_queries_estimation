# GPT image polish: prompt record

Date: 2026-09-16. Tool: built-in GPT image tool; one edit call per original figure. Original quantitative figures remain authoritative and unchanged. No local image-editing or raster-compositing script was used. Generated PNGs are separate presentation editions, not editable vector replacements.

## figure1_capture_failure

Edit target: ../urop_validated_20260916/figure1_capture_failure.png

Use case: style-transfer. This is an edit of the attached scientific figure, NOT permission to invent data. Produce a polished, high-resolution landscape scientific report graphic. Elegant Swiss editorial typography, white background, deep ink headings, desaturated blue #31688E, burnt amber #C7752C, cool grey; generous margins, fine rules, strong panel hierarchy, no gradients, no 3D, no decorative icons. Keep every quantitative panel, data point, axis scale, curve, confidence interval, and scientific distinction from the source. Improve layout, typography and spacing. Avoid tiny text. Use a landscape canvas about 2:1, at least 2400 pixels wide if possible. Retain source plots with extremely high fidelity. All labels sharp and spelled correctly. Bottom small readable label: "Presentation edition • source-backed data; see validated original". No logo, no fake journal branding.
Title: "Full numerical rank can miss a signal direction". Subtitle: "Coordinate-aligned step spectrum • d = 100 • m = 60 • five leading eigenvalues • tail = 0.001".
Three panels with labels a, b, c. a: "Conditional residual risk" preserving every point and log y-axis; blue hollow circles Standard, amber diamonds Gated, 10 trials indexed0–9. Gated trial7risk0.02272744518, other9about1.31e-7; Standardabout1.255e-6; keep actual original points. Arrow callout at trial7 "Missed signal direction". b: "A sharp but incomplete Ritz knee". Reference trial2 grey dashed shows5 near1 then3 near.001. Failure trial7 amber solid shows4 near1 then4near.001. Index1–8, logy. Move legend to avoid crossing data. c:"Exact signal-sketch witness". Preserve the exact5x8signmatrix fromsource. Rows in order:
− − − + + − + +
− + + − − − + −
+ + − − + + − −
− − + + − − + +
− + − + − + − −
Thin amber outline across rows3and4 only; label "Rows 3 and 4 are exact opposites".
Bottom takeaway band: "Accepted basis rank = 8; captured signal rank = 4 of 5."
Footnote: "Gated: q = r = 8, ℓ = 44. Standard: q = r = 20, ℓ = 20. Exact conditional risks; 10 original coordinate paths."
Do not call matrix rankfive: positive tail means fullrank. Do not claim an estimated population failure rate. Keep allmatrixsigns exact.

## figure2_empirical_signal

Edit target: ../urop_validated_20260916/figure2_empirical_signal.png

Use case: style-transfer. This is an edit of the attached scientific figure, NOT permission to invent data. Produce a polished, high-resolution landscape scientific report graphic. Elegant Swiss editorial typography, white background, deep ink headings, desaturated blue #31688E, burnt amber #C7752C, cool grey; generous margins, fine rules, strong panel hierarchy, no gradients, no 3D, no decorative icons. Keep every quantitative panel, data point, axis scale, curve, confidence interval, and scientific distinction from the source. Improve layout, typography and spacing. Avoid tiny text. Use a landscape canvas about 2:1, at least 2400 pixels wide if possible. Retain source plots with extremely high fidelity. All labels sharp and spelled correctly. Bottom small readable label: "Presentation edition • source-backed data; see validated original". No logo, no fake journal branding.
Title: "Fresh probes reveal useful risk information". Subtitle: "Phase 1A • s = 16 • m = 160 • tail = 10⁻⁶ • candidate q = r* + 1 versus q = r*".
Keep two side-by-side forestplotpanels with alignedthree rows: "Sample variance", "MoM: 1 pair/block", "MoM: 2 pairs/block". Bluecircle,amberhollowsquare,amberdiamond.
Panel a "False-safe acceptance"; log xaxispercentage withticks0.001,0.01,0.1,1,10. Points and95%intervals in PERCENT: SV0.013435[0.002767,0.030509]; MoM1 4.697140[4.548594,4.851792]; MoM2 1.547764[1.451309,1.650362]. Vertical dashedthreshold0.5%,label"0.5% ceiling". Do not treatthosepercentages asfractions.
Panel b "Catastrophic-path detection"; linear xaxis0,25,50,75,100percent. SV79.6667[69.06625,89.28333]; MoM1 72[63.08333,80.83333]; MoM2 75.6[66.26625,84.46708]. Verticaldashedline75%,label"75% target".
Axis labels "Accepted among truly worse paths (%)" and "Eligible catastrophic paths accepted (%)".
Addsmallneatrowpointlabels if space without obscuringintervals. No gold stars orfalseproofbadges.
Footer: "600 frozen paths • 200 certification repetitions per path • equal-rank path-averaged rates".
Secondfooter: "Bars: original conditional 95% bootstrap intervals. Two selected criteria, not the complete gate."
Takeaway band: "Empirical signal—not a confidence certificate."
Preserve actual intervalpositions and widths carefully. No error bar zero or log originzero.

## figure3_certification_cost

Edit target: ../urop_validated_20260916/figure3_certification_cost.png

Use case: style-transfer. This is an edit of the attached scientific figure, NOT permission to invent data. Produce a polished, high-resolution landscape scientific report graphic. Elegant Swiss editorial typography, white background, deep ink headings, desaturated blue #31688E, burnt amber #C7752C, cool grey; generous margins, fine rules, strong panel hierarchy, no gradients, no 3D, no decorative icons. Keep every quantitative panel, data point, axis scale, curve, confidence interval, and scientific distinction from the source. Improve layout, typography and spacing. Avoid tiny text. Use a landscape canvas about 2:1, at least 2400 pixels wide if possible. Retain source plots with extremely high fidelity. All labels sharp and spelled correctly. Bottom small readable label: "Presentation edition • source-backed data; see validated original". No logo, no fake journal branding.
Title: "Lower mean risk does not mean every path benefits". Subtitle: "Phase 1B • sample variance • s = 16 • m = 160 • tail = 10⁻⁶ • 600 frozen paths".
Two panels preserve original quantitative plots exactly.
Panel a "Aggregate risk comparison", horizontal forestplot, log xaxis ticks0.01,0.1,1; referenceverticaldashedat1. Three rows: "Paid fallback" greysquare point1.172196706 zero-widthdisplayinterval; "Paid oracle" bluecircle point0.035410719 with95%CI[0.014256889,0.628404049]; "Empirical selection" amberdiamondpoint0.036329116CI[0.014627614,0.639830084]. Displayvalues1.1722,.0354,.0363. Axislabel"Equal-rank mean of mean-risk ratios".
Panel b "Pathwise net effects", preserve exact empirical stepCDF fromreference, log xaxisrange5e-8to4 ticks1e-7,1e-5,1e-3,.1,1, yaxis0–100percent. Nearlyallmass rises sharplyjustabove1, notlinearslope. Retain the entire tiny-ratio lefttail. Annotate "575 / 600 paths harmed" and "95.83%" unobtrusively. xaxis"Pathwise selected/original risk ratio (log)";yaxis"Cumulative fraction of paths (%)". Do not redraw the curve fromimagineddata: retain sourcecurvefaithfully.
Bottom takeaway band: "Rare-failure protection can improve the mean while most paths pay a cost."
Footnote: "Ratio 1 = original unstarted adjacent baseline. Costs are charged."
Footnote: "a: original conditional 95% bootstrap intervals. b: empirical distribution, no confidence band. Different estimands."
No claimofuniversalsafety, no newcurves.

