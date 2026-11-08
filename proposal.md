# TITLE
<Project title> — <Name 1> (andrewid1), <Name 2> (andrewid2)

> Teams must be exactly two students.

# URL
<https://...>  <!-- project web page (GitHub Pages or Andrew UserWeb) -->

# SUMMARY
<2–3 sentences: what you will build/study + which parallel systems (e.g., GHC CPU cores, NVIDIA GPU, PSC machines).>

# BACKGROUND
<A few paragraphs on the application/piece you will implement (e.g., DLRM inference: embedding lookups + bottom/top MLP + feature interaction).>

- Block diagram or pseudocode of the basic idea: <figure>
- Which parts could benefit from parallelism, and why: <...>

# THE CHALLENGE
**Workload**
- Dependencies: <...>
- Memory access characteristics (locality? communication-to-computation ratio?): <...>
- Divergent execution: <...>

**Constraints** (properties of the system that make mapping the workload hard)
- <e.g., CPU–GPU transfer bandwidth, GPU memory capacity vs. embedding table size, ...>

**What we hope to learn:** <...>

# RESOURCES
- Machines: <GHC / PSC / EC2 / other>
- Starter code / code base (or from scratch): <...>
- References (papers, books, with citations): <...>
- Resources we still need / special machines: <...>

# GOALS AND DELIVERABLES
**PLAN TO ACHIEVE** (must-haves for a successful project)
1. <goal + justification, e.g., "X speedup over baseline because ...">
2. <...>

**HOPE TO ACHIEVE** (if ahead of schedule)
1. <...>

**FALLBACK** (if work goes slower than expected)
1. <...>

**Live demo (optional):** <what you'd show at the poster session, or "none">

**Analysis questions** (if an analysis project): <what do you want to learn about the workload/system?>

**System capability / target performance** (if a systems project): <...>

# PLATFORM CHOICE
<Why this computer/language (e.g., CUDA + multicore CPU) fits this workload.>

# SCHEDULE
At least one item per week; account for busier weeks. Milestone report due Mon Nov 23.

| Week | Dates | Tasks |
|------|-------|-------|
| 1 | Nov 9 – Nov 15 | <...> |
| 2 | Nov 16 – Nov 22 | <... (finish before milestone Nov 23)> |
| 3 | Nov 23 – Nov 29 | <...> |
| 4 | Nov 30 – Dec 6 | <...> |
| 5 | Dec 7 – Dec 10 | <final report due Dec 10, 11:59pm> |
| — | Dec 11 | Poster session |
