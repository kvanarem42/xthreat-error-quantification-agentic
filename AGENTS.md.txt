# AGENTS.md

## Purpose

This repository contains an existing research-oriented Python codebase with Jupyter notebooks and standalone scripts. The goal of this Codex-driven restructuring effort is to transform the repository into a more maintainable, user-friendly, package-oriented project structure while preserving all original functionality and execution logic.

This document defines the mandatory workflow, constraints, review process, and approval gates for all repository restructuring work.

---

# Core Principles

1. Preserve all original functionality.
2. Never silently delete or rewrite behavior.
3. Maintain execution order and workflow clarity.
4. Prefer conservative, minimal changes.
5. Separate analysis from modification.
6. Require explicit user approval before making changes.
7. Clearly map old files to new locations.
8. Preserve reproducibility of notebooks and scripts.
9. Treat notebooks as first-class project artifacts.
10. Highlight redundancy, but do not remove redundant code unless explicitly approved.

---

# Repository Goals

The target repository should:

- Be easier to navigate for new users.
- Have a clear package/module structure where possible.
- Separate reusable code from experiments/notebooks.
- Make execution order and dependencies explicit.
- Support future maintainability and testing.
- Retain all original code unless explicitly approved otherwise.

Preferred characteristics:

- `src/`-style package layout when feasible
- clear notebook organization
- centralized configuration
- explicit data/workflow stages
- reduced ambiguity between scripts and library code

---

# Strict Phase-Gated Workflow

The restructuring process MUST follow the phases below exactly.

No phase may be skipped.

No modifications may occur before approval.

---

# Phase 1 — Repository Analysis ONLY

## Objectives

Analyze the repository without making changes.

Identify:

- current file structure
- workflow order
- execution dependencies
- duplicated logic
- notebook/script relationships
- reusable modules
- problematic structure patterns
- ambiguous naming
- dead code candidates
- configuration handling
- data flow
- entry points

## Deliverables

Produce:

1. Current repository structure summary
2. Proposed target structure
3. File mapping proposal
4. Workflow/dependency analysis
5. Risks and uncertainties
6. Redundancy report
7. Questions requiring clarification

## Rules

- DO NOT modify files.
- DO NOT rename files.
- DO NOT move files.
- DO NOT rewrite code.
- DO NOT remove redundancy.
- DO NOT infer intent without marking uncertainty.

## Mandatory Approval Gate

After analysis, STOP and ask the user:

- whether the proposed structure is correct
- which proposed changes are approved
- whether notebook handling is acceptable
- whether packaging changes are acceptable

No implementation may begin before explicit approval.

---

# Phase 2 — Structural Proposal Review

After user feedback:

## Objectives

Refine the restructuring plan.

Provide:

- exact directory tree
- exact file moves
- exact proposed renames
- package/module boundaries
- notebook placement
- import migration strategy
- execution order documentation plan
- compatibility concerns

## Requirements

For every original file, explicitly state:

- original location
- proposed new location
- whether content changes are required
- whether imports must change
- whether notebook references must change

## Rules

- DO NOT implement changes yet.
- DO NOT modify repository contents.
- Only propose changes.

## Mandatory Approval Gate

STOP again and request explicit approval for:

- file moves
- renames
- package restructuring
- import changes
- notebook conversions (if any)
- config restructuring
- workflow documentation changes

No edits may be made before approval.

---

# Phase 3 — Conservative Implementation

Only after explicit approval.

## Objectives

Implement approved structural changes conservatively.

## Allowed Actions

Only perform approved operations:

- move files
- rename files
- add package structure
- update imports
- add `__init__.py`
- add documentation
- reorganize notebooks
- add workflow documentation

## Rules

- Preserve behavior.
- Preserve execution semantics.
- Preserve notebook outputs unless requested otherwise.
- Do not refactor algorithms unless explicitly approved.
- Do not remove duplicate code.
- Do not optimize logic unless requested.
- Do not combine modules unless approved.

## Requirements

Every change must be:

- traceable
- minimal
- reversible
- documented

---

# Phase 4 — Post-Change Validation

After implementation:

## Mandatory Checks

### Workflow Validation

Verify:

- execution order still works
- notebooks still execute logically
- scripts still run
- imports resolve correctly
- package structure is consistent

### Dependency Validation

Check:

- internal imports
- circular dependencies
- notebook dependencies
- hidden relative-path assumptions
- configuration references
- data path assumptions

### Coverage Validation

Verify that:

- every original file still exists somewhere
- no code was accidentally omitted
- no notebook content was lost
- all workflows remain represented

Produce a complete old-to-new mapping table.

---

# Phase 5 — Redundancy Review ONLY

After restructuring is complete:

## Objectives

Identify redundant or duplicated code.

Examples:

- duplicated utility functions
- repeated notebook logic
- repeated preprocessing
- near-identical scripts
- obsolete wrappers
- dead helper modules

## Rules

- DO NOT modify redundant code.
- DO NOT delete anything.
- DO NOT consolidate automatically.

## Deliverables

Provide:

1. redundancy report
2. severity/practical impact
3. proposed consolidation strategy
4. risks of consolidation
5. recommended future cleanup plan

## Mandatory Approval Gate

Ask the user which redundancy changes, if any, are approved.

Only after explicit approval may redundancy reduction occur.

---

# Packaging Guidance

When feasible, prefer a structure similar to:

```text
repo/
├── pyproject.toml
├── README.md
├── src/
│   └── package_name/
│       ├── __init__.py
│       ├── pipelines/
│       ├── preprocessing/
│       ├── models/
│       ├── evaluation/
│       ├── utils/
│       └── configs/
├── notebooks/
│   ├── exploratory/
│   ├── experiments/
│   └── reports/
├── scripts/
├── tests/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
└── docs/

This is only a guideline.

The actual structure must follow the repository’s existing workflow and user approval.

# Notebook Handling Rules

Notebooks often contain important workflow logic.

Therefore:
- preserve notebook content
- preserve notebook execution order
- identify duplicated notebook code separately
- avoid aggressive notebook splitting
- avoid converting notebooks into scripts unless approved
- clearly document notebook purpose

If notebook logic should become reusable modules:
1. propose extraction first
2. explain rationale
3. request approval before implementation


# File Mapping Requirements

For every restructuring proposal, include a mapping table like:

Original File	Proposed Location	Action	Notes
train.py	src/package/pipelines/train.py	Move	imports updated
utils.ipynb	notebooks/experiments/utils.ipynb	Move	no content changes

No file may be omitted from the mapping.

# Change Safety Rules

The agent MUST:
- prefer minimal edits
- avoid speculative refactors
- avoid architectural rewrites
- preserve public APIs where possible
- preserve notebook reproducibility
- preserve script entry points unless approved otherwise

The agent MUST NOT:
- silently delete code
- silently merge files
- silently alter algorithms
- silently remove dependencies
- silently rewrite notebooks
- silently normalize style across the entire repository

# Communication Rules

The agent must clearly distinguish between:
- observations
- assumptions
- confirmed facts
- proposed changes
- approved changes
- implemented changes

The agent must explicitly label:
- uncertainties
- risks
- blocked decisions
- possible regressions

# Required Review Style

At every review gate:
1. Report findings first.
2. List uncertainties separately.
3. Ask for approval explicitly.
4. Wait for confirmation before modifying anything.

# Final Deliverables

At completion, provide:
1. final repository tree
2. old-to-new file mapping
3. dependency/workflow validation summary
4. unresolved concerns
5. redundancy report
6. recommended future improvements (non-applied)


# Success Criteria

The restructuring is successful only if:
- all original functionality remains accessible
- repository organization is clearer
- execution order is understandable
- reusable logic is easier to locate
- no code is lost
- all changes were explicitly approved
- redundancy was only reported unless approved for removal