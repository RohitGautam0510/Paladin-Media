# PaladinMedia — Workspace Agent Rules

## Engineering Reliability Runtime (Gravitas Core)

For engineering, implementation, testing, debugging, review, and audit tasks in this repository, read and follow the installed Gravitas Core skill:
- **Skill Path:** [.agents/skills/gravitas/SKILL.md](.agents/skills/gravitas/SKILL.md)

Follow the core Gravitas invariants:
1. **Ground before mutating:** Inspect targets, callers, and tests before making modifications. Never act on assumed file state.
2. **Stay in declared scope:** Limit modifications strictly to the task's authorized scope.
3. **Evidence-backed completion:** Verify changes with freshly observed execution output (`VERDICT: PASS` or `VERDICT: FAIL`). Self-review or stale runs do not count as completion evidence.
4. **Adaptive failure recovery:** Never exact-retry failed attempts without a material change in diagnosis, context, or strategy.
5. **Declare uncertainties:** Explicitly report unverified, unverifiable, or partially verified claims rather than assuming certainty.

## PaladinMedia Safety Boundaries & Invariant Constraints

Gravitas Core operates as an engineering reliability runtime and **does not** authorize actions restricted by project governance:
- **No Model Training:** Strictly no `optimizer.step()`, training epochs, loss weight tuning, or parameter weight updates unless explicitly authorized by the user.
- **Frozen Architecture:** Preserve the ACS-Sub-INN v3 design (`gamma_min=1.0`, transform-domain Red-HH target, Green/Blue-HH context).
- **Cryptographic & ECC Integrity:** Preserve the strict unidirectional pipeline (`SecretPayload` -> `SecretSerializer` -> `AESGCMProvider` -> `ReedSolomonEncoder`), capacity gates, and fail-closed security properties.
- **Dataset Preservation:** Do not modify, regenerate, delete, or repartition DIV2K images or the canonical split manifest (`docs/project/div2k_hr_split.json`).
- **Environment & Scope Stability:** Do not install or upgrade packages, perform destructive operations, or execute out-of-scope refactoring without explicit authorization.
