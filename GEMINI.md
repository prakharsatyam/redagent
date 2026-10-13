# Workflow Rules & Operating Principles

## Scope Estimation & Chunked Execution Protocol

To maintain maximum code quality, prevent context exhaustion, and ensure full transparency, the agent must adhere to the following execution protocol for every user prompt:

---

### 1. Pre-Flight Scope Estimation
Before writing code or executing multi-step modifications, evaluate the task scale:
- **Files Affected**: How many files will be created, refactored, or inspected?
- **Complexity & Risk**: Does the task cross multiple system layers (e.g., config, database/data schema, core logic, API, tests)?
- **Context Footprint**: Will heavy file reading and diff generation consume significant context window headroom?

#### Threshold Criteria
A task is classified as **Large / Above Threshold** if any of the following are met:
- Modifies or creates **more than 3 files** with non-trivial logic.
- Requires major structural refactoring (>150 lines of complex business logic).
- Involves multiple distinct sequential stages (e.g., schema migration -> service layer -> endpoint -> integration tests).
- Poses a risk of context exhaustion or degraded response quality if executed in a single turn.

---

### 2. Execution Strategy

#### A. Below Threshold (Small to Medium Tasks)
- Proceed directly: outline a brief plan, implement changes, verify/test, and provide a concise summary.

#### B. Above Threshold (Large / Multi-Stage Tasks)
When a task exceeds the threshold, strictly execute in **chunked batches**:
1. **Roadmap & Batching**:
   - Decompose the request into 2 to 4 logical, self-contained batches (e.g., Batch 1: Setup/Data types, Batch 2: Core processing, Batch 3: Verification & Integration).
   - State why the task was broken into chunks and display the full high-level roadmap.
2. **Execute Only the Active Batch**:
   - Work **only** on the active batch (e.g., Batch 1).
   - Ensure the codebase remains in a valid, functional state at the end of the batch (no broken syntax or incomplete dangling imports).
   - Stop execution with enough context headroom preserved to generate the hand-off.
3. **Mandatory Hand-Off Structure**:
   Conclude the response with:
   - **Completed in this batch**: Specific files created/modified and tests run.
   - **Current System State**: Any key decisions, assumptions, or artifacts generated.
   - **Next Batch Scope**: Clear overview of what Batch 2 (or N) will accomplish.
   - **Next Prompt (Ready to Run)**: A dedicated, copy-pasteable prompt enclosed in a fenced code block for the user to submit in their next turn.

---

### 3. Next Prompt Template Specification
The provided prompt for the next turn must be self-contained so that even if context is compressed, the agent can resume immediately:
```markdown
```text
Continue with Batch [N]: [Batch Title]

Context & Prior Work:
- Completed in prior batch: [Summary of files and state]
- Key references: [file paths touched]

Tasks for this batch:
1. [Task 1]
2. [Task 2]

Remember to follow the Scope Estimation & Chunked Execution Protocol.
```
```
