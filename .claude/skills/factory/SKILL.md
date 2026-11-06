---
name: factory
description: Software factory orchestrator. Runs a feature through spec, build, four parallel reviews, and an approver, with a live dashboard. Use when the user types /factory, or asks to run a feature through the factory, build the next backlog item, approve a needs-human job, or send a job back for rework.
---

# /factory

You are the orchestrator. You spawn every agent, and you are the **only writer of `factory/board.json`**. Agents read and write markdown files in `factory/jobs/<job-id>/`; they never see this conversation, so every task you give them is a folder path, not a briefing.

```
feature -> spec-writer -> builder -> [security|ux|ui|code] -> all PASS -> approver -> APPROVE: merge
                            ^              wait for all 4        |            \-> ESCALATE: needs-human
                            '--- any CHANGES: resume builder (max 3 rounds)
```

## Commands
- `/factory <feature>` — new job
- `/factory next` — first `- [ ]` line in `factory/backlog.md`; tick it `- [x]` once the job is created
- `/factory all` — run every `- [ ]` item in `factory/backlog.md`, top to bottom, one job at a time (they share one checkout). A `needs-human` job does not stop the queue: leave its branch unmerged, switch back to `main`, and start the next item. Later jobs branch from `main`, so they do not include earlier unmerged work. Tick each item `- [x]` when its job is created, and finish with one summary line per job.
- `/factory approve <job-id>` — job must be `needs-human`: `git switch main && git merge --no-ff factory/<job-id>`, stage -> `merged`
- `/factory rework <job-id> <note>` — write the note to `jobs/<id>/rework.md`, bump `round`, resume `builder-<id>` (SendMessage) pointing at that file, then run the review step again

## Setup (every run)
1. Ensure `factory/repo.md` exists. If not, inspect the repo (language, install, **test command**, conventions, layout) and write it in under 25 lines. Every agent reads it.
2. Require a clean tree on `main` (`git status`); if not, stop and tell the user.
3. Load `factory/board.json` (create `{"jobs":[]}` if missing). New id = highest number + 1, three digits, plus a short slug: `003-rate-limiting`.

## Running a job
Update `board.json` at every transition (stage and review verdicts) so the dashboard stays live. Stages: `spec`, `build`, `review`, `approve`, `merged`, `needs-human`.

1. **spec** — spawn `spec-writer` with the job folder and the feature text. Check `spec.md` exists.
2. **build** — spawn `builder` with `name: "builder-<job-id>"`, task: "Job folder: factory/jobs/<id>". Check `build.md` exists and the branch `factory/<id>` has commits. If the builder reports it can't pass tests, mark `needs-human`.
3. **review** — set `round` to 1, create `round-1/`, set `reviews` to all `pending`. Spawn the four reviewers **in one message** (security-reviewer, ux-reviewer, ui-reviewer, code-reviewer), each told the job folder and round folder. WAIT FOR ALL: the round is done only when all four `review-*.md` files exist. Read line 1 of each, update `reviews` as they land. Never touch the builder mid-round.
4. **Verdicts**
   - all PASS -> step 5
   - any CHANGES and round < 3 -> push the finished `reviews` object onto the job's `history` array, `round += 1`, create the new round folder, resume the builder with `SendMessage` to `builder-<id>`: "Round N reviews are in factory/jobs/<id>/round-<N-1>/. Fix every CHANGES item." Then repeat step 3 (all four reviewers again).
   - any CHANGES in round 3 -> stage `needs-human`, write `decision.md` yourself starting with `ESCALATE` and the unresolved findings.
5. **approve** — spawn `approver`. Read line 1 of `decision.md`: `APPROVE` -> `git switch main && git merge --no-ff factory/<id>`, run the test command once more, stage `merged`. `ESCALATE` -> stage `needs-human`.

Finish by switching back to `main` and summarising each job in one line: stage, branch, and the reason if it needs a human.

## board.json
```json
{ "jobs": [ { "id": "003-rate-limiting", "title": "Rate limiting", "branch": "factory/003-rate-limiting",
  "stage": "review", "round": 2,
  "reviews": { "security": "CHANGES", "ux": "PASS", "ui": "pending", "code": "PASS" },
  "history": [ { "security": "CHANGES", "ux": "PASS", "ui": "PASS", "code": "CHANGES" } ] } ] }
```
`reviews` is the current round; `history` holds earlier rounds. Verdict values: `pending`, `PASS`, `CHANGES`. Write the file atomically (whole file each time).

## Dashboard
Tell the user once: `cd factory && python3 -m http.server 8000` then open http://localhost:8000/dashboard.html.

Jobs run one at a time because they share one checkout. Agents in `.claude/agents/` load at session start, so after editing them restart the session.
