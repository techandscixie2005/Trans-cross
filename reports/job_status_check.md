# Trans-cross Slurm Job Status Check

## 1. Check Time
- **Date/Time**: 2026-05-21 ~18:40 CST
- **Server**: bjhpc
- **User**: sczc698
- **Code path**: /data/home/sczc698/run/xxy/Trans-cross/code/
- **Git commit**: 8f3db4a (master)

## 2. Queue Status

| job id | array task | name | state | reason | elapsed | node | exit code |
|---|---|---|---|---|---|---|---|
| 987883 | [0-9] | spe-grid | PENDING | Priority | 0:00 | — | — |
| 986753 | — | jupyter_ | RUNNING | — | 1-03:07 | g0057 | — |
| 976222 | — | jupyter_ | RUNNING | — | 12-02:35 | g0045 | — |

### Previous Grid Job (987794) — ALL FAILED

| task | config | model | seed | exit | error |
|---|---|---|---|---|---|
| 0 | spe_equal | concat_equal | 43 | 1:0 | `activate: No such file or directory` |
| 1 | spe_equal | intra_cross_equal | 43 | 1:0 | same |
| 2 | spe_equal | concat_equal | 44 | 1:0 | same |
| 3 | spe_equal | intra_cross_equal | 44 | 1:0 | same |
| 4 | spe512 | concat_equal | 42 | 1:0 | same |
| 5 | spe512 | intra_cross_equal | 42 | 1:0 | same |
| 6 | spe512 | concat_equal | 43 | 1:0 | same |
| 7 | spe512 | intra_cross_equal | 43 | 1:0 | same |
| 8 | spe512 | concat_equal | 44 | 1:0 | same |
| 9 | spe512 | intra_cross_equal | 44 | 1:0 | same |

**Root cause**: `source activate transpec` failed on node g0053 because conda's shell init was not sourced. Fixed by changing to `eval "$(conda shell.bash hook)" && conda activate transpec`.

## 3. GPU / Partition Status

```
PARTITION AVAIL  TIMELIMIT  NODES(A/I/O/T) NODELIST
gpu*         up   infinite       26/0/30/56 g[0002-0003,0005-0006,...]
```

- **26 nodes available**, 30 other (down/drain/alloc/mix)
- **Blockers**: 2 jupyter sessions (both sczc698's):
  - 986753 on g0057 — running 1d 3h
  - 976222 on g0045 — running 12d 2h
- Neither jupyter job blocks the grid — free GPUs exist (26 available nodes)
- Grid job `Priority` status suggests Slurm scheduler policy, not resource scarcity

## 4. Run Directory Completion

| run | exists | checkpoint | metrics | valid eval | test eval | predictions | status |
|---|---|---|---|---|---|---|---|
| equal_concat_seed42 | YES | YES | YES | YES | YES | YES | COMPLETE |
| equal_intra_cross_seed42 | YES | YES | YES | YES | YES | YES | COMPLETE |
| spe_equal_concat_seed42 | YES | YES | YES | YES | YES | YES | COMPLETE |
| spe_equal_intra_cross_seed42 | YES | YES | YES | YES | YES | YES | COMPLETE |
| spe_equal_concat_seed43 | NO | — | — | — | — | — | PENDING (987883) |
| spe_equal_intra_cross_seed43 | NO | — | — | — | — | — | PENDING (987883) |
| spe_equal_concat_seed44 | NO | — | — | — | — | — | PENDING (987883) |
| spe_equal_intra_cross_seed44 | NO | — | — | — | — | — | PENDING (987883) |
| spe512_equal_concat_seed42 | NO | — | — | — | — | — | PENDING (987883) |
| spe512_equal_intra_cross_seed42 | NO | — | — | — | — | — | PENDING (987883) |
| spe512_equal_concat_seed43 | NO | — | — | — | — | — | PENDING (987883) |
| spe512_equal_intra_cross_seed43 | NO | — | — | — | — | — | PENDING (987883) |
| spe512_equal_concat_seed44 | NO | — | — | — | — | — | PENDING (987883) |
| spe512_equal_intra_cross_seed44 | NO | — | — | — | — | — | PENDING (987883) |

## 5. Log Inspection

| log file | job/task | last status | errors |
|---|---|---|---|
| spe_grid_987794_0.out | 987794_0 | Header printed, then exit | `activate: No such file or directory` |
| spe_grid_987794_[0-9].out | all 10 tasks | Same — failed before training | Same conda activation error |
| spe_e0_concat_987740.out | 987740 | Complete, 30 epochs | None |
| spe_e1_intra_cross_987741.out | 987741 | Complete, 30 epochs | None |
| concat_eq_987636.out | 987636 | Complete, 30 epochs | None |
| intra_cross_eq_987637.out | 987637 | Complete, 30 epochs | None |

## 6. Completed Results So Far (seed42)

| tokenizer | model | loss | canon exact | validity | unique ratio | mode collapse | Tanimoto | scaffold | FG-F1 |
|---|---|---|---|---|---|---|---|---|---|
| atom | E0 | 1.442 | 0.000 | 0.677 | 0.320 | 0.078 | 0.104 | 0.000 | 0.188 |
| atom | E1 | 1.478 | 0.000 | 0.721 | 0.241 | 0.050 | 0.106 | 0.000 | 0.198 |
| spe256 | E0 | 3.880 | 0.000 | 1.000 | 0.066 | 0.221 | 0.140 | 0.000 | 0.301 |
| spe256 | E1 | 3.942 | 0.000 | 0.728 | 0.177 | 0.159 | 0.108 | 0.000 | 0.205 |

## 7. Missing Runs

All 10 multi-seed runs: SPE-256 seeds 43,44 + SPE-512 seeds 42,43,44. Job 987883 pending.

## 8. Recommended Action

**Wait for job 987883 to start.** The script fix (robust conda activation) is in place. Free GPUs exist (26 available nodes). The `Priority` reason should resolve shortly as the scheduler assigns resources.

Once 987883 completes:
1. Run `evaluate_smiles_model.py` on each new run directory (valid + test)
2. Run `condition_shuffle.py` on representative runs
3. Run `audit_generation_behavior.py` on all runs
4. Run `aggregate_final_ablation_results.py` with all 14 run dirs
5. Update `reports/final_e0_vs_e1_smiles_generation_report.md`

No further script fixes needed.
