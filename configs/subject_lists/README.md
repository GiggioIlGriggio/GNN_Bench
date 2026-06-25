# Subject allowlists

Subject-ID allowlists passed to a run via `dataset.subject_list_file=<path>`
(one `sub-XXXXXXXXXX` id per line; `#` comments and blank lines allowed).

`*.txt` files here are **git-ignored** because they list PNC (restricted-access)
subject IDs — generate them locally, don't commit them.

For the age→VWM transfer experiment, generate the VWM cohort with:

```bash
PYTHONPATH=$(pwd) .venv/bin/python scripts/make_pnc_vwm_cohort.py
# -> configs/subject_lists/pnc_vwm_cohort.txt
```

See `docs/runbooks/age-vwm-transfer.md` for why a fixed cohort is required
(byte-for-byte fold alignment between the age source run and the VWM runs).
