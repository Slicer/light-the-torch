---
title: Linux nightly binaries cannot be installed exclusively from the PyTorch wheel indices
assignees: jamesobutler
---

See https://github.com/{{ env.REPO }}/actions/runs/{{ env.ID }} for details.

This opens up the possibility of a supply chain attack if `--extra-index-url` is used.
Please report this to PyTorch immediately.
