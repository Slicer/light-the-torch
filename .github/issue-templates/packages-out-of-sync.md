---
title: light-the-torch is out of sync with the PyTorch package indices
assignees: jamesobutler
---
{%- if env.MISSING|length %}
The following packages are available, but not patched:
{% for package in env.MISSING.split(",") %}
  - `{{ package }}`
{%- endfor %}
{%- endif %}
{% if env.EXTRA|length %}
The following packages are patched, but not available:
{% for package in env.EXTRA.split(",") %}
  - `{{ package }}`
{%- endfor %}
{%- endif %}
