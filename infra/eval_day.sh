#!/usr/bin/env bash
# Evaluation days: keep one api and one reader instance warm so judges never wait on a cold start. Undo with infra/after_eval.sh.
exec "$(dirname "${BASH_SOURCE[0]}")/set_min_instances.sh" 1 api reader
