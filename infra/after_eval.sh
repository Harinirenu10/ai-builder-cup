#!/usr/bin/env bash
# After evaluation: back to min instances 0 on api and reader.
exec "$(dirname "${BASH_SOURCE[0]}")/set_min_instances.sh" 0 api reader
