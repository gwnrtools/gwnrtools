# Copyright (c) 2026, Prayush Kumar
# See LICENSE file for details: <https://github.com/gwnr/gwnr/blob/master/LICENSE>
"""Shared pytest configuration for the gwnr test suite.

Ensures the working-tree package (repository root) is imported in
preference to any installed copy, and forces a non-interactive
matplotlib backend so plotting tests can run headless.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
