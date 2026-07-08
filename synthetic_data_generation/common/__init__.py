"""Shared utilities for Go2 synthetic data generation scripts.

This package is intentionally lightweight (numpy + PyYAML only) so every
generator can also run in ``--mock`` mode on machines without Isaac Sim /
Isaac Lab. Torch and Isaac imports happen lazily inside the Isaac adapter.
"""