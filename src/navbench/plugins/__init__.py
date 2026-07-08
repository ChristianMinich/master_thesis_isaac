"""Approach plugins (one folder per approach).

Each subpackage is self-registering: importing it (via ``plugin_module`` in an
approach YAML) runs its ``@register_approach()`` decorator. Core never imports
anything from this package directly.
"""