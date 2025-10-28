"""Conflict resolution primitives for the multi-agent coordinator.

The conflict module will host trajectory reservation and resolution helpers
once the centralized CBS-style solver is reintroduced.  The file is kept so the
package exports remain stable for downstream imports, and future commits can
grow it with dedicated data structures for temporal conflicts.
"""

