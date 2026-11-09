"""Validate the bundled structured-query registry without clinical data."""

from semantic.structured_query import AnatomicalQueryParser


def test_registry_loads_and_contains_compositional_targets():
    parser = AnatomicalQueryParser()
    assert parser.schema_by_target
    assert "whole_airway" in parser.schema_by_target

