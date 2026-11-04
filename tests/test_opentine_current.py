"""Behaviour against current opentine: compressed/v4 artifacts, signature
schemes, key hygiene, read-only MCP, and CLI quality-of-life surfaces."""

from __future__ import annotations

import json

import pytest

from graphforge import GraphRecorder
from graphforge.recorder import (
    artifact_signature_info,
    min_hmac_key_bytes,
    read_artifact_document,
)

KEY = b"a-test-key-at-least-16-bytes"


def _signed(tmp_path):
    recorder = GraphRecorder("sig", runs_dir=tmp_path)
    recorder.add_tag("team:graphs")
    recorder.record_done("done")
    return recorder.save(sign_key=KEY, signer="ci")


def test_signature_info_reads_whatever_format_opentine_wrote(tmp_path):
    info = artifact_signature_info(read_artifact_document(_signed(tmp_path)))
    assert info["signature_present"] is True
    assert info["signature_schemes"]
    if info["signature_schemes"] != ["tine-sig/1"]:
        assert info["tags_covered_by_signature"] is True


def test_signature_info_on_unsigned_and_hostile_documents(tmp_path):
    recorder = GraphRecorder("plain", runs_dir=tmp_path)
    recorder.record_done("done")
    unsigned = artifact_signature_info(read_artifact_document(recorder.save()))
    assert unsigned["signature_present"] is False
    assert unsigned["tags_covered_by_signature"] is False
    for hostile in (None, [], "x", {"signatures": "nope"}, {"metadata": {"integrity": 3}}):
        assert artifact_signature_info(hostile)["signature_present"] is False


def test_legacy_v1_signature_does_not_claim_tag_coverage():
    doc = {"metadata": {"integrity": {"signature": {"scheme": "tine-sig/1"}}}}
    info = artifact_signature_info(doc)
    assert info["signature_present"] and not info["tags_covered_by_signature"]


def test_short_hmac_key_is_refused_before_saving(tmp_path):
    if min_hmac_key_bytes() == 0:
        pytest.skip("installed opentine has no HMAC key floor")
    recorder = GraphRecorder("short", runs_dir=tmp_path)
    recorder.record_done("done")
    with pytest.raises(ValueError, match="at least"):
        recorder.save(sign_key=b"short")


def test_runs_dir_env_override(monkeypatch, tmp_path):
    import importlib

    import graphforge.recorder as rec

    monkeypatch.setenv("GRAPHFORGE_RUNS_DIR", str(tmp_path / "runs"))
    try:
        assert importlib.reload(rec).DEFAULT_RUNS_DIR == tmp_path / "runs"
    finally:
        monkeypatch.delenv("GRAPHFORGE_RUNS_DIR")
        importlib.reload(rec)


# --- CLI -----------------------------------------------------------------------


def test_verify_takes_the_key_from_the_environment(tmp_path, monkeypatch):
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from graphforge.cli import app

    path = _signed(tmp_path)
    runner = CliRunner()
    monkeypatch.setenv("GF_KEY", KEY.decode())
    good = runner.invoke(app, ["verify", str(path), "--hmac-key-env", "GF_KEY"])
    assert good.exit_code == 0, good.stdout
    payload = json.loads(good.stdout)
    assert payload["signature_checked"] and payload["signature_ok"]
    # an unset variable fails closed rather than skipping the check
    unset = runner.invoke(app, ["verify", str(path), "--hmac-key-env", "GF_MISSING"])
    assert unset.exit_code != 0
    both = runner.invoke(
        app, ["verify", str(path), "--hmac-key-env", "GF_KEY", "--hmac-key", "x" * 20]
    )
    assert both.exit_code != 0


def test_verify_rejects_the_wrong_key(tmp_path):
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from graphforge.cli import app

    path = _signed(tmp_path)
    bad = CliRunner().invoke(app, ["verify", str(path), "--hmac-key", "z" * 32])
    assert bad.exit_code == 1


def test_version_and_list(tmp_path):
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from graphforge.cli import app

    runner = CliRunner()
    version = runner.invoke(app, ["--version"])
    assert version.exit_code == 0 and "opentine" in version.stdout
    _signed(tmp_path)
    listed = runner.invoke(app, ["list", "--runs-dir", str(tmp_path)])
    assert listed.exit_code == 0
    assert len(json.loads(listed.stdout)) == 1


# --- MCP -----------------------------------------------------------------------


def _tool_names(server):
    import asyncio

    return {tool.name for tool in asyncio.run(server.list_tools())}


def test_read_only_mcp_server_drops_write_tools(tmp_path):
    pytest.importorskip("mcp")
    from graphforge.mcp_server import build_server

    assert "fork_graph_run" in _tool_names(build_server(str(tmp_path)))
    read_only = _tool_names(build_server(str(tmp_path), read_only=True))
    assert "fork_graph_run" not in read_only
    assert {"list_graph_runs", "show_graph_run", "graph_diff_runs"} <= read_only
