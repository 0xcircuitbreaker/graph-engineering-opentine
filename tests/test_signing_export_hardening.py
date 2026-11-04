"""CLI signing, OTel export, Ed25519 key files, and input/endpoint hardening."""

from __future__ import annotations

import json

import pytest

from graphforge import GraphEngine, GraphRecorder
from graphforge.examples import fanout_graph

KEY = "a-test-key-at-least-16-bytes"


def _cli():
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from graphforge.cli import app

    return CliRunner(), app


def _finished(tmp_path):
    recorder = GraphRecorder("sign me", runs_dir=tmp_path)
    recorder.record_node("a", state_after={"k": 1})
    recorder.record_done("done")
    return recorder.save()


def _ed25519_keys(tmp_path):
    signing = pytest.importorskip("opentine.signing")
    if not getattr(signing, "HAS_ED25519", False) or not hasattr(signing, "generate_ed25519"):
        pytest.skip("installed opentine cannot generate Ed25519 keys")
    private, public = signing.generate_ed25519()
    priv, pub = tmp_path / "k.pem", tmp_path / "k.pub"
    priv.write_text(private)
    pub.write_text(public)
    return priv, pub


def test_sign_with_hmac_from_env_then_verify(tmp_path, monkeypatch):
    runner, app = _cli()
    path = _finished(tmp_path)
    monkeypatch.setenv("GF_SIGN", KEY)
    signed = runner.invoke(app, ["sign", str(path), "--hmac-key-env", "GF_SIGN", "--signer", "ci"])
    assert signed.exit_code == 0, signed.stdout
    verified = runner.invoke(app, ["verify", str(path), "--hmac-key-env", "GF_SIGN"])
    assert verified.exit_code == 0, verified.stdout
    assert json.loads(verified.stdout)["signer"] == "ci"


def test_sign_refuses_missing_or_ambiguous_keys(tmp_path, monkeypatch):
    runner, app = _cli()
    path = _finished(tmp_path)
    assert runner.invoke(app, ["sign", str(path)]).exit_code != 0
    assert runner.invoke(app, ["sign", str(path), "--hmac-key-env", "GF_UNSET"]).exit_code != 0
    monkeypatch.setenv("GF_SHORT", "short")
    short = runner.invoke(app, ["sign", str(path), "--hmac-key-env", "GF_SHORT"])
    assert short.exit_code != 0


def test_sign_to_output_leaves_the_source_unsigned(tmp_path, monkeypatch):
    from graphforge.recorder import artifact_signature_info, read_artifact_document

    runner, app = _cli()
    path = _finished(tmp_path)
    out = tmp_path / "signed" / "copy.tine"
    monkeypatch.setenv("GF_SIGN", KEY)
    result = runner.invoke(
        app, ["sign", str(path), "--hmac-key-env", "GF_SIGN", "--output", str(out)]
    )
    assert result.exit_code == 0, result.stdout
    assert artifact_signature_info(read_artifact_document(out))["signature_present"]
    assert not artifact_signature_info(read_artifact_document(path))["signature_present"]


def test_ed25519_sign_and_verify_with_opentine_key_files(tmp_path):
    runner, app = _cli()
    priv, pub = _ed25519_keys(tmp_path)
    path = _finished(tmp_path)
    signed = runner.invoke(app, ["sign", str(path), "--ed25519-key", str(priv)])
    assert signed.exit_code == 0, signed.stdout
    good = runner.invoke(app, ["verify", str(path), "--public-key", str(pub)])
    assert good.exit_code == 0, good.stdout
    assert json.loads(good.stdout)["signature_state"] == "verified"
    other = tmp_path / "other"
    other.mkdir()
    _, other_pub = _ed25519_keys(other)
    bad = runner.invoke(app, ["verify", str(path), "--public-key", str(other_pub)])
    assert bad.exit_code == 1


def test_export_otel(tmp_path):
    runner, app = _cli()
    opentine = pytest.importorskip("opentine")
    if not hasattr(opentine, "to_otel_genai_document"):
        pytest.skip("installed opentine has no OTel exporter")
    result = GraphEngine(fanout_graph(), runs_dir=tmp_path).run(
        goal="q", initial_state={"question": "q"}
    )
    out = tmp_path / "trace.json"
    exported = runner.invoke(
        app, ["export-otel", result.run_id[:12], "--runs-dir", str(tmp_path), "--out", str(out)]
    )
    assert exported.exit_code == 0, exported.stdout
    assert "resourceSpans" in json.loads(out.read_text())


# --- hardening -----------------------------------------------------------------


@pytest.mark.parametrize(
    "endpoint",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "http://api.example.com/v1/chat/completions",
        "https:///no-host",
        "not a url",
    ],
)
def test_model_adapter_refuses_unsafe_endpoints(endpoint):
    from graphforge.models import OpenAICompatAdapter

    with pytest.raises(ValueError, match="refusing model endpoint"):
        OpenAICompatAdapter(api_key="k", endpoint=endpoint).complete("hi")


@pytest.mark.parametrize(
    "endpoint",
    ["https://api.example.com/v1", "http://localhost:8000/v1", "http://127.0.0.1/v1"],
)
def test_model_adapter_accepts_https_and_loopback(endpoint):
    from graphforge.models import _check_endpoint

    _check_endpoint(endpoint)


def test_ingest_refuses_oversized_and_non_regular_files(tmp_path):
    from graphforge.ingest import read_bounded_text

    big = tmp_path / "big.jsonl"
    big.write_text("x" * 100)
    with pytest.raises(ValueError, match="limit"):
        read_bounded_text(big, limit=10)
    with pytest.raises(ValueError, match="regular file"):
        read_bounded_text(tmp_path)
    assert read_bounded_text(big, limit=100) == "x" * 100
