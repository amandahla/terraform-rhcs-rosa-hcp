#!/usr/bin/env python3
"""
Terraform AWS provider compatibility checker for pull requests.

Reads PR diff for new aws_* resource/data blocks, resolves hashicorp/aws constraints
from the nearest versions.tf, fetches AWS provider CHANGELOG excerpts, and uses an LLM
to infer the introduction version. Posts or updates a single PR comment when a bump
may be needed.

Environment:
  GITHUB_TOKEN (required)
  GITHUB_REPOSITORY (required in Actions)
  GITHUB_EVENT_PATH (required for pull_request events)
  GOOGLE_API_KEY or GEMINI_API_KEY (required for Gemini / google-genai)
  GEMINI_MODEL (optional, default gemini-2.0-flash)
  PR_NUMBER (optional override for local testing)
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import requests
from github import Auth, Github
from packaging.version import InvalidVersion, Version

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CHANGELOG_URL = (
    "https://raw.githubusercontent.com/hashicorp/terraform-provider-aws/main/CHANGELOG.md"
)
COMMENT_MARKER = "<!-- tf-aws-compat-check -->"
RESOURCE_LINE_RE = re.compile(
    r'^\+\s*(resource|data)\s+"([^"]+)"\s+"([^"]+)"\s*\{?\s*$'
)
VERSIONS_TF_NAME = "versions.tf"
AWS_PROVIDER_SOURCES = frozenset({"hashicorp/aws", "aws"})

# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------


@dataclass
class IntroResult:
    resource_type: str
    introduced_in: str | None
    confidence: str
    evidence_line: str
    error: str | None = None


@dataclass
class Issue:
    file_path: str
    resource_type: str
    versions_tf: str
    constraint_raw: str
    lower_bound: Version
    intro: IntroResult


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    unverified: list[tuple[str, str, str]] = field(default_factory=list)  # path, type, reason


# -----------------------------------------------------------------------------
# Diff parsing
# -----------------------------------------------------------------------------


def parse_added_aws_types_from_patch(patch: str | None) -> set[str]:
    """Extract aws_* resource/data types from added lines in a unified diff patch."""
    if not patch:
        return set()
    types: set[str] = set()
    for line in patch.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        m = RESOURCE_LINE_RE.match(line)
        if m:
            rtype = m.group(2)
            if rtype.startswith("aws_"):
                types.add(rtype)
    return types


def min_lower_bound_from_constraint(version_string: str) -> Version | None:
    """
    Best-effort minimum lower bound from constraint string (e.g. '>= 5.38.0').
    Uses the maximum of all '>= X' clauses when multiple are present.
    """
    parts = re.findall(r">=\s*([0-9]+(?:\.[0-9]+)*(?:-[a-zA-Z0-9.]+)?)", version_string)
    if not parts:
        return None
    versions: list[Version] = []
    for p in parts:
        try:
            versions.append(Version(p))
        except InvalidVersion:
            continue
    return max(versions) if versions else None


def find_versions_tf_with_aws_constraint(repo_root: Path, start_dir: Path) -> tuple[Path | None, str | None]:
    """
    Walk upward from start_dir to repo_root looking for versions.tf that declares
    hashicorp/aws (or alias aws) in required_providers.
    """
    current = start_dir.resolve()
    root = repo_root.resolve()
    while root in current.parents or current == root:
        vf = current / VERSIONS_TF_NAME
        if vf.is_file():
            text = vf.read_text(encoding="utf-8", errors="replace")
            ver = extract_aws_provider_version(text)
            if ver is not None:
                return vf, ver
        if current == root:
            break
        current = current.parent
    return None, None


def extract_aws_provider_version(versions_tf_content: str) -> str | None:
    """
    Very small HCL scanner: find aws block under required_providers and return version string.
    """
    # Strip single-line comments
    lines = []
    for line in versions_tf_content.splitlines():
        if "#" in line:
            line = line.split("#", 1)[0]
        lines.append(line)
    text = "\n".join(lines)

    # Find required_providers { ... } (non-nested brace matching simplified)
    m = re.search(r"required_providers\s*\{", text)
    if not m:
        return None
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    block = text[start : i - 1]

    # aws = { source = "..." version = "..." }
    for name in ("aws", '"aws"'):
        block_re = re.compile(
            rf"{name}\s*=\s*\{{([^}}]*)\}}",
            re.DOTALL | re.IGNORECASE,
        )
        bm = block_re.search(block)
        if not bm:
            continue
        inner = bm.group(1)
        src_m = re.search(
            r'source\s*=\s*"([^"]+)"',
            inner,
        )
        if src_m and src_m.group(1) not in AWS_PROVIDER_SOURCES:
            continue
        ver_m = re.search(r'version\s*=\s*"([^"]+)"', inner)
        if ver_m:
            return ver_m.group(1).strip()
    return None


# -----------------------------------------------------------------------------
# CHANGELOG + LLM
# -----------------------------------------------------------------------------


def fetch_changelog() -> str:
    r = requests.get(CHANGELOG_URL, timeout=60)
    r.raise_for_status()
    return r.text


def changelog_snippets_for_resource(changelog: str, resource_type: str, max_chars: int = 12000) -> str:
    """
    Collect changelog sections likely mentioning this resource (slug without aws_ prefix).
    """
    slug = resource_type[4:] if resource_type.startswith("aws_") else resource_type
    patterns = [
        re.compile(rf"\b{re.escape(slug)}\b", re.IGNORECASE),
        re.compile(rf"\b{re.escape(resource_type)}\b", re.IGNORECASE),
    ]

    # Split by ## x.y.z headings
    sections = re.split(r"(?m)^##\s+([0-9]+\.[0-9]+\.[0-9][^\n]*)\s*$", changelog)
    # sections[0] is preamble, then pairs (version_line, body)
    chunks: list[str] = []
    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            break
        heading = sections[i].strip()
        body = sections[i + 1]
        combined = f"## {heading}\n{body}"
        if any(p.search(combined) for p in patterns):
            chunks.append(combined[:4000])
        if sum(len(c) for c in chunks) > max_chars:
            break
    if not chunks:
        # Fallback: any line mentioning slug
        lines = [ln for ln in changelog.splitlines() if slug.replace("_", "") in ln.replace("_", "").lower() or slug in ln]
        return "\n".join(lines[:200])[:max_chars]
    return "\n\n---\n\n".join(chunks)[:max_chars]


def llm_intro_version(resource_type: str, snippet: str) -> IntroResult:
    """Call Gemini (google-genai) to infer introduction version from changelog excerpt."""
    system = (
        "You are a precise assistant. Given Terraform AWS provider changelog excerpts, "
        "identify the FIRST provider version that introduced or added the resource type. "
        "Respond with JSON only, no markdown: "
        '{"resource_type":"...","introduced_in":"MAJOR.MINOR.PATCH|null",'
        '"confidence":"high|medium|low","evidence_line":"short quote or empty"}'
    )
    user = f"Resource type: {resource_type}\n\nCHANGELOG excerpts:\n{snippet}\n"

    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return _gemini_json(system, user, resource_type)
    return IntroResult(
        resource_type=resource_type,
        introduced_in=None,
        confidence="low",
        evidence_line="",
        error="No GOOGLE_API_KEY or GEMINI_API_KEY set",
    )


def _gemini_json(system: str, user: str, resource_type: str) -> IntroResult:
    # New Google Gen AI SDK: https://ai.google.dev/gemini-api/docs/migrate
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return IntroResult(
            resource_type, None, "low", "", "google-genai package not installed"
        )
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return IntroResult(
            resource_type, None, "low", "", "GOOGLE_API_KEY or GEMINI_API_KEY not set"
        )
    client = genai.Client(api_key=api_key)
    # GitHub Actions may set GEMINI_MODEL to "" when the repo variable is unset;
    # os.environ.get("GEMINI_MODEL", "x") then returns "" and the API raises "model is required".
    model_name = (os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip() or "gemini-2.0-flash"
    resp = client.models.generate_content(
        model=model_name,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    raw = (resp.text or "").strip()
    # Strip markdown fences if any
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return _parse_llm_json(raw, resource_type)


def _parse_llm_json(raw: str, resource_type: str) -> IntroResult:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return IntroResult(resource_type, None, "low", "", f"JSON parse error: {e}")
    introduced = data.get("introduced_in")
    if isinstance(introduced, str) and introduced.lower() in ("null", "none", "unknown", ""):
        introduced = None
    return IntroResult(
        resource_type=data.get("resource_type") or resource_type,
        introduced_in=introduced,
        confidence=str(data.get("confidence", "low")).lower(),
        evidence_line=str(data.get("evidence_line", "")),
        error=None,
    )


# -----------------------------------------------------------------------------
# PR comment
# -----------------------------------------------------------------------------


def build_comment_body(report: Report) -> str | None:
    # Only open/ update a PR thread when we have high/medium-confidence bump findings.
    if not report.issues:
        return None
    lines = [
        COMMENT_MARKER,
        "### Terraform AWS provider compatibility",
        "",
    ]
    if report.issues:
        lines.append(
            "The following **new `aws_*` blocks** may require a **higher `hashicorp/aws`** "
            "constraint than the minimum implied by your `versions.tf` "
            "(Terraform could resolve an older provider within the allowed range).",
        )
        lines.append("")
        for iss in report.issues:
            lines.append(f"- **{iss.resource_type}** in `{iss.file_path}`")
            lines.append(
                f"  - `versions.tf`: `{iss.versions_tf}` (current constraint: `{iss.constraint_raw}`)",
            )
            lines.append(
                f"  - Estimated introduction: **{iss.intro.introduced_in}** "
                f"(confidence: _{iss.intro.confidence}_)",
            )
            if iss.intro.evidence_line:
                lines.append(f"  - Evidence: _{iss.intro.evidence_line[:200]}_")
            lines.append(
                f"  - **Suggestion:** set `version = \">= {iss.intro.introduced_in}\"` "
                f"(or higher) for `aws` in that `versions.tf`, if not already constrained by a higher floor.",
            )
            lines.append("")
    if report.unverified:
        lines.append("**Also noted (could not verify)**:")
        for fp, rtype, reason in report.unverified:
            lines.append(f"- `{rtype}` in `{fp}`: {reason}")
        lines.append("")
    lines.append("_This check uses CHANGELOG excerpts and an LLM; confirm against the [AWS provider changelog](https://github.com/hashicorp/terraform-provider-aws/blob/main/CHANGELOG.md)._")
    return "\n".join(lines)


def upsert_pr_comment(pr, body: str) -> None:
    login = pr.user.raw_data["login"] if pr.user else ""
    for c in pr.get_issue_comments():
        if COMMENT_MARKER in c.body:
            c.edit(body)
            return
    pr.create_issue_comment(body)


# -----------------------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------------------


def collect_report(
    repo_root: Path,
    changed_files: Iterable[tuple[str, str | None]],
    changelog: str,
) -> Report:
    report = Report()
    seen_types: dict[str, tuple[str, str, Version, str]] = {}

    for file_path, patch in changed_files:
        if not file_path.endswith(".tf"):
            continue
        aws_types = parse_added_aws_types_from_patch(patch)
        if not aws_types:
            continue
        start_dir = repo_root / Path(file_path).parent
        vf_path, constraint = find_versions_tf_with_aws_constraint(repo_root, start_dir)
        if not constraint or not vf_path:
            for t in aws_types:
                report.unverified.append(
                    (file_path, t, "No versions.tf with hashicorp/aws found in parent path"),
                )
            continue
        lower = min_lower_bound_from_constraint(constraint)
        if lower is None:
            for t in aws_types:
                report.unverified.append(
                    (file_path, t, f"Could not parse lower bound from `{constraint}`"),
                )
            continue

        rel_vf = str(vf_path.relative_to(repo_root))

        for rtype in sorted(aws_types):
            snippet = changelog_snippets_for_resource(changelog, rtype)
            intro = llm_intro_version(rtype, snippet)
            if not intro.introduced_in or intro.introduced_in.lower() in ("null", "none", "unknown"):
                report.unverified.append(
                    (file_path, rtype, intro.error or "LLM could not determine introduced_in"),
                )
                continue
            try:
                v_intro = Version(intro.introduced_in)
            except InvalidVersion:
                report.unverified.append(
                    (file_path, rtype, f"Invalid introduced_in `{intro.introduced_in}`"),
                )
                continue

            # Bump if declared floor is strictly older than introduction (semver).
            if v_intro > lower and intro.confidence in ("high", "medium"):
                key = f"{rel_vf}::{rtype}"
                if key not in seen_types:
                    seen_types[key] = (file_path, rel_vf, lower, constraint)
                    report.issues.append(
                        Issue(
                            file_path=file_path,
                            resource_type=rtype,
                            versions_tf=rel_vf,
                            constraint_raw=constraint,
                            lower_bound=lower,
                            intro=intro,
                        )
                    )
            elif v_intro > lower and intro.confidence == "low":
                report.unverified.append(
                    (file_path, rtype, "introduced_in may exceed constraint but confidence is low"),
                )

    return report


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 1

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print(
            "No GOOGLE_API_KEY or GEMINI_API_KEY: skipping compatibility check (no PR comment).",
        )
        return 0

    repo_name = os.environ.get("GITHUB_REPOSITORY")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    pr_number = os.environ.get("PR_NUMBER")

    if not pr_number:
        if not event_path or not os.path.isfile(event_path):
            print("GITHUB_EVENT_PATH must be set or pass PR_NUMBER", file=sys.stderr)
            return 1
        with open(event_path, encoding="utf-8") as f:
            event = json.load(f)
        if event.get("pull_request"):
            pr_number = str(event["pull_request"]["number"])
        else:
            print("Not a pull_request event; set PR_NUMBER to run manually", file=sys.stderr)
            return 0

    if not repo_name:
        print("GITHUB_REPOSITORY is required", file=sys.stderr)
        return 1

    g = Github(auth=Auth.Token(token))
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(int(pr_number))

    repo_root = Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()

    changed: list[tuple[str, str | None]] = []
    for f in pr.get_files():
        if f.status == "removed":
            continue
        changed.append((f.filename, f.patch))

    try:
        changelog = fetch_changelog()
    except requests.RequestException as e:
        print(f"CHANGELOG fetch failed: {e}", file=sys.stderr)
        return 1

    report = collect_report(repo_root, changed, changelog)
    if report.unverified:
        print("Unverified (logged only or appended if comment posted):", file=sys.stderr)
        for fp, rtype, reason in report.unverified:
            print(f"  {rtype} @ {fp}: {reason}", file=sys.stderr)
    body = build_comment_body(report)
    if body:
        upsert_pr_comment(pr, body)
        print("Posted compatibility comment.")
    else:
        print("No high/medium-confidence compatibility bumps to report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
