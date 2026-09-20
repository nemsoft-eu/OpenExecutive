from __future__ import annotations

import contextlib
import os
import secrets
import stat
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TargetCustomer(BaseModel):
    profile: str = ""
    pain_points: list[str] = Field(default_factory=list)


class CompetitiveLandscape(BaseModel):
    primary_competitors: list[str] = Field(default_factory=list)
    competitive_advantages: list[str] = Field(default_factory=list)


class OrgStructure(BaseModel):
    departments: list[str] = Field(default_factory=list)
    leadership_team: list[str] = Field(default_factory=list)


class StrategicPriorities(BaseModel):
    current_year: list[str] = Field(default_factory=list)
    north_star_metric: str = ""


class Culture(BaseModel):
    values: list[str] = Field(default_factory=list)
    operating_principles: list[str] = Field(default_factory=list)


class Financials(BaseModel):
    burn_rate_monthly: float | None = None
    runway_months: float | None = None
    key_metrics: dict[str, Any] = Field(default_factory=dict)


def _bullets(label: str, items: list[str]) -> str:
    """Render *items* as one-per-line bullets under a bold *label*.

    A separator that can occur inside an entry silently invents entries. Both
    obvious candidates do occur in the shipped fixtures: ", " sits inside 5 of 8
    halcyon competitors and every leadership entry ("Jordan Avery, CEO &
    Co-founder — …"), and "; " sits inside 3 of 8 tandem competitors ("Figure AI
    — best-funded humanoid pure-play; BMW and logistics pilots", which a
    semicolon join turns into a phantom competitor called "BMW and logistics
    pilots"). A newline plus "- " cannot collide with either.

    This also makes each entry its own line, which is what lets the specialist
    digest be trimmed a whole line at a time rather than mid-character — see
    ``orchestrator.session._fit_lines``.
    """
    return f"**{label}**:\n" + "\n".join(f"- {item}" for item in items)


class CompanyProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = ""
    industry: str = ""
    stage: str = ""
    founding_year: int | None = None
    headcount: int | None = None
    annual_revenue_arr: float | None = None
    mission: str = ""
    vision: str = ""
    target_customer: TargetCustomer = Field(default_factory=TargetCustomer)
    competitive_landscape: CompetitiveLandscape = Field(
        default_factory=CompetitiveLandscape
    )
    org_structure: OrgStructure = Field(default_factory=OrgStructure)
    strategic_priorities: StrategicPriorities = Field(
        default_factory=StrategicPriorities
    )
    culture: Culture = Field(default_factory=Culture)
    financials: Financials = Field(default_factory=Financials)
    # External entities the company depends on or tracks. Named here so the
    # research watchlist policy can treat a watch on them as grounded in
    # company data (auto-added) rather than inferred (needs approval).
    vendors: list[str] = Field(default_factory=list)  # e.g. ["Stripe", "AWS"]
    tickers: list[str] = Field(default_factory=list)  # own + competitor tickers

    @classmethod
    def load_from_yaml(cls, path: Path | str) -> CompanyProfile:
        path = Path(path)
        if not path.exists():
            return cls()
        # Explicit UTF-8: profile.yaml is routinely hand-edited, so it can hold
        # real non-ASCII text. Without this, the platform default (cp1252 on
        # Windows) silently mojibakes it into the cached company-profile prompt
        # block rather than raising.
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        company_data = data.get("company", data)
        return cls.model_validate(company_data)

    def save_to_yaml(self, path: Path | str) -> None:
        # Resolve first so a symlinked profile path (a common deployment
        # pattern for COMPANY_PROFILE_PATH) is written at its target and the
        # link survives, instead of being replaced by a regular file. A
        # symlink loop raises RuntimeError on 3.11 and OSError on 3.13; fall
        # back to the unresolved path and let the open() below report it.
        path = Path(path)
        try:
            path = path.resolve()
        except (OSError, RuntimeError):
            path = path.absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"company": self.model_dump()}
        # Write-then-rename so a failure mid-dump (disk full, unrepresentable
        # value) can never leave a truncated profile behind: every other
        # subsystem loads this file, and callers such as the onboarding
        # route roll back on failure assuming the old profile survived.
        #
        # - O_EXCL via `opener`: the temp name is created fresh and never
        #   follows a pre-planted symlink. Going through open() (not
        #   os.fdopen) keeps the explicit-encoding contract visible.
        # - A random component plus O_EXCL: two processes on one volume
        #   (both PID 1 in their containers) cannot collide.
        # - The temp is always created 0600 and only widened to the
        #   destination's mode after fsync, right before the rename. A
        #   crash leftover (SIGKILL, OOM) is therefore private clutter,
        #   never a readable copy of the financials. There is no sweep of
        #   leftovers: one cannot be told apart from another process's
        #   in-flight file, and unlinking that makes its rename fail.
        # - The destination's mode is carried over: rename creates a new
        #   inode, so a file an operator chmod'd 0600 would otherwise
        #   silently revert to the umask default. A symlink's own mode
        #   (always 0777) is never used.
        # - fsync file and directory: a hard crash between write and
        #   rename must not leave a zero-length profile.
        tmp_path = path.with_name(f".tmp-{os.getpid()}-{secrets.token_hex(4)}-{path.name}")
        existing_mode: int | None = None
        with contextlib.suppress(OSError):
            st = path.lstat()
            if not stat.S_ISLNK(st.st_mode):
                existing_mode = stat.S_IMODE(st.st_mode)

        def _exclusive(p: str, flags: int) -> int:
            return os.open(p, flags | os.O_EXCL, 0o600)

        try:
            with open(tmp_path, "w", encoding="utf-8", opener=_exclusive) as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
                if existing_mode is not None and hasattr(os, "fchmod"):
                    os.fchmod(f.fileno(), existing_mode)
            os.replace(tmp_path, path)
            with contextlib.suppress(OSError):
                dir_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def to_prompt_block(self) -> str:
        if not self.name:
            return ""

        lines = ["## Company Context", ""]
        lines.append(f"**Company**: {self.name}")
        if self.industry:
            lines.append(f"**Industry**: {self.industry}")
        if self.stage:
            lines.append(f"**Stage**: {self.stage}")
        if self.founding_year:
            lines.append(f"**Founded**: {self.founding_year}")
        if self.headcount:
            lines.append(f"**Headcount**: {self.headcount}")
        if self.annual_revenue_arr:
            lines.append(f"**ARR**: ${self.annual_revenue_arr:,.0f}")

        if self.mission:
            lines.extend(["", f"**Mission**: {self.mission}"])
        if self.vision:
            lines.append(f"**Vision**: {self.vision}")

        if self.target_customer.profile:
            lines.extend(["", "**Target Customer**:"])
            lines.append(f"  {self.target_customer.profile}")
            if self.target_customer.pain_points:
                lines.append("  Pain points: " + "; ".join(self.target_customer.pain_points))

        if self.competitive_landscape.primary_competitors:
            lines.extend(["", "**Competitive Landscape**:"])
            lines.append(
                "  Competitors: " + ", ".join(self.competitive_landscape.primary_competitors)
            )
            if self.competitive_landscape.competitive_advantages:
                lines.append(
                    "  Our advantages: "
                    + "; ".join(self.competitive_landscape.competitive_advantages)
                )

        if self.vendors or self.tickers:
            lines.extend(["", "**External Dependencies**:"])
            if self.vendors:
                lines.append("  Vendors: " + ", ".join(self.vendors))
            if self.tickers:
                lines.append("  Tracked tickers: " + ", ".join(self.tickers))

        if self.strategic_priorities.current_year:
            lines.extend(["", "**Strategic Priorities (Current Year)**:"])
            for p in self.strategic_priorities.current_year:
                lines.append(f"  - {p}")
            if self.strategic_priorities.north_star_metric:
                lines.append(
                    f"  North Star: {self.strategic_priorities.north_star_metric}"
                )

        if self.culture.values:
            lines.extend(["", f"**Values**: {', '.join(self.culture.values)}"])

        if self.financials.burn_rate_monthly is not None:
            lines.extend(["", "**Financial Position**:"])
            lines.append(
                f"  Monthly burn: ${self.financials.burn_rate_monthly:,.0f}"
            )
            if self.financials.runway_months is not None:
                lines.append(f"  Runway: {self.financials.runway_months:.1f} months")
            for k, v in self.financials.key_metrics.items():
                lines.append(f"  {k}: {v}")

        if self.org_structure.leadership_team:
            lines.extend(["", "**Leadership**: " + ", ".join(self.org_structure.leadership_team)])

        return "\n".join(lines)

    def to_specialist_block(self) -> str:
        """A compact profile for a specialist's conversation context.

        Not a truncation of ``to_prompt_block``: that block runs ~3.7-4.1k chars
        on the shipped fixtures and orders financials near the END, after target
        customer, competitors, vendors, priorities and values — so any head
        slice cheap enough to send to every specialist in a fan-out drops burn
        and runway, which are the facts the block exists to supply. This selects
        and ORDERS the decision-relevant fields instead, so what a cut costs is
        a property of this function rather than of where the cut lands.

        Field order is load-bearing, because the limit downstream
        (``_TAIL_PROFILE_MAX_CHARS`` in ``orchestrator/session.py``) drops whole
        trailing LINES. Identity and financials first, then priorities, then
        market context, then mission and leadership — so an over-long profile
        sheds the roster and the mission statement first. Every list field is
        rendered one entry per line by ``_bullets`` precisely so that trimming
        can never cut through a figure.

        Market context is included rather than trimmed because it is what
        several specialists are explicitly instructed to produce: the CSO is
        told to "name which [moat] the company actually has", the CMO to
        position "against a specific competitive alternative", the CPO to
        "clarify the customer problem being solved" (``prompts.domain_prompts``).
        Competitor entries keep their gloss — "Unitree / UBTech — China cost
        leaders; export-control and tariff-gated" is the company's own thesis,
        which no model prior supplies. The lists are not truncated either: in
        the shipped fixtures the LAST competitor and advantage entries are where
        the regulatory posture sits (meridian's sanctions/compliance advantage
        is entry 5 of 5; tandem's export-control competitor is entry 7 of 8), so
        a "keep the first three" rule would drop exactly what the GC and COO
        need.

        This block never enters the Executive's own routing call: it reaches
        specialists only, via ``route_parallel(conversation_context=...)``, so
        its cost is per-specialist prefill run in parallel, not routing latency.

        The Executive keeps the full block; only specialists get this.
        """
        if not self.name:
            return ""

        head = [f"**Company**: {self.name}"]
        if self.industry:
            head.append(f"industry {self.industry}")
        if self.stage:
            head.append(f"stage {self.stage}")
        if self.headcount:
            head.append(f"headcount {self.headcount}")
        if self.annual_revenue_arr:
            head.append(f"ARR ${self.annual_revenue_arr:,.0f}")
        lines = ["## Company Context", "", " · ".join(head)]

        money: list[str] = []
        if self.financials.burn_rate_monthly is not None:
            money.append(f"monthly burn ${self.financials.burn_rate_monthly:,.0f}")
        if self.financials.runway_months is not None:
            money.append(f"runway {self.financials.runway_months:.1f} months")
        money.extend(f"{k}: {v}" for k, v in self.financials.key_metrics.items())
        if money:
            lines.append("**Financial position**: " + " · ".join(money))

        if self.strategic_priorities.current_year:
            lines.append(_bullets("Priorities", self.strategic_priorities.current_year))
            if self.strategic_priorities.north_star_metric:
                lines.append(
                    f"**North star**: {self.strategic_priorities.north_star_metric}"
                )

        if self.target_customer.profile:
            lines.append(f"**Target customer**: {self.target_customer.profile}")
        if self.target_customer.pain_points:
            lines.append(_bullets("Customer pain points", self.target_customer.pain_points))
        if self.competitive_landscape.competitive_advantages:
            lines.append(
                _bullets(
                    "Competitive advantages",
                    self.competitive_landscape.competitive_advantages,
                )
            )
        if self.competitive_landscape.primary_competitors:
            lines.append(
                _bullets("Competitors", self.competitive_landscape.primary_competitors)
            )

        if self.mission:
            lines.append(f"**Mission**: {self.mission}")
        if self.org_structure.leadership_team:
            lines.append(_bullets("Leadership", self.org_structure.leadership_team))
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return not bool(self.name)
