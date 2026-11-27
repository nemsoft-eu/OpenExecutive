from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# Topic tag on an alert only the principal may see: one raised on a turn
# about mail from one of their contacts or mail they forwarded, or a meeting
# proposal with a contact. Every surface that serves anyone else — /today for
# a teammate, a teammate's chat digest, the alert review, the activity rail,
# the unattended reflection — leaves it out (see ``is_private_alert``). A
# namespaced token, so no tag the triage model invents can collide with it.
PRIVATE_ALERT_TAG = "private:principal"


def is_private_alert(alert: object) -> bool:
    """Whether ``alert`` (an Alert, ProposalItem or anything with
    ``topic_tags``) is private to the principal."""
    tags = getattr(alert, "topic_tags", None) or []
    return any(str(t).lower() == PRIVATE_ALERT_TAG for t in tags)


# Source of a document the Executive published (`draft_artifact`). Each one
# is its owner's alone (``owner_person_id``; none = the principal's): it never
# reaches a surface that serves anyone else (see ``artifact_visible_to``).
ARTIFACT_SOURCE = "artifact"


def artifact_visible_to(
    owner_person_id: int | None, person_id: int | None, *, is_principal: bool
) -> bool:
    """Whether the person ``person_id`` (``is_principal`` when they are the
    principal) may see a drafted artifact owned by ``owner_person_id``. Not
    even the principal sees a teammate's."""
    if owner_person_id is None:
        return is_principal
    return person_id is not None and person_id == owner_person_id


# Sources of cards that belong to the one person they are routed to: a
# browser hand-over ("Waiting on you: sign in to ...", with the task from that
# person's own chat). Only that person sees one, not even the principal, and
# no surface shared by everyone (``visible_alert``).
# An email watch's card (``monitoring.mailbox``) names a sender and a subject
# from one person's own inbox: theirs alone.
MAILBOX_SOURCE = "mailbox"
PERSONAL_SOURCES: frozenset[str] = frozenset({"browser_handover", MAILBOX_SOURCE})


def visible_alert(
    alert: object,
    viewer: object | None,
    *,
    chat_reader: int | None = None,
    own_chat_readers: dict[int, frozenset[int]] | None = None,
) -> bool:
    """Whether ``alert`` may be shown to ``viewer`` (anything with
    ``person_id`` and ``is_principal``, e.g. ``artifact_records.Viewer``).
    Every alert but a drafted artifact, a personal card or a card from
    someone's own chat may; an artifact only to its owner, a personal card
    (``PERSONAL_SOURCES``) only to the person it is routed to, a card a held
    step raised from one person's own chat only to them and its approver
    (``alerts.own_chat``), and none of them to anyone when there is no viewer
    (a surface shared by everyone). ``chat_reader`` names whose own-chat
    cards a build with no viewer keeps (the principal's own brief). A caller
    filtering a whole list passes ``own_chat_readers`` (``own_chat.readers``
    of that list) so the lookup runs once, not per card."""
    from openexecutive.alerts import own_chat

    reader = getattr(viewer, "person_id", None) if viewer is not None else chat_reader
    if own_chat_readers is None:
        own_chat_readers = own_chat.readers([alert])
    if not own_chat.shown_to(alert, reader, own_chat_readers):
        return False
    source = getattr(alert, "source", None)
    if source in PERSONAL_SOURCES:
        routed = getattr(alert, "routed_to_person_id", None)
        return viewer is not None and routed is not None and getattr(viewer, "person_id", None) == routed
    if source == TEAM_RAISED_SOURCE:
        return _team_card_reader(alert, reader, viewer)
    if source != ARTIFACT_SOURCE:
        return True
    if viewer is None:
        return False
    return artifact_visible_to(
        getattr(alert, "owner_person_id", None),
        getattr(viewer, "person_id", None),
        is_principal=bool(getattr(viewer, "is_principal", False)),
    )


def _team_card_reader(alert: object, reader: int | None, viewer: object | None) -> bool:
    """Whether ``reader`` reads a card a team member raised from their own
    chat (``TEAM_RAISED_SOURCE``): the one who raised it, and whom it is
    for: the person it is routed to, else the people of its department (when
    anyone is in it), else the principal, who owns an unrouted card. Nobody else on the team: it
    carries one person's ask. ``viewer`` None is a build for ``reader``
    alone (the principal's own brief) or, with no reader, for everyone."""
    if reader is None:
        return False
    routed = getattr(alert, "routed_to_person_id", None)
    if reader in (getattr(alert, "owner_person_id", None), routed):
        return True
    if routed is not None:
        return False
    from openexecutive.people import registry as people_registry

    slugs = {t.split(":", 1)[1] for t in getattr(alert, "topic_tags", None) or [] if t.startswith("department:")}
    # Its department's people, when it has any: a department nobody is in
    # (new, or a slug the model got wrong) leaves the card to the principal,
    # so the ask still reaches someone.
    members = {p.id for p in people_registry.list_people() if slugs & set(p.department_slugs or [])} if slugs else set()
    if members:
        return reader in members
    if viewer is not None:
        return bool(getattr(viewer, "is_principal", False))
    principal = people_registry.get_principal()
    return principal is not None and principal.id == reader


class AlertSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


SEVERITY_RANK: dict[str, int] = {
    AlertSeverity.LOW.value: 0,
    AlertSeverity.MEDIUM.value: 1,
    AlertSeverity.HIGH.value: 2,
    AlertSeverity.URGENT.value: 3,
}


class AlertChannel(StrEnum):
    WEB = "web"
    SLACK_DM = "slack_dm"
    EMAIL = "email"
    PERSISTED = "persisted"
    # Shift 3 — broadcast channels. The triage agent picks these when the
    # alert is dept-scoped (no single human owner) or company-wide. The
    # dispatcher reads the accompanying TriageDecision.department_slug /
    # broadcast_integration fields to know which room to post to.
    DEPARTMENT_CHANNEL = "department_channel"
    COMPANY_BROADCAST = "company_broadcast"


# What a card can need from its reader (``Alert.needs``).
NEEDS_ANSWER = "answer"
NEEDS_CHOICE = "choice"
NEEDS_OK = "ok"
NEEDS_FYI = "fyi"
CARD_NEEDS: frozenset[str] = frozenset({NEEDS_ANSWER, NEEDS_CHOICE, NEEDS_OK, NEEDS_FYI})
# A pick offers this many choices, each this long at most.
MIN_CHOICES, MAX_CHOICES, MAX_CHOICE_CHARS = 2, 4, 60


class AlertEvent(BaseModel):
    """An external event that may or may not become an alert."""

    model_config = ConfigDict(extra="ignore")

    source: str  # "email" | "slack" | "document"
    external_id: str  # source-system id for idempotency
    subject: str = ""
    body: str = ""
    from_: str | None = Field(default=None, alias="from")
    channel: str | None = None
    user: str | None = None
    title: str | None = None  # for documents
    # Producer-supplied dedup key that overrides the triage model's own
    # (monitoring sets ``watch:<slug>`` so one watch = one open alert).
    dedup_hint: str = ""
    # With ``dedup_hint``: a later event folds into the open card by adding
    # its words under the earlier ones ("Then: ..."), not replacing them, so
    # a follow-up never erases what was first asked.
    follow_up: bool = False
    # The open card the turn said this is the same matter as
    # (``create_alert``'s ``joins_card``): it joins that card when it goes to
    # the same reader (``alerts.one_card``).
    joins_alert_id: int | None = None
    # Person the alert should be routed to (assigned_to_person_id from the
    # Executive's create_alert tool). Lands on the row so the "principal
    # owns unrouted" rule stops sweeping every triage-born alert to them.
    routed_to_person_id: int | None = None
    # Department the alert belongs to (slug). Becomes the row's
    # ``department:<slug>`` topic tag — a routing hint only, unlike
    # ``channel``, which triage reads as the room a message came from and may
    # answer with a team-room broadcast.
    department: str = ""
    # Private to the principal (see PRIVATE_ALERT_TAG): the pipeline tags it,
    # routes it to the principal, keeps it out of every broadcast and live
    # push, and never coalesces it into a card someone else can see.
    private: bool = False
    # Set only by the create_alert tool on a team member's chat turn: who
    # raised it, for its "Raised by <name>" label. Such an alert keeps a
    # source and id the server picked (``TEAM_RAISED_SOURCE``), whatever the
    # turn asked, and never goes out as a company-wide broadcast — a person
    # or department only (alerts/pipeline.py).
    raised_by: str = ""
    # Who that team member is (their People id), with ``raised_by``: their
    # card is theirs and its reader's, not the whole team's
    # (``visible_alert``).
    raised_by_person_id: int | None = None
    # Set only by the create_alert tool: the Executive itself chose to raise
    # this card. When review makes no decision on it (the call failed, or its
    # answer was cut off), the card is kept as raised instead of dropped
    # (agents.triage). Other events (chat lines, polled mail, monitors) keep
    # the quiet fallback.
    from_executive: bool = False
    # A message passed on to one person (``message_person`` with no chat
    # app to reach them): the card shows the subject and body as written,
    # with no triage rewrite, so nothing from other cards is added to what
    # the sender asked to pass on. Set only by server code, never the model.
    as_written: bool = False
    # What the card needs from its reader (``Alert.needs``).
    needs: str = ""
    question: str = ""
    choices: list[str] = Field(default_factory=list)
    # An email watch's card for one email that reached the Executive: its
    # mail thread and sender, so it joins the card the Executive's handling
    # of that email made for the same person (``alerts.one_card``). Set only
    # by server code (``monitoring.pipeline``).
    mail_thread: str = ""
    mail_from: str = ""


# The source of every alert a team member raises from chat (see
# ``AlertEvent.raised_by``): never "email" or a watch, so it can't pose as
# one, coalesce into one, or claim the id a real one will arrive with.
TEAM_RAISED_SOURCE = "team_member"


class TriageDecision(BaseModel):
    """Structured output of the Triage agent."""

    model_config = ConfigDict(extra="ignore")

    alert: bool
    severity: AlertSeverity = AlertSeverity.LOW
    channels: list[AlertChannel] = Field(default_factory=list)
    headline: str = ""
    body: str = ""
    suggested_action: str = ""
    # The step in ``suggested_action`` in a few words, for its button.
    action_label: str = ""
    topic_tags: list[str] = Field(default_factory=list)
    dedup_key: str = ""
    reason_if_suppressed: str = ""
    # Shift 3 broadcast routing. Set when `channels` contains
    # DEPARTMENT_CHANNEL or COMPANY_BROADCAST so the dispatcher knows
    # WHERE to post. Empty strings when the alert is per-person or
    # web-only — those channels never read these fields.
    department_slug: str = ""
    broadcast_integration: str = ""


class Alert(BaseModel):
    """A persisted alert row."""

    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    external_id: str = ""
    source: str
    severity: str
    headline: str
    body: str
    suggested_action: str = ""
    topic_tags: list[str] = Field(default_factory=list)
    channels_attempted: list[str] = Field(default_factory=list)
    channels_delivered: list[str] = Field(default_factory=list)
    dedup_key: str = ""
    # unread | read | ack | dismissed | resolved | expired. `ack` means the
    # user approved (the Executive executes the suggested action);
    # `resolved` means the Executive's review closed it with evidence;
    # `expired` means the lifecycle sweep aged it out (alerts/lifecycle.py).
    status: str = "unread"
    created_at: str
    # Phase 4: person the alert is routed to for approval. NULL means the
    # alert is general (not routed to a specific approver).
    routed_to_person_id: int | None = None
    # Soft-delete for the Executive Artifacts gallery. NULL = active; an ISO
    # timestamp means the artifact was archived (hidden from the default list
    # but restorable). Only meaningful for source='artifact' rows.
    archived_at: str | None = None
    # Lifecycle (alerts/lifecycle.py + alerts/review.py). All additive with
    # defaults so rows written by older builds read back unchanged.
    # Coalescing: a repeat of an open alert bumps these instead of stacking.
    last_seen_at: str | None = None
    occurrence_count: int = 1
    # Executive review verdict: '' | relevant | changed | likely_stale |
    # drafted | routed | merged, plus resolved | stale on rows the review
    # closed; `review_note` is the one-line "what changed since
    # you last looked"; `recommended_move` the next move the card leads with;
    # `why_now` a short urgency note; `due_at` an ISO deadline when one exists.
    last_reviewed_at: str | None = None
    review_verdict: str = ""
    review_note: str = ""
    recommended_move: str = ""
    why_now: str = ""
    due_at: str | None = None
    # Set on a row folded into another (merge): the surviving alert's id.
    superseded_by_alert_id: int | None = None
    # Hidden from the live queue until this ISO timestamp (snooze).
    snoozed_until: str | None = None
    # Registry workflow the review suggested as the next step ('' = none).
    suggested_workflow: str = ""
    # Artifact format + link metadata (source='artifact' rows only; see
    # orchestrator/artifact_formats.py). Legacy rows read back as Markdown.
    artifact_format: str = "markdown"
    artifact_url: str | None = None
    artifact_link_label: str | None = None
    # Composite id ('alert:<n>' / 'run:<hex>') of the version this revised.
    supersedes_id: str | None = None
    # Whose document a drafted artifact is (orchestrator/artifact_records.py);
    # NULL on a draft = the principal's. Unused on every other source.
    owner_person_id: int | None = None
    # What the card needs from its reader (CARD_NEEDS): an answer
    # typed on the card, a pick of ``choices``, an OK to do one exact thing,
    # or nothing (for them to know). '' on rows written before, and on cards
    # that keep their own buttons. ``question`` is what an answer or pick is
    # to; the reply goes back into the work (POST /alerts/{id}/answer).
    needs: str = ""
    question: str = ""
    choices: list[str] = Field(default_factory=list)
    # The step Approve does, in a few words for its button ("Send the
    # reply"); "" keeps Approve (alerts.store.clean_label).
    action_label: str = ""
    # When the tidy-up settled this card on its own, and why
    # (``alerts.lifecycle.settle``). None / '' otherwise.
    tidied_at: str | None = None
    tidied_reason: str = ""


class UserPreferences(BaseModel):
    model_config = ConfigDict(extra="ignore")

    severity_threshold: AlertSeverity = AlertSeverity.MEDIUM
    quiet_hours_start: str = ""  # "22:00"
    quiet_hours_end: str = ""  # "07:00"
    # IANA zone the quiet hours are read in. "UTC" (the historical default,
    # and the column DEFAULT) and "" both mean "the user's zone" — see
    # alerts.preferences._quiet_hours_zone.
    quiet_hours_tz: str = "UTC"
    channels_enabled: list[AlertChannel] = Field(
        default_factory=lambda: [
            AlertChannel.WEB,
            AlertChannel.SLACK_DM,
            AlertChannel.EMAIL,
            AlertChannel.PERSISTED,
        ]
    )


class MuteTopic(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    pattern: str
    created_at: str


