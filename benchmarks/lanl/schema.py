"""Canonical schema helpers for the LANL authentication benchmark slice."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class CanonicalField:
    """Maps a semantic field to expected LANL columns."""

    name: str
    aliases: tuple[str, ...]
    required: bool = False


@dataclass(slots=True)
class LanlAuthSchema:
    """Resolved schema for LANL auth and red-team files."""

    dataset_name: str = "lanl_auth"
    source_file: str = ""
    redteam_file: str = ""
    has_header: bool = False
    time: str = "time"
    src_user: str = "src_user"
    dst_user: str = "dst_user"
    src_computer: str = "src_computer"
    dst_computer: str = "dst_computer"
    auth_type: str = "auth_type"
    logon_type: str = "logon_type"
    auth_orientation: str = "auth_orientation"
    success: str = "success"
    columns: dict[str, str] = field(default_factory=dict)
    redteam_time: str = "time"
    redteam_user: str = "src_user"
    redteam_src_computer: str = "src_computer"
    redteam_dst_computer: str = "dst_computer"


AUTH_TIME_FIELD = CanonicalField("time", ("time", "timestamp"), required=True)
AUTH_SRC_USER_FIELD = CanonicalField("src_user", ("src_user", "source_user", "user"), required=True)
AUTH_DST_USER_FIELD = CanonicalField("dst_user", ("dst_user", "destination_user", "target_user"), required=True)
AUTH_SRC_COMPUTER_FIELD = CanonicalField("src_computer", ("src_computer", "source_computer"), required=True)
AUTH_DST_COMPUTER_FIELD = CanonicalField("dst_computer", ("dst_computer", "destination_computer"), required=True)
AUTH_TYPE_FIELD = CanonicalField("auth_type", ("auth_type", "authentication_type"), required=True)
LOGON_TYPE_FIELD = CanonicalField("logon_type", ("logon_type",), required=True)
AUTH_ORIENTATION_FIELD = CanonicalField("auth_orientation", ("auth_orientation", "orientation"), required=True)
SUCCESS_FIELD = CanonicalField("success", ("success", "result"), required=True)

AUTH_FIELDS: tuple[CanonicalField, ...] = (
    AUTH_TIME_FIELD,
    AUTH_SRC_USER_FIELD,
    AUTH_DST_USER_FIELD,
    AUTH_SRC_COMPUTER_FIELD,
    AUTH_DST_COMPUTER_FIELD,
    AUTH_TYPE_FIELD,
    LOGON_TYPE_FIELD,
    AUTH_ORIENTATION_FIELD,
    SUCCESS_FIELD,
)


def default_auth_schema(source_file: str = "", redteam_file: str = "") -> LanlAuthSchema:
    """Return the default schema for official LANL auth and red-team files."""

    schema = LanlAuthSchema(source_file=source_file, redteam_file=redteam_file)
    schema.columns = {
        "time": schema.time,
        "src_user": schema.src_user,
        "dst_user": schema.dst_user,
        "src_computer": schema.src_computer,
        "dst_computer": schema.dst_computer,
        "auth_type": schema.auth_type,
        "logon_type": schema.logon_type,
        "auth_orientation": schema.auth_orientation,
        "success": schema.success,
    }
    return schema


def detect_auth_schema(
    columns: Iterable[str],
    *,
    source_file: str = "",
    redteam_file: str = "",
    dataset_name: str = "lanl_auth",
) -> LanlAuthSchema:
    """Resolve a LANL auth schema from header names when a header is present."""

    normalized = {normalize_column_name(column): column for column in columns}
    schema = default_auth_schema(source_file=source_file, redteam_file=redteam_file)
    schema.dataset_name = dataset_name
    schema.has_header = True
    for field in AUTH_FIELDS:
        for alias in field.aliases:
            key = normalize_column_name(alias)
            if key in normalized:
                setattr(schema, field.name, normalized[key])
                break
    schema.columns = {
        "time": schema.time,
        "src_user": schema.src_user,
        "dst_user": schema.dst_user,
        "src_computer": schema.src_computer,
        "dst_computer": schema.dst_computer,
        "auth_type": schema.auth_type,
        "logon_type": schema.logon_type,
        "auth_orientation": schema.auth_orientation,
        "success": schema.success,
    }
    return schema


def normalize_column_name(value: str) -> str:
    """Normalize a potential header for fuzzy matching."""

    cleaned = value.strip().lower()
    for char in ("-", "/", ".", ",", "(", ")"):
        cleaned = cleaned.replace(char, " ")
    return "_".join(cleaned.split())

