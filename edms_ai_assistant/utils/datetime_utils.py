"""
EDMS AI Assistant — DateTime Utilities (FIXED VERSION).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, date, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _detect_system_timezone() -> timezone:
    """
    Автоматически определяет часовой пояс системы.
    """
    # ── 1. Переменная окружения TZ ────────────────────────────────
    env_tz = os.environ.get("TZ")
    if env_tz:
        try:
            if env_tz.startswith(("+", "-")):
                hours, minutes = _parse_offset(env_tz)
                return timezone(timedelta(hours=hours, minutes=minutes))
            else:
                return _get_local_timezone_no_dst()
        except Exception:
            pass

    # ── 2. Системный timezone (БЕЗ DST!) ─────────────────────────
    return _get_local_timezone_no_dst()


def _get_local_timezone_no_dst() -> timezone:
    """
    Определяет системный timezone, ИГНОРИРУЯ DST.
    """
    utc_offset_sec = time.timezone

    offset = timedelta(seconds=-utc_offset_sec)

    logger.debug(
        "System timezone detected (NO-DST mode)",
        extra={
            "offset_hours": offset.total_seconds() / 3600,
            "tz_name": time.tzname[0],  # Стандартное имя
            "raw_time_timezone": time.timezone,
            "raw_time_altzone": time.altzone,
            "raw_tm_isdst": time.localtime().tm_isdst,
            "note": "DST ignored to prevent +1h error",
        },
    )

    return timezone(offset)


def _parse_offset(offset_str: str) -> tuple[int, int]:
    """Парсит строку offset вида '+03:00'."""
    match = re.match(r"^([+-])(\d{1,2}):(\d{2})$", offset_str.strip())
    if not match:
        return 0, 0

    sign = 1 if match.group(1) == "+" else -1
    hours = int(match.group(2))
    minutes = int(match.group(3))

    return sign * hours, sign * minutes


# ══════════════════════════════════════════════════════════════════════════════
# КОНСТАНТЫ
# ══════════════════════════════════════════════════════════════════════════════

LOCAL_TZ: timezone = _detect_system_timezone()

_DB_TZ_OVERRIDE = os.environ.get("EDMS_DB_TIMEZONE")
if _DB_TZ_OVERRIDE:
    try:
        if _DB_TZ_OVERRIDE.startswith(("+", "-")):
            h, m = _parse_offset(_DB_TZ_OVERRIDE)
            DB_TZ = timezone(timedelta(hours=h, minutes=m))
        else:
            DB_TZ = LOCAL_TZ
        logger.info(f"DB timezone overridden from ENV: {DB_TZ}")
    except Exception:
        DB_TZ = LOCAL_TZ
else:
    DB_TZ = LOCAL_TZ

UTC_TZ = timezone.utc


def get_timezone_info() -> dict[str, Any]:
    """Информация о timezone для дебага."""
    now_aware = datetime.now(LOCAL_TZ)
    return {
        "local_tz": str(LOCAL_TZ),
        "local_offset_hours": LOCAL_TZ.utcoffset(now_aware).total_seconds() / 3600,
        "db_tz": str(DB_TZ),
        "db_offset_hours": DB_TZ.utcoffset(now_aware).total_seconds() / 3600,
        "system_tz_name": time.tzname[0],
        "system_alt_name": time.tzname[1] if len(time.tzname) > 1 else None,
        "is_dst": False,
        "env_tz": os.environ.get("TZ", "<not set>"),
        "current_time_local": now_aware.isoformat(),
        "current_time_utc": datetime.now(UTC_TZ).isoformat(),
        "raw_time_timezone_sec": time.timezone,
        "raw_time_altzone_sec": (
            time.altsize if hasattr(time, "altsize") else time.altzone
        ),
        "raw_tm_isdst": time.localtime().tm_isdst,
        "warning": "DST disabled to prevent +1h error",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Основные функции (без изменений)
# ══════════════════════════════════════════════════════════════════════════════


def to_local_timezone(dt: Any) -> str | None:
    """Конвертирует дату в строку ISO с локальным timezone."""
    if dt is None:
        return None

    if isinstance(dt, str):
        return _parse_string_to_local(dt)

    if isinstance(dt, datetime):
        return _datetime_to_local(dt)

    if isinstance(dt, date):
        return _date_to_local(dt)

    try:
        if hasattr(dt, "isoformat"):
            return to_local_timezone(str(dt))
    except Exception:
        pass

    logger.warning("Unknown date type: %s (%s)", type(dt).__name__, dt)
    return None


def now_local() -> datetime:
    """Текущее время в локальном TZ."""
    return datetime.now(LOCAL_TZ)


def today_local() -> date:
    """Сегодняшняя дата."""
    return now_local().date()


def start_of_day_local(dt=None) -> datetime:
    """Начало дня."""
    if dt is None:
        d = today_local()
    elif isinstance(dt, datetime):
        d = dt.date()
    elif isinstance(dt, date):
        d = dt
    else:
        parsed = to_local_timezone(dt)
        if not parsed:
            return datetime.now(LOCAL_TZ).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        d = datetime.fromisoformat(parsed).date()

    return datetime(d.year, d.month, d.day, tzinfo=LOCAL_TZ)


def end_of_day_local(dt=None) -> datetime:
    """Конец дня."""
    start = start_of_day_local(dt)
    return start.replace(hour=23, minute=59, second=59, microsecond=999999)


# ══════════════════════════════════════════════════════════════════════════════
# Хелперы
# ══════════════════════════════════════════════════════════════════════════════


def _parse_string_to_local(dt_str: str) -> str | None:
    """Парсит строку → локальный TZ."""
    if not dt_str or not dt_str.strip():
        return None

    dt_str = dt_str.strip().replace(" ", "T")

    if dt_str.endswith("Z"):
        utc_dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return utc_dt.astimezone(LOCAL_TZ).isoformat()

    if "+" in dt_str[10:] or (len(dt_str) > 6 and "-" in dt_str[10:]):
        try:
            dt_with_tz = datetime.fromisoformat(dt_str)
            return dt_with_tz.astimezone(LOCAL_TZ).isoformat()
        except ValueError:
            pass

    try:
        naive_dt = datetime.fromisoformat(dt_str)
        aware_dt = naive_dt.replace(tzinfo=DB_TZ)
        return aware_dt.astimezone(LOCAL_TZ).isoformat()
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y%m%d"):
        try:
            naive_dt = datetime.strptime(dt_str, fmt)
            aware_dt = naive_dt.replace(tzinfo=DB_TZ)
            return aware_dt.astimezone(LOCAL_TZ).isoformat()
        except ValueError:
            continue

    logger.debug("Could not parse date string: '%s'", dt_str)
    return dt_str


def _datetime_to_local(dt: datetime) -> str:
    """datetime → локальный TZ."""
    if dt.tzinfo is not None:
        return dt.astimezone(LOCAL_TZ).isoformat()
    else:
        return dt.replace(tzinfo=DB_TZ).astimezone(LOCAL_TZ).isoformat()


def _date_to_local(d: date) -> str:
    """date → локальный TZ."""
    dt = datetime(d.year, d.month, d.day, tzinfo=LOCAL_TZ)
    return dt.isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# Normalize / Encoder / Utils
# ══════════════════════════════════════════════════════════════════════════════

_DATE_FIELDS = {
    "regDate",
    "receiptDate",
    "createDate",
    "lastModifyDate",
    "dateDocCorrespondentOrg",
    "indexDateCoverLetter",
    "executionDate",
    "deadlineDate",
    "dateControlEnd",
    "dateControlStart",
    "dateOfActualExecution",
    "versionDate",
    "eventDate",
    "actionDate",
    "fromDate",
    "toDate",
    "startDate",
    "endDate",
    "createdAt",
    "updatedAt",
    "deletedAt",
}


def normalize_dates_in_dict(data, parent_key="", depth=0, max_depth=10):
    """Рекурсивная нормализация дат."""
    if depth > max_depth:
        return data

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            new_parent = f"{parent_key}.{key}" if parent_key else key
            if key in _DATE_FIELDS and value is not None:
                converted = to_local_timezone(value)
                result[key] = converted if converted is not None else value
            else:
                result[key] = normalize_dates_in_dict(
                    value, parent_key=new_parent, depth=depth + 1, max_depth=max_depth
                )
        return result

    if isinstance(data, list):
        return [
            normalize_dates_in_dict(
                item, parent_key=parent_key, depth=depth + 1, max_depth=max_depth
            )
            for item in data
        ]

    return data


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return to_local_timezone(obj)
        if isinstance(obj, date):
            return to_local_timezone(obj)
        if isinstance(obj, timezone):
            return str(obj)
        return super().default(obj)


def format_date_for_display(dt: Any, fmt: str = "%d.%m.%Y") -> str | None:
    iso = to_local_timezone(dt)
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).strftime(fmt)
    except (ValueError, TypeError):
        return iso


def is_today(dt: Any) -> bool:
    iso = to_local_timezone(dt)
    if not iso:
        return False
    try:
        return datetime.fromisoformat(iso).date() == today_local()
    except (ValueError, TypeError):
        return False


def days_diff(dt1: Any, dt2: Any | None = None) -> int:
    iso1 = to_local_timezone(dt1)
    iso2 = to_local_timezone(dt2) if dt2 else None
    if not iso1:
        return 0
    try:
        d1 = datetime.fromisoformat(iso1).date()
        d2 = datetime.fromisoformat(iso2).date() if iso2 else today_local()
        return abs((d1 - d2).days)
    except (ValueError, TypeError):
        return 0


# ══════════════════════════════════════════════════════════════════════════════
logger.info(
    "DateTimeUtils initialized (NO-DST mode)",
    extra={
        "local_tz": str(LOCAL_TZ),
        "db_tz": str(DB_TZ),
        "offset_hours": LOCAL_TZ.utcoffset(datetime.now()).total_seconds() / 3600,
        "warning": "DST auto-detection DISABLED to prevent +1h error",
    },
)
