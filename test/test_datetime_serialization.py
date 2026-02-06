# test/test_datetime_serialization.py
import json
import pytest
from datetime import datetime, timezone
from uuid import uuid4
from edms_ai_assistant.utils.json_encoder import CustomJSONEncoder


def test_datetime_to_java_instant():
    """
    Test that Python datetime serializes to format compatible with java.time.Instant.

    Java expects: "2026-02-05T14:34:28.123Z" (ISO 8601 with 'Z' timezone)
    """
    # Test 1: UTC datetime with timezone info
    dt_utc = datetime(2026, 2, 5, 14, 34, 28, 123000, tzinfo=timezone.utc)
    json_str = json.dumps({"timestamp": dt_utc}, cls=CustomJSONEncoder)

    assert "2026-02-05T14:34:28.123000+00:00" in json_str or "2026-02-05T14:34:28.123Z" in json_str
    print(f"✅ UTC datetime: {json_str}")

    # Test 2: Naive datetime (should add 'Z')
    dt_naive = datetime(2026, 2, 5, 14, 34, 28, 123000)
    json_str_naive = json.dumps({"timestamp": dt_naive}, cls=CustomJSONEncoder)

    assert "2026-02-05T14:34:28.123000Z" in json_str_naive
    print(f"✅ Naive datetime: {json_str_naive}")

    # Test 3: Parse back in Python
    parsed = json.loads(json_str_naive)
    assert "Z" in parsed["timestamp"]  # Ensure timezone marker is present


def test_uuid_serialization():
    """Test UUID serialization to string."""
    test_uuid = uuid4()
    json_str = json.dumps({"id": test_uuid}, cls=CustomJSONEncoder)

    parsed = json.loads(json_str)
    assert isinstance(parsed["id"], str)
    assert len(parsed["id"]) == 36  # Standard UUID string length

    print(f"✅ UUID serialization: {test_uuid} → {parsed['id']}")


def test_mixed_types():
    """Test serialization of mixed special types."""
    from enum import Enum

    class Status(Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    data = {
        "id": uuid4(),
        "created_at": datetime.now(timezone.utc),
        "status": Status.ACTIVE,
        "employee_ids": [uuid4(), uuid4()],
    }

    json_str = json.dumps(data, cls=CustomJSONEncoder)
    parsed = json.loads(json_str)

    assert isinstance(parsed["id"], str)
    assert isinstance(parsed["status"], str)
    assert parsed["status"] == "active"
    assert len(parsed["employee_ids"]) == 2

    print(f"✅ Mixed types serialization passed")


def test_java_instant_compatibility():
    """
    Test format compatibility with Java backend expectations.

    Java code from EmployeeController:
    - Instant.parse("2026-02-05T14:34:28.123Z")
    - Requires 'Z' suffix or '+00:00' offset
    """
    test_cases = [
        datetime(2026, 2, 5, 14, 34, 28, 123000),  # Naive
        datetime(2026, 2, 5, 14, 34, 28, 123000, tzinfo=timezone.utc),  # UTC
    ]

    for dt in test_cases:
        json_str = json.dumps({"instant": dt}, cls=CustomJSONEncoder)
        parsed = json.loads(json_str)

        # Must have timezone info (Z or +00:00)
        assert "Z" in parsed["instant"] or "+00:00" in parsed["instant"]

        print(f"✅ Java Instant compatible: {parsed['instant']}")


if __name__ == "__main__":
    test_datetime_to_java_instant()
    test_uuid_serialization()
    test_mixed_types()
    test_java_instant_compatibility()
    print("\n🎉 All datetime serialization tests passed!")