import pytest
import jwt
from edms_ai_assistant.security import extract_user_id_from_token
from edms_ai_assistant.config import settings

def test_extract_user_id_from_token_valid():
    payload = {"id": "user123"}
    token = jwt.encode(payload, settings.JWT_SECRET_KEY.get_secret_value(), algorithm=settings.JWT_ALGORITHM)
    assert extract_user_id_from_token(token) == "user123"

def test_extract_user_id_from_token_unverified_debug():
    # Force DEBUG = True for this test
    original_debug = settings.DEBUG
    settings.DEBUG = True
    try:
        payload = {"id": "user456"}
        # Encode with a different key
        token = jwt.encode(payload, "wrong-key", algorithm=settings.JWT_ALGORITHM)
        # Should work in debug mode
        assert extract_user_id_from_token(token) == "user456"
    finally:
        settings.DEBUG = original_debug

def test_extract_user_id_from_token_invalid_no_debug():
    # Force DEBUG = False for this test
    original_debug = settings.DEBUG
    settings.DEBUG = False
    try:
        payload = {"id": "user456"}
        token = jwt.encode(payload, "wrong-key", algorithm=settings.JWT_ALGORITHM)
        with pytest.raises(ValueError, match="Невалидный токен"):
            extract_user_id_from_token(token)
    finally:
        settings.DEBUG = original_debug
