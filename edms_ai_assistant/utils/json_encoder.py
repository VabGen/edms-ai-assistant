# edms_ai_assistant/utils/json_encoder.py
import json
from uuid import UUID
from datetime import datetime
from enum import Enum


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'model_dump'):
            return obj.model_dump(mode='json')
        if hasattr(obj, 'dict'):
            return obj.dict()

        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value

        return json.JSONEncoder.default(self, obj)
