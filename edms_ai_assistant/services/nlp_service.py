# edms_ai_assistant/services/nlp_service.py
"""
Semantic Dispatcher for EDMS AI Assistant

Transforms raw user input and documents into structured, LLM-ready context.
Follows Anthropic's methodology: structured outputs, clear intent, minimal prompting.

Architecture:
    User Input → [Intent Classification, Entity Extraction, Query Refinement] → Structured Context → LLM

Responsibilities:
    1. Extract ALL entities from DocumentDto (complete semantic decomposition)
    2. Classify user intent (analysis, creation, search, modification)
    3. Refine user queries (remove noise, extract core task)
    4. Build structured context for LLM consumption
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ========================================
# ENUMS & TYPES
# ========================================

class UserIntent(str, Enum):
    """Classification of user intent."""
    ANALYZE_DOCUMENT = "analyze_document"  # "проанализируй", "расскажи о документе"
    SUMMARIZE_CONTENT = "summarize_content"  # "сводка", "краткое содержание"
    SEARCH_EMPLOYEE = "search_employee"  # "найди сотрудника"
    CREATE_INTRODUCTION = "create_introduction"  # "добавь в ознакомление"
    CREATE_TASK = "create_task"  # "создай поручение"
    MODIFY_DOCUMENT = "modify_document"  # "измени", "обнови"
    GET_STATUS = "get_status"  # "какой статус", "где документ"
    UNKNOWN = "unknown"  # Не определено


class QueryComplexity(str, Enum):
    """Complexity level of user query."""
    SIMPLE = "simple"  # Single-step task
    MEDIUM = "medium"  # Multi-step, single tool
    COMPLEX = "complex"  # Multi-step, multiple tools


# ========================================
# STRUCTURED OUTPUTS (Pydantic Models)
# ========================================

class RefinedQuery(BaseModel):
    """Refined user query (noise removed, core task extracted)."""

    original: str = Field(..., description="Original user message")
    refined: str = Field(..., description="Refined core task (no politeness, no noise)")
    intent: UserIntent = Field(..., description="Classified intent")
    complexity: QueryComplexity = Field(..., description="Query complexity")
    extracted_entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Entities mentioned in query (names, dates, keywords)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "original": "Привет! Не мог бы ты, пожалуйста, найти для меня информацию о сотруднике Иванове?",
                "refined": "Найти сотрудника Иванов",
                "intent": "search_employee",
                "complexity": "simple",
                "extracted_entities": {"last_name": "Иванов"}
            }
        }


class DocumentContext(BaseModel):
    """Complete semantic decomposition of DocumentDto."""

    # Core Identity
    document_id: Optional[str] = None
    document_type: Optional[str] = None
    category: Optional[str] = None

    # Registration
    reg_number: Optional[str] = None
    reg_date: Optional[str] = None
    reserved_reg_number: Optional[str] = None

    # Content
    profile_name: Optional[str] = None
    short_summary: Optional[str] = None

    # Lifecycle
    status: Optional[str] = None
    prev_status: Optional[str] = None
    created_date: Optional[str] = None

    # People (Actors)
    author: Optional[Dict[str, str]] = None
    responsible_executor: Optional[Dict[str, str]] = None
    initiator: Optional[Dict[str, str]] = None
    controller: Optional[Dict[str, str]] = None
    addressees: List[Dict[str, str]] = Field(default_factory=list)

    # Tasks (Delegations)
    tasks: List[Dict[str, Any]] = Field(default_factory=list)

    # Process
    current_stage: Optional[str] = None
    process_completed: Optional[bool] = None
    process_stages: List[Dict[str, Any]] = Field(default_factory=list)

    # Attachments
    attachments: List[Dict[str, Any]] = Field(default_factory=list)

    # Type-Specific (Appeals, Meetings, Contracts)
    appeal_data: Optional[Dict[str, Any]] = None
    meeting_data: Optional[Dict[str, Any]] = None
    contract_data: Optional[Dict[str, Any]] = None

    # Relations
    linked_documents: Dict[str, Optional[str]] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "document_type": "INCOMING",
                "category": "APPEAL",
                "reg_number": "123/2026",
                "profile_name": "Обращение гражданина",
                "status": "IN_WORK",
                "author": {"name": "Иванов И.И.", "post": "Директор"}
            }
        }


class StructuredContext(BaseModel):
    """Final structured context for LLM."""

    query: RefinedQuery
    document: Optional[DocumentContext] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    def to_xml(self) -> str:
        """Convert to XML format (Anthropic-style)."""
        xml_parts = [
            "<context>",
            f"<timestamp>{self.timestamp}</timestamp>",
            "",
            "<query>",
            f"  <original>{self.query.original}</original>",
            f"  <refined>{self.query.refined}</refined>",
            f"  <intent>{self.query.intent.value}</intent>",
            f"  <complexity>{self.query.complexity.value}</complexity>",
            "</query>",
        ]

        if self.document:
            xml_parts.extend([
                "",
                "<document>",
                f"  <id>{self.document.document_id or 'N/A'}</id>",
                f"  <type>{self.document.document_type or 'N/A'}</type>",
                f"  <status>{self.document.status or 'N/A'}</status>",
                f"  <title>{self.document.profile_name or 'N/A'}</title>",
                "</document>",
            ])

        xml_parts.append("</context>")
        return "\n".join(xml_parts)


# ========================================
# SEMANTIC DISPATCHER
# ========================================

class SemanticDispatcher:
    """
    Semantic Dispatcher for EDMS Agent.

    Transforms raw inputs into structured, LLM-ready context.
    Follows Anthropic's methodology: clear, concise, actionable.
    """

    # Intent classification patterns
    INTENT_PATTERNS = {
        UserIntent.ANALYZE_DOCUMENT: [
            r"(проанализируй|анализ|расскажи о|опиши|что в|детали)",
            r"(какие поля|какая информация|что содержится)",
        ],
        UserIntent.SUMMARIZE_CONTENT: [
            r"(сводка|краткое содержание|суммаризируй|резюме|тезисы|пересказ)",
            r"(о чем документ|главное|ключевые моменты)",
        ],
        UserIntent.SEARCH_EMPLOYEE: [
            r"(найди сотрудника|кто такой|контакты|должность|информация о)",
            r"(сотрудник|работник|employee)",
        ],
        UserIntent.CREATE_INTRODUCTION: [
            r"(добавь в ознакомление|ознакомь|список ознакомления|направь на ознакомление)",
        ],
        UserIntent.CREATE_TASK: [
            r"(создай поручение|дай задание|назначь исполнителя|поручи|задача)",
        ],
        UserIntent.MODIFY_DOCUMENT: [
            r"(измени|обнови|поменяй|редактируй|исправь)",
        ],
        UserIntent.GET_STATUS: [
            r"(какой статус|где документ|на каком этапе|состояние)",
        ],
    }

    # Noise patterns (to remove)
    NOISE_PATTERNS = [
        r"^(привет|здравствуй|добрый день|добрый вечер|доброе утро)",
        r"(пожалуйста|будь добр|будь добра|если можешь|не мог бы)",
        r"(спасибо|благодарю|thanks)",
        r"\?+$",  # Multiple question marks
        r"!+$",  # Multiple exclamation marks
    ]

    def __init__(self):
        """Initialize Semantic Dispatcher."""
        logger.info("SemanticDispatcher initialized")

    # ========================================
    # 1. INTENT CLASSIFICATION
    # ========================================

    def classify_intent(self, message: str) -> UserIntent:
        """
        Classify user intent from message.

        Args:
            message: User message

        Returns:
            UserIntent enum
        """
        msg_lower = message.lower()

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, msg_lower, re.IGNORECASE):
                    logger.debug(f"Intent classified: {intent.value} (pattern: {pattern})")
                    return intent

        logger.debug("Intent classified: UNKNOWN")
        return UserIntent.UNKNOWN

    def assess_complexity(self, message: str, intent: UserIntent) -> QueryComplexity:
        """
        Assess query complexity.

        Args:
            message: User message
            intent: Classified intent

        Returns:
            QueryComplexity enum
        """
        # Simple checks
        word_count = len(message.split())
        has_multiple_requests = bool(re.search(r"(и|а также|кроме того|еще)", message, re.I))

        # Simple: short, single-intent, no conjunctions
        if word_count < 10 and not has_multiple_requests:
            return QueryComplexity.SIMPLE

        # Complex: multiple requests or complex intent
        if has_multiple_requests or intent in [UserIntent.CREATE_TASK, UserIntent.CREATE_INTRODUCTION]:
            return QueryComplexity.COMPLEX

        return QueryComplexity.MEDIUM

    # ========================================
    # 2. ENTITY EXTRACTION
    # ========================================

    def extract_query_entities(self, message: str) -> Dict[str, Any]:
        """
        Extract entities mentioned in user query.

        Args:
            message: User message

        Returns:
            Dictionary of extracted entities
        """
        entities = {}

        # Extract names (capitalized words)
        name_pattern = r'\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*)\b'
        names = re.findall(name_pattern, message)
        if names:
            entities["names"] = names

        # Extract dates
        date_pattern = r'\d{1,2}[./]\d{1,2}[./]\d{2,4}'
        dates = re.findall(date_pattern, message)
        if dates:
            entities["dates"] = dates

        # Extract keywords (5+ letter words, not common)
        common_words = {'привет', 'пожалуйста', 'спасибо', 'можешь', 'найти', 'создать'}
        words = re.findall(r'\b[а-яё]{5,}\b', message.lower())
        keywords = [w for w in words if w not in common_words]
        if keywords:
            entities["keywords"] = keywords[:5]  # Top 5

        return entities

    def extract_all_entities(self, doc: Any) -> DocumentContext:
        """
        Extract ALL entities from DocumentDto.

        Complete semantic decomposition following Anthropic's structured approach.

        Args:
            doc: DocumentDto instance

        Returns:
            DocumentContext with all extracted entities
        """
        try:
            # Core Identity
            context = DocumentContext(
                document_id=str(doc.id) if doc.id else None,
                document_type=self._get_safe(doc, "docType.value"),
                category=self._get_safe(doc, "docCategoryConstant.value"),
            )

            # Registration
            context.reg_number = doc.regNumber
            context.reg_date = doc.regDate.isoformat() if doc.regDate else None
            context.reserved_reg_number = doc.reservedRegNumber

            # Content
            context.profile_name = doc.profileName
            context.short_summary = doc.shortSummary

            # Lifecycle
            context.status = self._get_safe(doc, "status.value")
            context.prev_status = self._get_safe(doc, "prevStatus.value")
            context.created_date = doc.createDate.isoformat() if doc.createDate else None

            # People (Actors)
            context.author = self._format_person(doc.author)
            context.responsible_executor = self._format_person(doc.responsibleExecutor)
            context.initiator = self._format_person(doc.initiator)
            context.controller = self._format_person(self._get_safe(doc, "control.controlEmployee"))

            if doc.whoAddressed:
                context.addressees = [self._format_person(p) for p in doc.whoAddressed]

            # Tasks (Delegations)
            if doc.taskList:
                context.tasks = [self._extract_task(t) for t in doc.taskList]

            # Process
            context.current_stage = doc.currentBpmnTaskName
            context.process_completed = self._get_safe(doc, "process.completed")

            if doc.process and doc.process.items:
                context.process_stages = [
                    {
                        "name": item.name,
                        "completed": item.completed,
                        "order": item.order,
                    }
                    for item in doc.process.items
                ]

            # Attachments
            if doc.attachmentDocument:
                context.attachments = [self._extract_attachment(a) for a in doc.attachmentDocument]

            # Type-Specific Data
            if doc.documentAppeal:
                context.appeal_data = self._extract_appeal(doc.documentAppeal)

            if doc.dateMeeting:
                context.meeting_data = self._extract_meeting(doc)

            if doc.contractNumber:
                context.contract_data = self._extract_contract(doc)

            # Relations
            context.linked_documents = {
                "answer_to": str(doc.answerDocId) if doc.answerDocId else None,
                "received_doc": str(doc.receivedDocId) if doc.receivedDocId else None,
            }

            logger.debug(f"Extracted entities from document {context.document_id}")
            return context

        except Exception as e:
            logger.error(f"Entity extraction error: {e}", exc_info=True)
            return DocumentContext(document_id="extraction_failed")

    # ========================================
    # 3. QUERY REFINEMENT
    # ========================================

    def refine_query(self, message: str) -> RefinedQuery:
        """
        Refine user query: remove noise, extract core task.

        Anthropic methodology: clear, concise, actionable.

        Args:
            message: Original user message

        Returns:
            RefinedQuery with cleaned task
        """
        original = message
        refined = message

        # Step 1: Remove noise patterns
        for pattern in self.NOISE_PATTERNS:
            refined = re.sub(pattern, "", refined, flags=re.IGNORECASE).strip()

        # Step 2: Remove extra whitespace
        refined = re.sub(r'\s+', ' ', refined).strip()

        # Step 3: Capitalize first letter
        if refined:
            refined = refined[0].upper() + refined[1:]

        # Step 4: Classify intent
        intent = self.classify_intent(refined)

        # Step 5: Assess complexity
        complexity = self.assess_complexity(refined, intent)

        # Step 6: Extract entities
        entities = self.extract_query_entities(refined)

        result = RefinedQuery(
            original=original,
            refined=refined,
            intent=intent,
            complexity=complexity,
            extracted_entities=entities
        )

        logger.debug(
            f"Query refined: '{original}' → '{refined}' (intent={intent.value}, complexity={complexity.value})")
        return result

    # ========================================
    # 4. CONTEXT BUILDING
    # ========================================

    def build_context(
            self,
            message: str,
            document: Optional[Any] = None
    ) -> StructuredContext:
        """
        Build complete structured context for LLM.

        Args:
            message: User message
            document: Optional DocumentDto instance

        Returns:
            StructuredContext ready for LLM consumption
        """
        # Refine query
        refined_query = self.refine_query(message)

        # Extract document entities (if provided)
        doc_context = None
        if document:
            doc_context = self.extract_all_entities(document)

        # Build final context
        context = StructuredContext(
            query=refined_query,
            document=doc_context
        )

        logger.info(
            f"Context built: intent={refined_query.intent.value}, "
            f"complexity={refined_query.complexity.value}, "
            f"has_document={bool(doc_context)}"
        )

        return context

    # ========================================
    # HELPER METHODS
    # ========================================

    def _get_safe(self, obj: Any, path: str, default: Any = None) -> Any:
        """Safely get nested attribute."""
        val = obj
        for part in path.split("."):
            if val is None:
                return default
            val = getattr(val, part, None) if hasattr(val, part) else val.get(part, default) if isinstance(val,
                                                                                                           dict) else default

        if hasattr(val, "value"):
            return val.value
        return val if val is not None else default

    def _format_person(self, person: Any) -> Optional[Dict[str, str]]:
        """Format person entity."""
        if not person:
            return None

        return {
            "name": f"{getattr(person, 'lastName', '')} {getattr(person, 'firstName', '')} {getattr(person, 'middleName', '')}".strip(),
            "post": self._get_safe(person, "post.postName") or self._get_safe(person, "authorPost", ""),
            "id": str(person.id) if hasattr(person, "id") and person.id else None,
        }

    def _extract_task(self, task: Any) -> Dict[str, Any]:
        """Extract task entity."""
        return {
            "number": task.taskNumber,
            "text": task.taskText,
            "executor": self._format_person(task.author),
            "deadline": task.planedDateEnd.isoformat() if task.planedDateEnd else None,
            "status": self._get_safe(task, "taskStatus.value"),
            "on_control": task.onControl,
        }

    def _extract_attachment(self, attachment: Any) -> Dict[str, Any]:
        """Extract attachment entity."""
        return {
            "id": str(attachment.id) if attachment.id else None,
            "name": attachment.name,
            "type": self._get_safe(attachment, "attachmentDocumentType.name"),
            "size_kb": round(attachment.size / 1024, 2) if attachment.size else 0,
            "upload_date": attachment.uploadDate.isoformat() if attachment.uploadDate else None,
            "has_signature": bool(attachment.signs and len(attachment.signs) > 0),
        }

    def _extract_appeal(self, appeal: Any) -> Dict[str, Any]:
        """Extract appeal-specific data."""
        return {
            "applicant": appeal.fioApplicant,
            "type": "Коллективное" if appeal.collective else "Индивидуальное",
            "address": f"{appeal.regionName or ''}, {appeal.cityName or ''}, {appeal.fullAddress or ''}".strip(", "),
            "subject": self._get_safe(appeal, "subject.name"),
            "solution_result": self._get_safe(appeal, "solutionResult.name"),
        }

    def _extract_meeting(self, doc: Any) -> Dict[str, Any]:
        """Extract meeting-specific data."""
        return {
            "date": doc.dateMeeting.isoformat() if doc.dateMeeting else None,
            "time": doc.startMeeting.isoformat() if doc.startMeeting else None,
            "place": doc.placeMeeting,
            "chairperson": self._format_person(doc.chairperson),
            "secretary": self._format_person(doc.secretary),
            "questions": [q.question for q in (doc.documentQuestions or [])],
        }

    def _extract_contract(self, doc: Any) -> Dict[str, Any]:
        """Extract contract-specific data."""
        return {
            "number": doc.contractNumber,
            "amount": f"{doc.contractSum} {self._get_safe(doc, 'currency.currencyName')}",
            "start_date": doc.contractStartDate.isoformat() if doc.contractStartDate else None,
            "end_date": doc.contractDurationEnd.isoformat() if doc.contractDurationEnd else None,
            "auto_prolong": doc.contractAutoProlongation,
        }


# ========================================
# LEGACY COMPATIBILITY (for backward compatibility)
# ========================================

class EDMSNaturalLanguageService:
    """
    Legacy NLP Service (backward compatibility).

    Delegates to SemanticDispatcher for new functionality.
    """

    def __init__(self):
        self.dispatcher = SemanticDispatcher()

    @staticmethod
    def format_user(user: Any) -> Optional[str]:
        """Legacy method: format user."""
        if not user:
            return None
        ln = getattr(user, "lastName", "") or ""
        fn = getattr(user, "firstName", "") or ""
        mn = getattr(user, "middleName", "") or ""
        post = getattr(user, "authorPost", "") or getattr(
            getattr(user, "post", None), "name", ""
        )
        name = f"{ln} {fn} {mn}".strip()
        return f"{name} ({post})" if post else name

    def get_safe(self, obj: Any, path: str, default: Any = None) -> Any:
        """Legacy method: get safe."""
        return self.dispatcher._get_safe(obj, path, default)

    def process_document(self, doc: Any) -> Dict[str, Any]:
        """
        Legacy method: process document.

        Now uses SemanticDispatcher for extraction.
        """
        context = self.dispatcher.extract_all_entities(doc)
        return context.model_dump()

    def process_employee_info(self, emp: Any) -> Dict[str, Any]:
        """Legacy method: process employee."""
        full_name = self.format_user(emp)

        return {
            "основное": {
                "фио": full_name,
                "должность": self.get_safe(emp, "post.postName"),
                "департамент": self.get_safe(emp, "department.name"),
                "статус": "Уволен" if emp.fired else "Активен",
                "является_ио": emp.io,
            },
            "контакты": {
                "email": emp.email,
                "телефон": emp.phone,
                "адрес": emp.address,
                "площадка": emp.place,
            },
        }