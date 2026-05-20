from enum import StrEnum


# ══════════════════════════════════════════════════════════════════════════════
# Document & Process Enums
# ══════════════════════════════════════════════════════════════════════════════

class DocCategory(StrEnum):
    INTERN = "INTERN"
    INCOMING = "INCOMING"
    OUTGOING = "OUTGOING"
    MEETING = "MEETING"
    QUESTION = "QUESTION"
    MEETING_QUESTION = "MEETING_QUESTION"
    APPEAL = "APPEAL"
    CONTRACT = "CONTRACT"
    CUSTOM = "CUSTOM"


class DocumentStatus(StrEnum):  # Status2 в OpenAPI
    DRAFT = "DRAFT"
    NEW = "NEW"
    STATEMENT = "STATEMENT"
    APPROVED = "APPROVED"
    SIGNING = "SIGNING"
    SIGNED = "SIGNED"
    AGREEMENT = "AGREEMENT"
    AGREED = "AGREED"
    REVIEW = "REVIEW"
    REVIEWED = "REVIEWED"
    REGISTRATION = "REGISTRATION"
    REGISTERED = "REGISTERED"
    EXECUTION = "EXECUTION"
    EXECUTED = "EXECUTED"
    DISPATCH = "DISPATCH"
    SENT = "SENT"
    REJECT = "REJECT"
    CANCEL = "CANCEL"
    PREPARATION = "PREPARATION"
    PAPERWORK = "PAPERWORK"
    FORMALIZED = "FORMALIZED"
    ACCEPTANCE = "ACCEPTANCE"
    ACCEPTED = "ACCEPTED"
    CONTRACT_EXECUTION = "CONTRACT_EXECUTION"
    CONTRACT_CLOSED = "CONTRACT_CLOSED"
    ARCHIVE = "ARCHIVE"
    DELETED = "DELETED"
    ALL = "ALL"


class DocumentProcessType(StrEnum):  # Type4 в OpenAPI
    NEW = "NEW"
    AGREEMENT = "AGREEMENT"
    SIGNING = "SIGNING"
    STATEMENT = "STATEMENT"
    REGISTRATION = "REGISTRATION"
    REVIEW = "REVIEW"
    EXECUTION = "EXECUTION"
    DISPATCH = "DISPATCH"
    PREPARATION = "PREPARATION"
    PAPERWORK = "PAPERWORK"
    ACCEPTANCE = "ACCEPTANCE"
    CONTRACT_EXECUTION = "CONTRACT_EXECUTION"


class JobStatus(StrEnum):
    NEW = "NEW"
    AWAITING_CONVERSION = "AWAITING_CONVERSION"
    READY = "READY"
    IN_WORK_CONVERSION = "IN_WORK_CONVERSION"
    COMPLETED = "COMPLETED"
    SIGNED = "SIGNED"
    DISPATCH = "DISPATCH"
    IN_WORK_DISPATCH = "IN_WORK_DISPATCH"
    SENT = "SENT"
    ERROR_CONVERSION = "ERROR_CONVERSION"
    ERROR_DISPATCH = "ERROR_DISPATCH"


class CreateType(StrEnum):
    MANUAL = "MANUAL"
    AISMV = "AISMV"
    DIRECTUM = "DIRECTUM"


class AttachmentType(StrEnum):  # Type1 в OpenAPI
    MAIN_ATTACHMENT = "MAIN_ATTACHMENT"
    ADDITIONAL_ATTACHMENT = "ADDITIONAL_ATTACHMENT"


class AttachmentDocumentType(StrEnum):
    ATTACHMENT = "ATTACHMENT"
    PRINT_DOCUMENT = "PRINT_DOCUMENT"
    PROJECT_SOLUTION = "PROJECT_SOLUTION"
    RATIONALE = "RATIONALE"
    DOCUMENTS_QUESTION = "DOCUMENTS_QUESTION"
    INTRODUCTION_LIST = "INTRODUCTION_LIST"
    AGREEMENT_LIST = "AGREEMENT_LIST"
    DECISION = "DECISION"
    RKK = "RKK"


# ══════════════════════════════════════════════════════════════════════════════
# Nomenclature & Archive Enums
# ══════════════════════════════════════════════════════════════════════════════

class StoreAttribute(StrEnum):
    PERMANENT = "PERMANENT"
    TEMPORARY = "TEMPORARY"
    END_OF_NEED = "END_OF_NEED"
    BY_STAFF = "BY_STAFF"


class StorageType(StrEnum):
    ARCHIVE = "ARCHIVE"
    OPERATIONAL = "OPERATIONAL"


class MethodAssemblyType(StrEnum):
    ED = "ED"
    GD = "GD"
    PAPER = "PAPER"
    IR = "IR"


class GroupByStoragePeriod(StrEnum):
    S_T_UP_TO_10_Y = "S_T_UP_TO_10_Y"
    S_T_OVER_10_Y = "S_T_OVER_10_Y"
    P_S = "P_S"


class YearPostfix(StrEnum):
    YEAR_1 = "год"
    YEAR_2 = "года"
    YEAR_5 = "лет"


class StoragePeriodType(StrEnum):
    NUMERIC = "Числовой"
    PERMANENT = "Постоянно"
    UNTIL_NEEDED = "До минования надобности"
    UNTIL_REPLACED = "До замены новыми"


class NomenclatureStatus(StrEnum):  # Status6
    FORMING = "FORMING"
    COMPLETED = "COMPLETED"
    CLOSED = "CLOSED"
    ARCHIVE = "ARCHIVE"
    DESTROYED = "DESTROYED"


class NomenclatureType(StrEnum):
    MAIN = "MAIN"
    ADDITIONAL = "ADDITIONAL"


class ExpertiseType(StrEnum):
    EPK = "EPK"
    CEK = "CEK"
    EK = "EK"


# ══════════════════════════════════════════════════════════════════════════════
# Employee & Access Enums
# ══════════════════════════════════════════════════════════════════════════════

class RoleObjectType(StrEnum):  # Type14
    DOCUMENT = "DOCUMENT"
    TASK = "TASK"
    TASK_PROJECT = "TASK_PROJECT"
    NOMENCLATURE_DEPARTMENT = "NOMENCLATURE_DEPARTMENT"
    SUMMARY_NOMENCLATURE_DEPARTMENT = "SUMMARY_NOMENCLATURE_DEPARTMENT"
    DESTRUCTION_ACT = "DESTRUCTION_ACT"
    NOMENCLATURE_AFFAIR = "NOMENCLATURE_AFFAIR"
    MINI_DOCUMENT = "MINI_DOCUMENT"
    SYSTEM = "SYSTEM"
    ACCEPTANCE_INVENTORY = "ACCEPTANCE_INVENTORY"
    CUSTOM = "CUSTOM"
    ADDITIONAL_DOCUMENT = "ADDITIONAL_DOCUMENT"
    CONTRACT = "CONTRACT"


class ActionTypeQueue(StrEnum):
    ANY = "ANY"
    ORDERED = "ORDERED"


class GroupType(StrEnum):  # Type7
    ACCESS = "ACCESS"
    DISTRIBUTION = "DISTRIBUTION"


class BlockedField(StrEnum):
    FIRST_NAME = "FIRST_NAME"
    LAST_NAME = "LAST_NAME"
    MIDDLE_NAME = "MIDDLE_NAME"
    POST = "POST"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    PHOTO = "PHOTO"
    DEPARTMENT = "DEPARTMENT"


class CorrespondentType(StrEnum):  # Type2
    NORMAL = "NORMAL"
    AISMV = "AISMV"
    DIRECTUM = "DIRECTUM"
    GTB_ORG = "GTB_ORG"


class DeclarantType(StrEnum):
    INDIVIDUAL = "INDIVIDUAL"
    ENTITY = "ENTITY"


# ══════════════════════════════════════════════════════════════════════════════
# Task & Report Enums
# ══════════════════════════════════════════════════════════════════════════════

class TaskStatus(StrEnum):
    ON_EXECUTION = "ON_EXECUTION"
    EXECUTED = "EXECUTED"


class TaskType(StrEnum):  # Type12
    INFORMATION = "INFORMATION"
    GENERAL = "GENERAL"


class Period(StrEnum):
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    THREE_MONTHS = "THREE_MONTHS"
    SIX_MONTHS = "SIX_MONTHS"
    YEAR = "YEAR"


class PanelType(StrEnum):
    ACTIVE_USER = "ACTIVE_USER"
    DOCUMENT_CREATE = "DOCUMENT_CREATE"
    DOCUMENT_REGISTERED = "DOCUMENT_REGISTERED"


class ReportType(StrEnum):  # Type16
    DOC = "DOC"
    TASK = "TASK"


class ReportColumn(StrEnum):
    DOC_PROFILE = "DOC_PROFILE"
    DOC_TYPE = "DOC_TYPE"
    DOC_CATEGORY = "DOC_CATEGORY"
    DATE_REG = "DATE_REG"
    AUTHOR_DOC = "AUTHOR_DOC"
    REG_NUMBER = "REG_NUMBER"
    DESTINATION = "DESTINATION"
    CORRESPONDENT = "CORRESPONDENT"
    OUTGOING_NUMBER = "OUTGOING_NUMBER"
    OUTGOING_DATE = "OUTGOING_DATE"
    SIGNED_BY = "SIGNED_BY"
    STATUS = "STATUS"
    EXECUTOR_STAGE = "EXECUTOR_STAGE"
    DATE_STAGE = "DATE_STAGE"
    AUTHOR_TASK = "AUTHOR_TASK"
    TASK_EXECUTOR = "TASK_EXECUTOR"
    TASK_LEADER = "TASK_LEADER"
    TASK_NUMBER = "TASK_NUMBER"
    DOC_SHORT_SUMMARY = "DOC_SHORT_SUMMARY"
    DATE_TASK_EXECUTE = "DATE_TASK_EXECUTE"
    TASK_ON_CONTROL = "TASK_ON_CONTROL"
    DOC_ON_CONTROL = "DOC_ON_CONTROL"
    DOC_DATE_CONTROL = "DOC_DATE_CONTROL"
    REPORT_TASK = "REPORT_TASK"
    TASK_SHORT_SUMMARY = "TASK_SHORT_SUMMARY"
    CONTRACT_SIGNING_DATE = "CONTRACT_SIGNING_DATE"
    CONTRACT_START_DATE = "CONTRACT_START_DATE"
    CONTRACT_DAYS_TO_COMPLETION = "CONTRACT_DAYS_TO_COMPLETION"
    CONTRACT_SUM = "CONTRACT_SUM"
    CURRENCY = "CURRENCY"
    CONTRACT_RECIPIENT = "CONTRACT_RECIPIENT"


# ══════════════════════════════════════════════════════════════════════════════
# Notification Enums
# ══════════════════════════════════════════════════════════════════════════════

class ReminderType(StrEnum):  # Id в RemindersRulesDto
    DEADLINE_CONTROL_DOCUMENT = "DEADLINE_CONTROL_DOCUMENT"
    DEADLINE_CONTROL_TASK = "DEADLINE_CONTROL_TASK"
    DEADLINE_EXECUTION_PROCESS = "DEADLINE_EXECUTION_PROCESS"
    DEADLINE_EXECUTION_TASK = "DEADLINE_EXECUTION_TASK"
    INACTION_SUMMARY_NOMENCLATURE = "INACTION_SUMMARY_NOMENCLATURE"
    INACTION_NOMENCLATURE_DEPARTMENT = "INACTION_NOMENCLATURE_DEPARTMENT"
    INACTION_DESTRUCTION_ACT = "INACTION_DESTRUCTION_ACT"
    INACTION_ACCEPTANCE_INVENTORY = "INACTION_ACCEPTANCE_INVENTORY"
    DEADLINE_CONTRACT_DURATION_END = "DEADLINE_CONTRACT_DURATION_END"
    DEADLINE_EXECUTION_DOCUMENT_CONTROL_POINT = "DEADLINE_EXECUTION_DOCUMENT_CONTROL_POINT"
