import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EDMSNaturalLanguageService:
    @staticmethod
    def format_user(user: Any) -> Optional[str]:
        """Универсальное форматирование UserInfoDto или EmployeeDto."""
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
        val = obj
        for part in path.split("."):
            if val is None:
                return default
            val = (
                val.get(part, default)
                if isinstance(val, dict)
                else getattr(val, part, default)
            )
        if hasattr(val, "value"):
            return val.value
        return val if val is not None else default

    def process_document(self, doc: Any) -> Dict[str, Any]:
        category = self.get_safe(doc, "docCategoryConstant")

        # 1. СУБЪЕКТЫ (Люди и организации)
        participants = {
            "автор": self.format_user(doc.author),
            "ответственный_исполнитель": self.format_user(doc.responsibleExecutor),
            "подписанты": [self.format_user(u) for u in (doc.whoAddressed or [])],
            "инициатор": self.format_user(doc.initiator),
            "контролер": self.format_user(
                self.get_safe(doc, "control.controlEmployee")
            ),
        }

        # 2. ЖИЗНЕННЫЙ ЦИКЛ (Статусы и Процессы)
        workflow = {
            "текущий_статус": self.get_safe(doc, "status"),
            "предыдущий_статус": self.get_safe(doc, "prevStatus"),
            "этап_БП": doc.currentBpmnTaskName,
            "процесс_завершен": self.get_safe(doc, "process.completed"),
            "этапы_маршрута": [
                {
                    "этап": item.name,
                    "статус": "Выполнен" if item.completed else "В работе",
                }
                for item in (self.get_safe(doc, "process.items") or [])
            ],
        }

        # 3. ПОРУЧЕНИЯ (TaskDto)
        tasks = []
        if doc.taskList:
            for t in doc.taskList:
                tasks.append(
                    {
                        "номер": t.taskNumber,
                        "текст": t.taskText,
                        "исполнитель": self.format_user(
                            t.author
                        ),  # У TaskDto часто автор важен
                        "срок": (
                            t.planedDateEnd.strftime("%d.%m.%Y")
                            if t.planedDateEnd
                            else "Бессрочно"
                        ),
                        "статус": self.get_safe(t, "taskStatus"),
                        "на_контроле": t.onControl,
                    }
                )

        # 4. СПЕЦИФИКАЦИЯ КАТЕГОРИИ
        details = {}
        # Обращения
        if doc.documentAppeal:
            app = doc.documentAppeal
            details["обращение"] = {
                "заявитель": app.fioApplicant,
                "тип": "Коллективное" if app.collective else "Индивидуальное",
                "адрес": f"{app.regionName or ''}, {app.cityName or ''}, {app.fullAddress or ''}".strip(
                    ", "
                ),
                "тематика": self.get_safe(app, "subject.name"),
                "результат_решения": self.get_safe(app, "solutionResult.name"),
            }

        # Совещания
        if doc.dateMeeting or category in ["MEETING", "QUESTION"]:
            details["совещание"] = {
                "дата_время": f"{doc.dateMeeting.strftime('%d.%m.%Y') if doc.dateMeeting else ''} {doc.startMeeting.strftime('%H:%M') if doc.startMeeting else ''}",
                "место": doc.placeMeeting,
                "председатель": self.format_user(doc.chairperson),
                "секретарь": self.format_user(doc.secretary),
                "вопросы_повестки": [q.question for q in (doc.documentQuestions or [])],
            }

        # Договоры
        if doc.contractNumber or category == "CONTRACT":
            details["финансы"] = {
                "номер": doc.contractNumber,
                "сумма": f"{doc.contractSum} {self.get_safe(doc, 'currency.currencyName')}",
                "период": f"{doc.contractStartDate.strftime('%d.%m.%Y') if doc.contractStartDate else ''} - {doc.contractDurationEnd.strftime('%d.%m.%Y') if doc.contractDurationEnd else ''}",
                "пролонгация": doc.contractAutoProlongation,
            }

        # 5. СВЯЗИ И ВЛОЖЕНИЯ
        links = {
            "вложения": [
                {"имя": a.name, "id": str(a.id)} for a in (doc.attachmentDocument or [])
            ],
            "связанные_id": {
                "в_ответ_на": str(doc.answerDocId) if doc.answerDocId else None,
                "ответный_док": str(doc.receivedDocId) if doc.receivedDocId else None,
            },
        }

        return {
            "мета": {
                "id": str(doc.id),
                "номер": doc.regNumber or doc.reservedRegNumber,
                "заголовок": doc.profileName,
                "содержание": doc.shortSummary,
            },
            "участники": participants,
            "жизненный_цикл": workflow,
            "задачи": tasks,
            "детали_типа": details,
            "связи": links,
        }

    def analyze_attachment_meta(self, attachment: Any) -> Dict[str, Any]:
        """Анализирует метаданные конкретного файла."""
        return {
            "название": attachment.name,
            "тип_вложения": self.get_safe(attachment, "attachmentDocumentType.name"),
            "размер_кб": round(attachment.size / 1024, 2) if attachment.size else 0,
            "дата_загрузки": (
                attachment.uploadDate.strftime("%d.%m.%Y %H:%M")
                if attachment.uploadDate
                else None
            ),
            "есть_эцп": bool(attachment.signs and len(attachment.signs) > 0),
            "автор_id": str(attachment.authorId) if attachment.authorId else None,
        }

    def analyze_local_file(self, file_path: str) -> Dict[str, Any]:
        """Анализирует параметры локального файла перед чтением."""
        stats = os.stat(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        return {
            "имя_файла": os.path.basename(file_path),
            "расширение": ext,
            "размер_мб": round(stats.st_size / (1024 * 1024), 2),
            "путь": file_path,
            "тип_контента": (
                "Документ" if ext in [".pdf", ".docx", ".doc"] else "Текст/Лог"
            ),
        }

    def suggest_summarize_format(self, text: str) -> Dict[str, Any]:
        """Определяет структуру текста и рекомендует формат анализа."""
        length = len(text)
        lines = text.count("\n")

        has_many_digits = len(re.findall(r"\d+", text)) > 20

        if length > 5000 or has_many_digits:
            recommendation = "thesis"
            reason = (
                "Текст объемный или содержит много данных, тезисный план будет удобнее."
            )
        elif lines < 5:
            recommendation = "abstractive"
            reason = "Текст компактный, лучше всего подойдет краткий пересказ сути."
        else:
            recommendation = "extractive"
            reason = "В тексте много конкретики, выделим ключевые факты."

        return {
            "recommended": recommendation,
            "reason": reason,
            "stats": {"chars": length, "lines": lines},
        }

    def process_employee_info(self, emp: Any) -> Dict[str, Any]:
        """Формирует расширенную аналитическую карточку сотрудника."""
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
            "структура": {
                "руководитель_подразделения": emp.currentUserLeader,
                "код_департамента": self.get_safe(emp, "department.departmentCode"),
                "id_департамента": str(emp.departmentId) if emp.departmentId else None,
            },
        }
