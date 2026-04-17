"""diagnose_timezone.py — запустите на сервере приложения"""

import time
import os
from datetime import datetime, timezone, timedelta

print("=" * 60)
print("ДИАГНОСТИКА TIMEZONE СЕРВЕРА")
print("=" * 60)

# 1. Системный time module
print(f"\n📌 time.timezone (сек от UTC, зима): {time.timezone}")
print(f"📌 time.altzone (сек от UTC, лето): {time.altzone}")
print(f"📌 time.tzname: {time.tzname}")
print(f"📌 time.daylight (DST активно?): {time.daylight}")
print(f"📌 localtime().tm_isdst: {time.localtime().tm_isdst}")

# 2. ENV переменные
print(f"\n📌 $TZ = {os.environ.get('TZ', '<не задано>')}")
print(f"📌 $EDMS_DB_TIMEZONE = {os.environ.get('EDMS_DB_TIMEZONE', '<не задано>')}")

# 3. Текущее время разными способами
now_utc = datetime.now(timezone.utc)
now_naive = datetime.now()  # Без tzinfo!
now_local = datetime.now()  # Системное время

print(f"\n⏰ ВРЕМЯ:")
print(f"   UTC (правильное):       {now_utc.strftime('%H:%M:%S')} UTC")
print(f"   Naive (без timezone):   {now_naive.strftime('%H:%M:%S')} ← ⚠️ может быть неверным!")
print(f"   Местное (система):      {now_local.strftime('%H:%M:%S')}")

# 4. Что думает Python о локальном TZ
try:
    import zoneinfo
    try:
        local_tz = zoneinfo.ZoneInfo("Europe/Moscow")
        moscow_now = now_utc.astimezone(local_tz)
        print(f"   Москва (Europe/Moscow): {moscow_now.strftime('%H:%M:%S')}")
    except Exception as e:
        print(f"   Москва (zoneinfo error): {e}")
except ImportError:
    pass

# 5. Ручной расчёт offset
offset_sec = time.timezone if not time.localtime().tm_isdst else time.altzone
offset_hours = -offset_sec / 3600
print(f"\n🌍 Определённый offset системы: {offset_hours:+.1f}h")

# 6. Проверка нашего модуля
try:
    from edms_ai_assistant.utils.datetime_utils import (
        LOCAL_TZ, DB_TZ, get_timezone_info, now_local
    )
    info = get_timezone_info()
    print(f"\n📦 DateTimeUtils:")
    print(f"   LOCAL_TZ: {info['local_tz']}")
    print(f"   Offset:  {info['local_offset_hours']:+.1f}h")
    print(f"   DST:     {'ДА ⚠️' if info['is_dst'] else 'нет ✅'}")
    print(f"   Now:     {info['current_time_local']}")
except Exception as e:
    print(f"\n❌ Ошибка импорта DateTimeUtils: {e}")

print("\n" + "=" * 60)
print("Сравните с https://time.is/ru/Moscow")
print("=" * 60)



# После замены файла выполните:
from edms_ai_assistant.utils.datetime_utils import now_local, get_timezone_info

info = get_timezone_info()
print(f"Системное время: {info['current_time_local']}")
print(f"UTC время:      {info['current_time_utc']}")
print(f"Offset:         {info['local_offset_hours']:+}h")
print(f"DST отключён:   ✅")

# Сравните с:
# https://time.is/ru/Moscow
# Должно совпадать!