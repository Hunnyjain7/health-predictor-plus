from typing import List, Optional
from pydantic import BaseModel


class HealthRecord(BaseModel):
    full_name: str
    email: str
    address: Optional[str] = None
    pincode: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    average_sleep_hours: Optional[float] = None
    last_physical_exam: Optional[str] = None
    daily_water_intake_litres: Optional[float] = None
    steps_count_per_day: Optional[int] = None
    daily_exercise_minutes: Optional[float] = None
    work_hours: Optional[float] = None
    systolic_pressure: Optional[int] = None
    diastolic_pressure: Optional[int] = None
    heart_rate_bpm: Optional[int] = None
    blood_sugar_levels_mg_dl: Optional[int] = None
    medical_history: Optional[List[str]] = None
    heredity_diseases: Optional[List[str]] = None
    smoking_status: Optional[List[str]] = None
    alcohol_consumption: Optional[List[str]] = None
    physical_activity_level: Optional[List[str]] = None
    diet_type: Optional[List[str]] = None
    stress_level: Optional[List[str]] = None
    current_medications: Optional[List[str]] = None
    frequency_of_checkups: Optional[List[str]] = None
    type_of_physical_activities: Optional[List[str]] = None
    additional_details: Optional[str] = None
