from pathlib import Path


patient_date_path = "../covid_dataset/aaa/vvv"
print(patient_date_path)
Path(patient_date_path).mkdir(parents=True, exist_ok=True)
