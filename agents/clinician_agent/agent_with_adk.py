from rules import ProtocolRulesEngine
from data import PatientDataLoader
from config import PROTOCOL_TABLE_PATH, DATASET_PATH
engine = ProtocolRulesEngine(PROTOCOL_TABLE_PATH)
loader = PatientDataLoader(DATASET_PATH)
print("Rules:", engine.total_rules)
print("Patients:", loader.all_ids())
sb = engine.evaluate_symptom_burden(1, 1, 62, 58)
print("SB:", sb.output)
