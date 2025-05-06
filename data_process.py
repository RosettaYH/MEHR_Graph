import torch
from tqdm import tqdm
import pandas as pd
import pickle

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import (
    # drug_recommendation_mimic4_fn,
    mortality_prediction_mimic4_fn,
    readmission_prediction_mimic4_fn
)

from torch_geometric.data import HeteroData
from pyhealth.data import Patient, Visit

def drug_recommendation_mimic4_fn(patient: Patient):
    """
    Based on https://pyhealth.readthedocs.io/en/latest/_modules/pyhealth/tasks/drug_recommendation.html#drug_recommendation_mimic4_fn
    Processes a single patient for the drug recommendation task.

    Drug recommendation aims at recommending a set of drugs given the patient health
    history  (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> mimic4_base = MIMIC4Dataset(
        ...     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        ...     tables=["diagnoses_icd", "procedures_icd"],
        ...     code_mapping={"ICD10PROC": "CCSPROC"},
        ... )
        >>> from pyhealth.tasks import drug_recommendation_mimic4_fn
        >>> mimic4_sample = mimic4_base.set_task(drug_recommendation_mimic4_fn)
        >>> mimic4_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'label': [['2', '3', '4']]}]
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        # drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_all": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + [
            samples[i]["drugs_all"]
        ]

    return samples

class EHRGraph:
    def __init__(self):
        self.dataset_name = "mimic4"
        self.graph_path = "./graphs/"
        self.raw_path = "./hosp"

        self.dataset = None
        self.graph = None
        self.mappings = None

    def load_mimic(self):
        self.dataset = MIMIC4Dataset(
                root=self.raw_path,
                tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
                code_mapping={},
                dev=False,
            )
        print(self.dataset.stat())

    def construct_graph(self):
        graph_data = self.get_graph_data()
        data = HeteroData()
        for node_type, mapping in self.mappings.items():
            data[node_type].num_nodes = len(mapping)
        for (src, rel, dst), (src_ids, dst_ids) in graph_data.items():
            edge_index = torch.stack([src_ids, dst_ids], dim=0)
            data[(src, rel, dst)].edge_index = edge_index
        self.graph = data

    def get_graph_data(self):
        patients = self.dataset.patients
        patients_dict = {k: i for i, k in enumerate(patients.keys())}
        visits_set = set()
        diagnosis_set = set()
        procedures_set = set()
        prescriptions_set = set()

        patient_visit_edges = []
        visit_diagnosis_edges = []
        visit_procedure_edges = []
        visit_prescription_edges = []

        for patient in tqdm(patients.values()):
            for visit in patient.visits.keys():
                visits_set.add(visit)
                patient_visit_edges.append((patient.patient_id, visit))
            for visit in patient.visits.values():
                ev_dict = visit.event_list_dict
                for table in visit.available_tables:
                    if table.upper() == "DIAGNOSES_ICD":
                        for ev in ev_dict[table]:
                            diagnosis_set.add(ev.code)
                            visit_diagnosis_edges.append((ev.visit_id, ev.code))
                    if table.upper() == "PROCEDURES_ICD":
                        for ev in ev_dict[table]:
                            procedures_set.add(ev.code)
                            visit_procedure_edges.append((ev.visit_id, ev.code))
                    if table.upper() == "PRESCRIPTIONS":
                        for ev in ev_dict[table]:
                            prescriptions_set.add(ev.code)
                            visit_prescription_edges.append((ev.visit_id, ev.code))

        def _set_to_dict(s):
            return {e: i for i, e in enumerate(s)}
    
        visits_dict = _set_to_dict(visits_set)
        diagnosis_dict = _set_to_dict(diagnosis_set)
        procedures_dict = _set_to_dict(procedures_set)
        prescriptions_dict = _set_to_dict(prescriptions_set)

        patient_visit_edges = [(patients_dict[p], visits_dict[v]) for (p, v) in patient_visit_edges]
        visit_diagnosis_edges = [(visits_dict[v], diagnosis_dict[d]) for (v, d) in visit_diagnosis_edges]
        visit_procedure_edges = [(visits_dict[v], procedures_dict[l]) for (v, l) in visit_procedure_edges]
        visit_prescription_edges = [(visits_dict[v], prescriptions_dict[l]) for (v, l) in visit_prescription_edges]

        graph_data = {}

        def update_edges(head, rel, tail, edges):
            src_tensor = torch.tensor([e[0] for e in edges])
            dst_tensor = torch.tensor([e[1] for e in edges])
            graph_data[(head, rel, tail)] = (src_tensor, dst_tensor)

        update_edges("patient", "makes", "visit", patient_visit_edges)
        update_edges("visit", "diagnosed", "diagnosis", visit_diagnosis_edges)
        update_edges("visit", "prescribed", "prescription", visit_prescription_edges)
        update_edges("visit", "treated", "procedure", visit_procedure_edges)

        self.mappings = {
            "patient": patients_dict,
            "visit": visits_dict,
            "diagnosis": diagnosis_dict,
            "procedure": procedures_dict,
            "prescription": prescriptions_dict,
        }

        with open(f'{self.graph_path}{self.dataset_name}_entity_mapping.pkl', 'wb') as outp:
            pickle.dump(self.mappings, outp, pickle.HIGHEST_PROTOCOL)

        return graph_data

    def set_tasks(self):
        mort_pred_samples = self.dataset.set_task(mortality_prediction_mimic4_fn)
        drug_rec_samples = self.dataset.set_task(drug_recommendation_mimic4_fn)
        readm_samples = self.dataset.set_task(readmission_prediction_mimic4_fn)
    
        vm = self.mappings["visit"]
        n_nodes = self.graph["visit"].num_nodes

        # Assign task-specific labels
        mort_pred = {}
        for s in mort_pred_samples:
            visit_id = s["visit_id"]
            mort_pred.update({vm[visit_id]: s["label"]})

        drug_rec = {}
        for s in drug_rec_samples:
            print(s['drugs'])
            visit_id = s["visit_id"]
            drug_rec.update({vm[visit_id]: s["drugs"]})

        readm = {}
        for s in readm_samples:
            visit_id = s["visit_id"]
            readm.update({vm[visit_id]: s["label"]})


        labels = {
            "mort_pred": mort_pred,
            "drug_rec": drug_rec,
            "all_drugs": drug_rec_samples.get_all_tokens("drugs"),
            "readm": readm
        }
        with open(f'{self.graph_path}{self.dataset_name}_labels.pkl', 'wb') as outp:
            pickle.dump(labels, outp, pickle.HIGHEST_PROTOCOL)

    def save_graph(self):
        with open(f'{self.graph_path}{self.dataset_name}.pkl', 'wb') as outp:
            pickle.dump(self.graph, outp, pickle.HIGHEST_PROTOCOL)

    def process_graph(self):
        self.load_mimic()
        self.construct_graph()
        self.set_tasks()
        self.save_graph()
