"""Dataset Catalog.

Central registry mapping logical dataset identifiers to raw input
directories or file patterns relative to FilePaths().INPUT_DIR.
"""

DATASET_CATALOG = {
    "30-80GeV_v2": {
        "pion_dir": "job_20183881_final",
        "photon_dir": "job_20183880_final",
        "description": "30-80 GeV Pion vs Photon dataset (v2) for BDT and GATr preprocessing.Contains angles of +- 10°"
    },
    "0.5-80GeV":{
        "pion_dir": "job_20201269_final",
        "photon_dir":"job_20201270_final",
        "description": "full engergy range dataset; angles +-10°"
        },
    "10-20GeV" : {
        "pion_dir" : "job_20238274_final",
        "photon_dir" : "job_20238273_final",
        "description" : "specific extra training set for low energy range"
        },
    "40-80GeV" :{
        "pion_dir" : "job_20238270_final",
        "photon_dir" : "job_20238271_final",
        "description" : "specific extra training set for the high energy range",
        },
    "full_range_test" : {
        "pion_dir" : "job_20238268_final",
        "photon_dir" : "job_20238267_final",
        "description" : "testing performance on an independent set to see weaknesses in the model",
        },
}
