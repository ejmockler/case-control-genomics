from typing import List, Optional, Dict
from pydantic import BaseModel


class VCFConfig(BaseModel):
    path: str
    binarize: bool = False
    zygosity: bool = True
    min_allele_frequency: float = 0.0025
    max_allele_frequency: float = 1.0
    max_variants: Optional[int] = None
    filter: str = ""

class GTFConfig(BaseModel):
    path: str
    filter: str = ""

class TrackingConfig(BaseModel):
    name: str
    entity: str
    project: str
    plot_all_sample_importances: bool = False

class TableMetadata(BaseModel):
    name: str
    path: str
    label: str
    id_column: str
    filter: str
    # Change strata_columns to a mapping from standard strata to table-specific column names
    strata_mapping: Optional[Dict[str, str]] = None  # e.g., {"sex": "Sex Call", "age": "Age"}

class SampleTableConfig(BaseModel):
    tables: List[TableMetadata]

class SamplingConfig(BaseModel):
    bootstrap_iterations: int = 60
    cross_val_iterations: int = 10
    test_size: float = 0.2
    sequestered_ids: List[str] = []
    shuffle_labels: bool = False
    # Specify which strata to use for sampling
    strata: Optional[List[str]] = ["sex"]  # Allowed values: "sex", "age"

class ModelConfig(BaseModel):
    hyperparameter_optimization: bool = True
    calculate_shapely_explanations: bool = False

class Config(BaseModel):
    vcf: VCFConfig
    gtf: GTFConfig
    tracking: TrackingConfig
    crossval_tables: SampleTableConfig
    holdout_tables: SampleTableConfig
    sampling: SamplingConfig
    model: ModelConfig

config = Config(
    vcf=VCFConfig(
        path="../adhoc analysis/mock.vcf.gz",
        binarize=False,
        zygosity=True,
        min_allele_frequency=0.0025,
        max_allele_frequency=1.0,
        max_variants=None,
        filter="",
    ),
    gtf=GTFConfig(
        path="../adhoc analysis/gencode.v46.chr_patch_hapl_scaff.annotation.gtf.gz",
        filter="(ht.transcript_type == 'protein_coding') | (ht.transcript_type == 'protein_coding_LoF')", # to keep
    ),
    tracking=TrackingConfig(
        name="NUP variants (rare-binned, rsID only)\nTrained on: AnswerALS cases & non-neurological controls (Caucasian)",
        entity="ejmockler",
        project="highReg-l1-NUPs60-aals-rsID-rareBinned-0.0025MAF",
        plot_all_sample_importances=False,
    ),
    crossval_tables=SampleTableConfig(
        tables=[
            TableMetadata(
                name="AnswerALS cases, EUR",
                path="../adhoc analysis/ACWM.xlsx",
                label="case",
                id_column="ExternalSampleId",
                filter="`Subject Group`=='ALS Spectrum MND' & `pct_european`>=0.85",
                strata_mapping={
                    "sex": "Sex Call",
                    # "age": "Age"  # Add if available
                },
            ),
            TableMetadata(
                name="AnswerALS non-neurological controls, EUR",
                path="../adhoc analysis/ACWM.xlsx",
                label="control",
                id_column="ExternalSampleId",
                filter="`Subject Group`=='Non-Neurological Control' & `pct_european`>=0.85",
                strata_mapping={
                    "sex": "Sex Call",
                    # "age": "Age"  # Add if available
                },
            ),
            TableMetadata(
                name="1000 Genomes EUR",
                path="../adhoc analysis/igsr-1000 genomes phase 3 release.tsv",
                label="control",
                id_column="Sample name",
                filter="`Superpopulation code`=='EUR'",
                strata_mapping={
                    "sex": "Sex",
                    # "age": "Age"  # Include if available
                },
            ),
        ]
    ),
    holdout_tables=SampleTableConfig(
        # TODO option to define comparison tables other than crossval
        tables=[
            TableMetadata(
                name="1000 Genomes ethnically-variable, non-EUR",
                path="../adhoc analysis/igsr-1000 genomes phase 3 release.tsv",
                label="control",
                id_column="Sample name",
                filter="`Superpopulation code`!='EUR'",
                strata_mapping={
                    "sex": "Sex",
                    # "age": "Age"  # Include if available
                },
            ),
            TableMetadata(
                name="AnswerALS cases, ethnically-variable, non-EUR",
                path="../adhoc analysis/ACWM.xlsx",
                label="case",
                id_column="ExternalSampleId",
                filter="`Subject Group`=='ALS Spectrum MND' & `pct_european`<0.85",
                strata_mapping={
                    "sex": "Sex Call",
                    # "age": "Age"  # Add if available
                },
            ),
        ]
    ),
    sampling=SamplingConfig(
        bootstrap_iterations=60,
        cross_val_iterations=10,
        test_size=0.2,
        sequestered_ids=[],
        shuffle_labels=False,
        strata=["sex"],  # Define which strata to use
    ),
    model=ModelConfig(
        hyperparameter_optimization=True,
        calculate_shapely_explanations=False,
    ),
)