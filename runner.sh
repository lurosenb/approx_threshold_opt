#!/bin/bash

datasets=(
    "ACSIncome_mfopt"
    "ACSIncome"
    "ACSMobility_mfopt"
    "ACSMobility"
    "ACSEmployment_mfopt"
    "ACSEmployment"
    "ACSPublicCoverage_mfopt"
    "ACSPublicCoverage"
    "ACSPoverty_mfopt"
    "ACSPoverty"
    "ACSInsurance_mfopt"
    "ACSInsurance"
    "ACSTravelTime_mfopt"
    "ACSTravelTime"
    "diabetes"
    "heart_disease"
    "loans"
    "students"
)

for dataset in "${datasets[@]}"; do
    echo "Running pipeline for dataset: $dataset"
    python3 runner.py --dataset_name "$dataset" --output_file "results/$dataset.pkl"
    echo "Completed pipeline for dataset: $dataset"
    echo " "
done
