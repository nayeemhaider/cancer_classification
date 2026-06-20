-- Build the feature table that will be fed directly into the model.
-- All transformations happen here in SQL so the ML code stays clean.

CREATE TABLE IF NOT EXISTS features AS
SELECT
    patient_id,
    cancer_type,

    -- Demographics
    age,
    gender,
    bmi,

    -- Lifestyle risk factors (raw scores 0-10)
    smoking,
    alcohol_use,
    physical_activity,
    fruit_veg_intake,
    diet_red_meat,
    diet_salted_processed,

    -- Environmental / occupational
    air_pollution,
    occupational_hazards,

    -- Medical / genetic markers
    obesity,
    family_history,
    brca_mutation,
    h_pylori_infection,
    calcium_intake,
    physical_activity_level,

    -- Composite score from source data
    overall_risk_score,
    risk_level_encoded,

    -- Derived: lifestyle burden (sum of behaviours known to raise risk)
    (smoking + alcohol_use + diet_red_meat + diet_salted_processed) AS lifestyle_burden,

    -- Derived: protective behaviours (higher = more protective)
    (fruit_veg_intake + physical_activity) AS protective_score,

    -- Derived: environmental exposure total
    (air_pollution + occupational_hazards) AS env_exposure,

    -- Derived: simple BMI category as integer
    CASE
        WHEN bmi < 18.5 THEN 0
        WHEN bmi BETWEEN 18.5 AND 24.9 THEN 1
        WHEN bmi BETWEEN 25.0 AND 29.9 THEN 2
        ELSE 3
    END AS bmi_category,

    -- Derived: flag for senior patients
    CASE WHEN age >= 65 THEN 1 ELSE 0 END AS is_senior

FROM cleaned_cancer;

-- Quick sanity check on new derived columns
SELECT
    ROUND(AVG(lifestyle_burden), 2)  AS avg_lifestyle_burden,
    ROUND(AVG(protective_score), 2)  AS avg_protective_score,
    ROUND(AVG(env_exposure), 2)      AS avg_env_exposure,
    COUNT(CASE WHEN is_senior = 1 THEN 1 END) AS senior_patients
FROM features;

-- Per-cancer-type averages of key features (useful EDA output)
SELECT
    cancer_type,
    ROUND(AVG(age), 1)               AS avg_age,
    ROUND(AVG(smoking), 2)           AS avg_smoking,
    ROUND(AVG(lifestyle_burden), 2)  AS avg_lifestyle_burden,
    ROUND(AVG(overall_risk_score), 4) AS avg_risk_score,
    COUNT(*)                          AS n
FROM features
GROUP BY cancer_type
ORDER BY avg_risk_score DESC;
