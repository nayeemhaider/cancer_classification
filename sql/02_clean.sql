-- Remove duplicate patient IDs (keep first occurrence by rowid)
CREATE TABLE IF NOT EXISTS cleaned_cancer AS
SELECT *
FROM raw_cancer
WHERE rowid IN (
    SELECT MIN(rowid)
    FROM raw_cancer
    GROUP BY patient_id
);

-- Drop rows where the target label is missing or blank
DELETE FROM cleaned_cancer
WHERE cancer_type IS NULL OR TRIM(cancer_type) = '';

-- Remove records where age is outside a realistic range
DELETE FROM cleaned_cancer WHERE age < 18 OR age > 110;

-- Remove records where BMI is physiologically impossible
DELETE FROM cleaned_cancer WHERE bmi < 10 OR bmi > 70;

-- Cap scores that should be 0-10 to that range (clamp outliers)
UPDATE cleaned_cancer
SET
    smoking              = MAX(0, MIN(10, smoking)),
    alcohol_use          = MAX(0, MIN(10, alcohol_use)),
    obesity              = MAX(0, MIN(10, obesity)),
    air_pollution        = MAX(0, MIN(10, air_pollution)),
    occupational_hazards = MAX(0, MIN(10, occupational_hazards)),
    fruit_veg_intake     = MAX(0, MIN(10, fruit_veg_intake)),
    physical_activity    = MAX(0, MIN(10, physical_activity));

-- Encode risk_level as an integer (useful later for SQL-side EDA)
ALTER TABLE cleaned_cancer ADD COLUMN risk_level_encoded INTEGER;
UPDATE cleaned_cancer
SET risk_level_encoded = CASE
    WHEN LOWER(risk_level) = 'low'    THEN 0
    WHEN LOWER(risk_level) = 'medium' THEN 1
    WHEN LOWER(risk_level) = 'high'   THEN 2
    ELSE NULL
END;

-- Verify final row count after cleaning
SELECT COUNT(*) AS rows_after_cleaning FROM cleaned_cancer;
