-- Explore the raw dataset before doing anything with it

-- Row count and column overview
SELECT COUNT(*) AS total_rows FROM raw_cancer;

-- Distribution of the target variable
SELECT cancer_type, COUNT(*) AS count,
       ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM raw_cancer), 2) AS pct
FROM raw_cancer
GROUP BY cancer_type
ORDER BY count DESC;

-- Check for nulls in every column
SELECT
    SUM(CASE WHEN patient_id IS NULL THEN 1 ELSE 0 END)           AS null_patient_id,
    SUM(CASE WHEN cancer_type IS NULL THEN 1 ELSE 0 END)          AS null_cancer_type,
    SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END)                  AS null_age,
    SUM(CASE WHEN gender IS NULL THEN 1 ELSE 0 END)               AS null_gender,
    SUM(CASE WHEN smoking IS NULL THEN 1 ELSE 0 END)              AS null_smoking,
    SUM(CASE WHEN bmi IS NULL THEN 1 ELSE 0 END)                  AS null_bmi,
    SUM(CASE WHEN overall_risk_score IS NULL THEN 1 ELSE 0 END)   AS null_risk_score
FROM raw_cancer;

-- Basic stats for numeric columns
SELECT
    ROUND(AVG(age), 2)                   AS avg_age,
    MIN(age)                             AS min_age,
    MAX(age)                             AS max_age,
    ROUND(AVG(bmi), 2)                   AS avg_bmi,
    ROUND(AVG(overall_risk_score), 4)    AS avg_risk_score,
    ROUND(AVG(smoking), 2)               AS avg_smoking
FROM raw_cancer;

-- Gender distribution per cancer type
SELECT cancer_type, gender, COUNT(*) AS count
FROM raw_cancer
GROUP BY cancer_type, gender
ORDER BY cancer_type, gender;

-- Age buckets - useful to see which age groups are most represented
SELECT
    CASE
        WHEN age < 40 THEN 'Under 40'
        WHEN age BETWEEN 40 AND 54 THEN '40-54'
        WHEN age BETWEEN 55 AND 69 THEN '55-69'
        ELSE '70+'
    END AS age_group,
    COUNT(*) AS count
FROM raw_cancer
GROUP BY age_group
ORDER BY count DESC;

-- Check duplicate patient IDs
SELECT patient_id, COUNT(*) AS occurrences
FROM raw_cancer
GROUP BY patient_id
HAVING COUNT(*) > 1;

-- Risk level breakdown
SELECT risk_level, COUNT(*) AS count
FROM raw_cancer
GROUP BY risk_level;
