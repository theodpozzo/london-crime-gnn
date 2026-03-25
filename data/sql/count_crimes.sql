SELECT 
    postcode,
    month,
    crimetype,
    COUNT(*) AS crime_count
FROM 
    crimes
GROUP BY 
    postcode, 
    month, 
    crimetype
ORDER BY 
    postcode, 
    month, 
    crimetype;
