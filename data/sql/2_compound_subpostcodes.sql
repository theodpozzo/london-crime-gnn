UPDATE crimes
SET postcode = CASE
--	E
    WHEN postcode LIKE 'E20' THEN 'E15'
    WHEN postcode LIKE 'E1W' THEN 'E1'
--	EC
	WHEN postcode LIKE 'EC1%' THEN 'EC1'
	WHEN postcode LIKE 'EC2%' THEN 'EC2'
	WHEN postcode LIKE 'EC3%' THEN 'EC3'
	WHEN postcode LIKE 'EC4%' THEN 'EC4'
--	WC
	WHEN postcode LIKE 'WC1%' THEN 'WC1'
	WHEN postcode LIKE 'WC2%' THEN 'WC2'
--	W1
	WHEN postcode LIKE 'W1A' THEN 'W1'
	WHEN postcode LIKE 'W1C' THEN 'W1'
	WHEN postcode LIKE 'W1D' THEN 'W1'
	WHEN postcode LIKE 'W1B' THEN 'W1'
	WHEN postcode LIKE 'W1F' THEN 'W1'
	WHEN postcode LIKE 'W1G' THEN 'W1'
	WHEN postcode LIKE 'W1H' THEN 'W1'
	WHEN postcode LIKE 'W1J' THEN 'W1'
	WHEN postcode LIKE 'W1K' THEN 'W1'
	WHEN postcode LIKE 'W1S' THEN 'W1'
	WHEN postcode LIKE 'W1T' THEN 'W1'
	WHEN postcode LIKE 'W1U' THEN 'W1'
	WHEN postcode LIKE 'W1W' THEN 'W1'
--	SW1
	WHEN postcode LIKE 'SW1A' THEN 'SW1'
	WHEN postcode LIKE 'SW1E' THEN 'SW1'
	WHEN postcode LIKE 'SW1H' THEN 'SW1'
	WHEN postcode LIKE 'SW1P' THEN 'SW1'
	WHEN postcode LIKE 'SW1V' THEN 'SW1'
	WHEN postcode LIKE 'SW1W' THEN 'SW1'
	WHEN postcode LIKE 'SW1X' THEN 'SW1'
	WHEN postcode LIKE 'SW1Y' THEN 'SW1'
--	SE1
	WHEN postcode LIKE 'SE1P' THEN 'SE1'
--	N1
	WHEN postcode LIKE 'N1C' THEN 'N1'
	
    ELSE postcode
END;
