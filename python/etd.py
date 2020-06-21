import duckdb

conn = duckdb.connect()

conn.execute("CREATE TABLE attributes AS SELECT * FROM read_csv_auto('titanic-attributes.csv');")
conn.execute("CREATE TABLE labels AS SELECT * FROM read_csv_auto('titanic-labels.csv');")

query = f"""
SELECT 
	(a.value > 3) AS left,
	l.label AS label,
	COUNT(*) AS num_records
FROM attributes a
JOIN labels l ON a.record_id = l.record_id
WHERE a.attribute = 'age'
GROUP BY (a.value > 3), l.label
"""

counts = conn.execute(query).fetchdf()

print(counts)

query = f"""
SELECT 
	(a.value > 3) AS left,
	l.label,
	COUNT(*)
FROM attributes a
JOIN labels l ON a.record_id = l.record_id
WHERE a.attribute = 'age'
GROUP BY (a.value > 3), l.label
"""

counts = conn.execute(query).fetchdf()

print(counts)