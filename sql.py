import mysql.connector

rows = []

# Replace with your MySQL database connection details
config = {
    'user': 'admin',
    'password': 'itbekasioke',
    'host': '192.168.10.223',
    'database': 'counter_hit',
    'raise_on_warnings': True
}

try:
    # Establish connection to MySQL server
    cnx = mysql.connector.connect(**config)

    # Create cursor object
    cursor = cnx.cursor()

    # Example complex SQL query with multiple WHERE conditions
    query = (
        "SELECT id_packing, jenis_packing, id_bagian "
        "FROM jenis_packing "
        "WHERE id_bagian > 3"
    )

    # Execute SQL query
    cursor.execute(query)

    # Fetch all rows from the result set into a list
    rows = cursor.fetchall()

    # Close cursor and database connection
    cursor.close()
    cnx.close()

except mysql.connector.Error as err:
    print(f"Error: {err}")

# Process each row (printing after connection is closed)
for row in rows:
    print(row[0])