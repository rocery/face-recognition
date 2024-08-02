import mysql.connector

rows = None

# Replace with your MySQL database connection details
config = {
    'user': 'admin',
    'password': 'itbekasioke',
    'host': '192.168.10.223',
    'database': 'counter_hit',
    'raise_on_warnings': True
}

def

try:
    # Establish connection to MySQL server
    cnx = mysql.connector.connect(**config)

    # Create cursor object
    cursor = cnx.cursor()
    id  =  1
    nip = 2
    name = 'Sastra'
    category = 'IN'
    date = '2023-05-01 10:00:00'

    # Example complex SQL query with multiple WHERE conditions
    query = (
            "INSET INTO `presensi` (`id`, `nip`, `name`, `category`, `date`) "
            "VALUES (id, nip, name, category, date) "
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