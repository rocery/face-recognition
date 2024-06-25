import pymysql

# Replace with your MySQL database connection details
db = pymysql.connect(host='192.168.10.223', user='admin', password='itbekasioke', database='counter_hit')

try:
    # Create a cursor object using cursor() method
    cursor = db.cursor()

    # Execute a SQL query
    cursor.execute("SELECT * FROM counter_hit WHERE id = 1")

    # Fetch all rows from the result set
    rows = cursor.fetchall()
    for row in rows:
        print(row)

except pymysql.Error as e:
    print(f"Error {e}")

finally:
    # Close the cursor and database connection
    if 'cursor' in locals():
        cursor.close()
    if db.open:
        db.close()
