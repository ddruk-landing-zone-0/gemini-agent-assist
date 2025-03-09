import hashlib
from sqlalchemy import create_engine, Column, String, Table, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text



class InMemoryDB:
    def __init__(self):
        self.engine = create_engine('sqlite:///:memory:', echo=False)
        self.metadata = MetaData()
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def create_table(self, table_name, columns):
        """Creates a table dynamically with a SHA-256 hash primary key to prevent duplicates."""
        table = Table(
            table_name, self.metadata,
            Column("id", String, primary_key=True),  # Primary key hash column
            *[Column(col, String) for col in columns],
        )
        table.create(self.engine)

    def generate_hash(self, data):
        """Generates a SHA-256 hash over the string representation of a row."""
        row_string = str(sorted(data.items()))  # Ensure consistent ordering
        return hashlib.sha256(row_string.encode()).hexdigest()

    def insert_data(self, table_name, data):
        """Inserts a row into the table using parameterized queries and avoids duplicates."""
        data["id"] = self.generate_hash(data)  # Add hash key to data
        placeholders = ", ".join([f":{key}" for key in data.keys()])
        query = text(f"""
            INSERT INTO {table_name} ({', '.join(data.keys())})
            VALUES ({placeholders})
            ON CONFLICT(id) DO NOTHING
        """)
        self.session.execute(query, data)
        self.session.commit()

    def query_data(self, query):
        """Executes a SELECT query and returns results with column names."""
        result = self.session.execute(text(query))
        columns = result.keys()  # Get column names
        data = result.fetchall()  # Get data rows

        # Convert data to list of list from list of tuples
        data = [list(row) for row in data]
        return list(columns), data  # Return both columns and data

    def __del__(self):
        self.session.close()





def get_title_data_inmemory_db(rule_based_title_comment_data):
    # Initialize DB and create table
    columns = list(rule_based_title_comment_data[0].keys())
    columns.remove("id") if "id" in columns else None  # Ensure id isn't duplicated
    inmemory_db = InMemoryDB()
    inmemory_db.create_table("title_data", columns)

    # Insert data
    for data in rule_based_title_comment_data:
        inmemory_db.insert_data("title_data", data)

    return inmemory_db


TITLE_DATA_INMEM_DB = None