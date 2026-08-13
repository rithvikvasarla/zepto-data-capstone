
import sqlite3
import pandas as pd


DB_PATH = "books.db"


queries = {

    "query_1_select_where": """
        SELECT
            title,
            price_gbp,
            rating,
            in_stock
        FROM books
        WHERE rating >= 4
    """,

    "query_2_order_by": """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        ORDER BY price_gbp DESC
    """,

    "query_3_limit": """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        ORDER BY rating DESC
        LIMIT 10
    """,

    "query_4_distinct": """
        SELECT DISTINCT
            c.category_name
        FROM categories c
        JOIN books b
            ON c.category_id = b.category_id
        ORDER BY c.category_name
    """,

    "query_5_between": """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp
    """,

    "query_6_join": """
        SELECT
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY
            b.rating DESC,
            b.title
        LIMIT 10
    """
}


def main():

    conn = sqlite3.connect(DB_PATH)

    print("=" * 70)
    print("SQL QUERY RESULTS")
    print("=" * 70)

    for name, query in queries.items():

        print("\n" + "-" * 70)
        print(name)
        print("-" * 70)

        print(query.strip())

        result = pd.read_sql(
            query,
            conn
        )

        print("\nOUTPUT:")
        print(result.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
