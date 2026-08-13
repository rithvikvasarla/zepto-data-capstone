
import os
import re
import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://books.toscrape.com/"
FIXED_GBP_TO_INR = 105.50

DB_PATH = "books.db"
OUTPUT_DIR = "outputs"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# ============================================================
# SCRAPING
# ============================================================

def scrape_books(num_pages=5):

    books = []

    for page in range(1, num_pages + 1):

        if page == 1:
            url = BASE_URL
        else:
            url = (
                f"{BASE_URL}"
                f"catalogue/page-{page}.html"
            )

        print(f"Scraping page {page}: {url}")

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        products = soup.select(
            "article.product_pod"
        )

        print(
            f"Books found on page {page}: "
            f"{len(products)}"
        )

        for product in products:

            title_tag = product.select_one(
                "h3 a"
            )

            title = (
                title_tag.get("title", "").strip()
                if title_tag
                else None
            )

            price_tag = product.select_one(
                ".price_color"
            )

            price = (
                price_tag.get_text(strip=True)
                if price_tag
                else None
            )

            availability_tag = product.select_one(
                ".availability"
            )

            availability = (
                availability_tag.get_text(
                    " ",
                    strip=True
                )
                if availability_tag
                else None
            )

            rating_tag = product.select_one(
                ".star-rating"
            )

            star_rating = None

            if rating_tag:

                classes = rating_tag.get(
                    "class",
                    []
                )

                for word in [
                    "One",
                    "Two",
                    "Three",
                    "Four",
                    "Five"
                ]:

                    if word in classes:
                        star_rating = word
                        break

            category = "Unknown"

            book_link = (
                title_tag.get("href")
                if title_tag
                else None
            )

            if book_link:

                book_url = requests.compat.urljoin(
                    url,
                    book_link
                )

                try:

                    book_response = requests.get(
                        book_url,
                        headers=HEADERS,
                        timeout=15
                    )

                    book_response.raise_for_status()

                    book_soup = BeautifulSoup(
                        book_response.text,
                        "html.parser"
                    )

                    breadcrumb = book_soup.select(
                        "ul.breadcrumb li a"
                    )

                    if len(breadcrumb) >= 3:

                        category = (
                            breadcrumb[-1]
                            .get_text(strip=True)
                        )

                except requests.RequestException:

                    category = "Unknown"

            books.append({
                "title": title,
                "price": price,
                "star_rating": star_rating,
                "availability": availability,
                "category": category
            })

    return pd.DataFrame(books)


# ============================================================
# CLEANING
# ============================================================

def clean_data(df):

    df = df.copy()

    def parse_price(value):

        if pd.isna(value):
            return None

        try:

            match = re.search(
                r"\d+(?:\.\d+)?",
                str(value)
            )

            if match:
                return float(match.group())

        except Exception:
            pass

        return None

    df["price_gbp"] = df[
        "price"
    ].apply(parse_price)

    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    df["rating"] = df[
        "star_rating"
    ].map(rating_map)

    def parse_stock(value):

        if pd.isna(value):
            return False

        text = str(value).strip().lower()

        if "out of stock" in text:
            return False

        if "in stock" in text:
            return True

        return False

    df["in_stock"] = df[
        "availability"
    ].apply(parse_stock)

    # Median imputation for numeric fields
    for column in [
        "price_gbp",
        "rating"
    ]:

        if df[column].isna().any():

            median_value = df[
                column
            ].median()

            df[column] = df[
                column
            ].fillna(median_value)

    df["category"] = df[
        "category"
    ].fillna("Unknown")

    # Fixed project-defined conversion
    df["price_inr"] = (
        df["price_gbp"]
        * FIXED_GBP_TO_INR
    ).round(2)

    df["price_gbp"] = df[
        "price_gbp"
    ].astype(float)

    df["rating"] = (
        df["rating"]
        .round()
        .astype(int)
    )

    df["in_stock"] = df[
        "in_stock"
    ].astype(bool)

    df["price_inr"] = df[
        "price_inr"
    ].astype(float)

    return df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ]


# ============================================================
# DATABASE CREATION
# ============================================================

def create_database(clean_df):

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    cursor = conn.cursor()

    cursor.execute(
        "DROP TABLE IF EXISTS books"
    )

    cursor.execute(
        "DROP TABLE IF EXISTS categories"
    )

    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    categories = sorted(
        clean_df["category"].unique()
    )

    for category in categories:

        cursor.execute(
            """
            INSERT INTO categories (
                category_name
            )
            VALUES (?)
            """,
            (category,)
        )

    cursor.execute("""
        SELECT category_id, category_name
        FROM categories
    """)

    category_map = {
        name: category_id
        for category_id, name
        in cursor.fetchall()
    }

    for _, row in clean_df.iterrows():

        cursor.execute(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(row["in_stock"]),
                category_map[
                    row["category"]
                ]
            )
        )

    conn.commit()

    return conn


# ============================================================
# SQL QUERIES
# ============================================================

def run_queries(conn):

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

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results = {}

    output_file = os.path.join(
        OUTPUT_DIR,
        "query_outputs.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for name, query in queries.items():

            result = pd.read_sql(
                query,
                conn
            )

            results[name] = result

            f.write("=" * 80 + "\n")
            f.write(name.upper() + "\n")
            f.write("=" * 80 + "\n\n")

            f.write("SQL QUERY:\n")
            f.write(query.strip() + "\n\n")

            f.write("OUTPUT:\n")
            f.write(
                result.to_string(
                    index=False
                )
            )

            f.write("\n\n")

            result.to_csv(
                os.path.join(
                    OUTPUT_DIR,
                    f"{name}.csv"
                ),
                index=False
            )

    return queries, results


# ============================================================
# SQL JOIN VS PANDAS MERGE
# ============================================================

def compare_sql_and_merge(
    clean_df,
    conn,
    sql_join_result
):

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        conn
    )

    category_id_map = dict(
        zip(
            categories_df["category_name"],
            categories_df["category_id"]
        )
    )

    books_df = clean_df.copy()

    books_df["category_id"] = (
        books_df["category"]
        .map(category_id_map)
    )

    merge_result = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    merge_result = merge_result[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_name"
        ]
    ]

    merge_result = (
        merge_result
        .sort_values(
            by=["rating", "title"],
            ascending=[False, True]
        )
        .head(10)
        .reset_index(drop=True)
    )

    sql_join_result = (
        sql_join_result
        .copy()
        .reset_index(drop=True)
    )

    sql_join_result["in_stock"] = (
        sql_join_result["in_stock"]
        .astype(bool)
    )

    merge_result["in_stock"] = (
        merge_result["in_stock"]
        .astype(bool)
    )

    for column in [
        "price_gbp",
        "price_inr"
    ]:

        sql_join_result[column] = (
            sql_join_result[column]
            .astype(float)
            .round(2)
        )

        merge_result[column] = (
            merge_result[column]
            .astype(float)
            .round(2)
        )

    sql_join_result["rating"] = (
        sql_join_result["rating"]
        .astype(int)
    )

    merge_result["rating"] = (
        merge_result["rating"]
        .astype(int)
    )

    merge_result = merge_result[
        sql_join_result.columns
    ]

    merge_result = (
        merge_result
        .reset_index(drop=True)
    )

    results_match = (
        sql_join_result.equals(
            merge_result
        )
    )

    sql_join_result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "sql_join_result.csv"
        ),
        index=False
    )

    merge_result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "merge_results.csv"
        ),
        index=False
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "sql_vs_merge_comparison.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SQL JOIN RESULT\n"
        )

        f.write(
            sql_join_result.to_string(
                index=False
            )
        )

        f.write(
            "\n\nPANDAS pd.merge() RESULT\n"
        )

        f.write(
            merge_result.to_string(
                index=False
            )
        )

        f.write(
            f"\n\nResults equivalent: "
            f"{results_match}\n"
        )

    return (
        categories_df,
        books_df,
        merge_result,
        results_match
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("ZEpto DATA PIPELINE")
    print("=" * 70)

    # Scrape
    raw_df = scrape_books(
        num_pages=5
    )

    print(
        f"\nTotal books scraped: "
        f"{len(raw_df)}"
    )

    # Validate scraping
    assert len(raw_df) >= 60
    assert raw_df["category"].nunique() >= 3

    # Clean
    clean_df = clean_data(
        raw_df
    )

    print(
        f"Cleaned books: "
        f"{len(clean_df)}"
    )

    # Validate conversion
    expected_price_inr = (
        clean_df["price_gbp"]
        * FIXED_GBP_TO_INR
    ).round(2)

    assert (
        clean_df["price_inr"]
        == expected_price_inr
    ).all()

    # Database
    conn = create_database(
        clean_df
    )

    print(
        f"Database created: "
        f"{DB_PATH}"
    )

    # SQL
    queries, results = run_queries(
        conn
    )

    print(
        f"SQL queries executed: "
        f"{len(queries)}"
    )

    # pd.read_sql requirement
    read_sql_result_1 = pd.read_sql(
        queries[
            "query_1_select_where"
        ],
        conn
    )

    read_sql_result_2 = pd.read_sql(
        queries[
            "query_6_join"
        ],
        conn
    )

    read_sql_result_1.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "read_sql_select_where.csv"
        ),
        index=False
    )

    read_sql_result_2.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "read_sql_join.csv"
        ),
        index=False
    )

    # SQL JOIN vs pandas merge
    (
        categories_df,
        books_df,
        merge_result,
        results_match
    ) = compare_sql_and_merge(
        clean_df,
        conn,
        read_sql_result_2
    )

    assert results_match

    # Final validation
    assert len(clean_df) >= 60
    assert clean_df["category"].nunique() >= 3
    assert clean_df["rating"].between(
        1, 5
    ).all()

    assert clean_df["in_stock"].dtype == bool

    assert (
        clean_df["price_inr"]
        == (
            clean_df["price_gbp"]
            * FIXED_GBP_TO_INR
        ).round(2)
    ).all()

    print("\n" + "=" * 70)
    print("DATA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(
        f"Books: {len(clean_df)}"
    )
    print(
        f"Categories: "
        f"{clean_df['category'].nunique()}"
    )
    print(
        "GBP → INR: "
        "1 GBP = 105.50 INR"
    )
    print(
        "SQL JOIN = pandas merge: "
        f"{results_match}"
    )
    print(
        f"Database: {DB_PATH}"
    )
    print(
        f"Outputs: {OUTPUT_DIR}/"
    )
    print("=" * 70)

    conn.close()


if __name__ == "__main__":
    main()
