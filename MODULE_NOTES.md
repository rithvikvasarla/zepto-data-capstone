# Module 1 Notes

## Pipeline Flow

The module follows the raw-to-relational pipeline:

Scrape → Clean → Convert → Store → Query → Validate

## Dataset

The pipeline scrapes the first five catalogue pages from Books to Scrape and collects 100 books across multiple categories.

## Currency Conversion

The required fixed project baseline is:

1 GBP = 105.50 INR

The conversion does not use a live currency API.

## Database

The SQLite database contains two related tables:

- categories
- books

The books table references categories using category_id.

## Validation

The pipeline verifies:

- at least 60 books
- at least 3 categories
- valid ratings from 1 to 5
- boolean stock status
- correct GBP to INR conversion
- required SQL queries
- SQL JOIN and pandas merge equivalence
