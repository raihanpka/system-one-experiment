# a partner table becomes rows; CSV and TSV are standard library, Excel uses the optional extra

import pytest
from openpyxl import Workbook

from column_mapping.table import read_table


def test_csv_and_tsv_become_rows(tmp_path):
    csv_path = tmp_path / "in.csv"
    csv_path.write_text("name,amount\nBudi,10\nSiti,20\n", encoding="utf-8")
    assert read_table(csv_path) == [
        {"name": "Budi", "amount": "10"},
        {"name": "Siti", "amount": "20"},
    ]
    tsv_path = tmp_path / "in.tsv"
    tsv_path.write_text("name\tamount\nBudi\t10\n", encoding="utf-8")
    assert read_table(tsv_path) == [{"name": "Budi", "amount": "10"}]


def test_excel_uses_the_first_row_as_the_header(tmp_path):
    path = tmp_path / "in.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.append(["name", "amount"])
    sheet.append(["Budi", 10])
    sheet.append(["Siti", 20])
    book.save(path)
    assert read_table(path) == [
        {"name": "Budi", "amount": 10},
        {"name": "Siti", "amount": 20},
    ]


def test_an_unknown_format_names_the_supported_ones(tmp_path):
    path = tmp_path / "in.pdf"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported table format"):
        read_table(path)
