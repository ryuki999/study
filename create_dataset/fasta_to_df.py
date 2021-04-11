import pandas as pd

DATA_COLUMNS = ["year", "month", "day"]
CLADE_COLUMNS = ["clade", "head2"]


def fasta_to_df(
    filename: str,
    continuous_base_num: int,
    header_columns: list,
    feature_columns: list,
) -> tuple:
    with open(f"./{filename}", encoding="utf-8") as f:
        data = f.read().split("\n")
        header = [data[i] for i in range(0, len(data) - 1, 2)]
        header1 = [line.split("|") for line in header]
        feature = [map(float, data[i].split()) for i in range(1, len(data), 2)]
    f.close()
    return pd.DataFrame(header1, columns=header_columns), pd.DataFrame(
        feature, columns=feature_columns
    )


def all_data_df_to_arange_df(header, feature):
    header2 = header.join(
        header["date"].str.split("-", expand=True).set_axis(DATA_COLUMNS, axis=1)
    )
    header3 = header2.dropna(subset=["host"])
    header4 = header3.join(
        header3["clade_head"]
        .str.split("_>", expand=True)
        .set_axis(CLADE_COLUMNS, axis=1)
    )
    data = header4.merge(feature, left_index=True, right_index=True)

    return data


def blsom_outfile_to_df(
    filename: str,
    header_columns: list = None,
):
    with open(f"./{filename}", encoding="utf-8") as f:
        data = f.read().split("\n")
        header = []
        max_x, max_y = map(int, data[0].split())
        for i in range(2, len(data) - 3):
            line = data[i].split("\t")
            x, y = line[1].split()
            head = line[0].strip().split("|")
            header.append(head + [x] + [y] + [line[2]])

        header2 = pd.DataFrame(header, columns=header_columns)
        header2[["x", "y"]] = header2[["x", "y"]].astype(int)
        header3 = header2.join(
            header2["date"].str.split("-", expand=True).set_axis(DATA_COLUMNS, axis=1)
        )
    f.close()
    return header3, max_x, max_y
