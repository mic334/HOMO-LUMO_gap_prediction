from parser.data_loader import QM9Parser


def main():
    input_path = "../data/raw/qm9_dataset"
    output_path = "../data/processed/qm9_gap_dataset.csv"

    parser = QM9Parser(input_path)
    print(parser.folder_path)

    df = parser.parse_folder()
    parser.save_csv(df, output_path)

    print("Pipeline completata!")


if __name__ == "__main__":
    main()