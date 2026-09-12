import csv
import os
import shutil


DATASET_FOLDER = "dataset"
VALIDATION_FOLDER = "validation"

ORIGINAL_TRAINING_SAMPLES = 200

GESTURES = [
    "fist",
    "open_palm",
    "peace",
    "thumbs_up"
]


def read_csv(path):
    with open(path, "r", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)

    return rows[0], rows[1:]


def write_csv(path, header, rows):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main():

    print("\n==============================")
    print("PREPARING VALIDATION DATA")
    print("==============================")

    os.makedirs(VALIDATION_FOLDER, exist_ok=True)

    total_validation = 0

    for gesture in GESTURES:

        source_path = os.path.join(
            DATASET_FOLDER,
            f"{gesture}.csv"
        )

        validation_path = os.path.join(
            VALIDATION_FOLDER,
            f"{gesture}.csv"
        )

        if not os.path.exists(source_path):
            print(f"Missing: {source_path}")
            continue

        header, rows = read_csv(source_path)

        training_rows = rows[:ORIGINAL_TRAINING_SAMPLES]
        validation_rows = rows[ORIGINAL_TRAINING_SAMPLES:]

        # Save validation samples separately
        write_csv(
            validation_path,
            header,
            validation_rows
        )

        # Keep ONLY the original 200 training samples
        write_csv(
            source_path,
            header,
            training_rows
        )

        print(
            f"{gesture.upper():12} "
            f"Training: {len(training_rows):3} | "
            f"Validation: {len(validation_rows):3}"
        )

        total_validation += len(validation_rows)

    print("\n==============================")
    print("PREPARATION COMPLETE")
    print("==============================")

    print(
        f"\nTotal validation samples: "
        f"{total_validation}"
    )

    print("\nTraining datasets now contain")
    print("only the original 200 samples/class.")

    print("\nValidation datasets are frozen in:")
    print("validation/")


if __name__ == "__main__":
    main()