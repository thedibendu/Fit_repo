from fit_repo import run_hill_fit_fully_automated
from fit_repo import run_hill_fit_multiple_from_excel
import os


home = os.path.expanduser("~")

raw_file_path = os.path.join(
    home,
    "Documents",
    "atp_analysis",
    "automated.xlsx"
)

combined_file_path = os.path.join(
    home,
    "Documents",
    "atp_analysis",
    "Book2.xlsx"
)

# Initial-rate fitting controls.
# MAX_POINTS uses the first N selected time points. Use None to use all selected points.
FIT_START = None
FIT_END = None
MAX_POINTS = 2


if __name__ == "__main__":
    if not os.path.exists(raw_file_path):
        print(f"Error: File not found at {raw_file_path}")
        raise SystemExit

    print(f"Loading raw data from: {raw_file_path}")

    run_hill_fit_fully_automated(
        raw_file_path,
        fit_start=FIT_START,
        fit_end=FIT_END,
        max_points=MAX_POINTS
    )

    answer = input(
        "\nAfter updating Book2.xlsx, type yes to run combined fit: "
    )

    if answer.strip().lower() == "yes":
        if not os.path.exists(combined_file_path):
            print(f"Error: File not found at {combined_file_path}")
            raise SystemExit

        print(f"Loading combined data from: {combined_file_path}")
        run_hill_fit_multiple_from_excel(combined_file_path)
    else:
        print("Stopped before combined fit.")
