from __future__ import annotations


def prompt_selection() -> tuple[str, str]:
    print("Choose what you want to process:")
    print("1. One file")
    print("2. One folder")
    print("3. Nested folder tree")

    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice == "1":
            path = input("Enter file path: ").strip().strip('"')
            return "file", path
        if choice == "2":
            path = input("Enter folder path: ").strip().strip('"')
            return "folder", path
        if choice == "3":
            path = input("Enter root folder path: ").strip().strip('"')
            return "nested_folder", path
        print("Please enter 1, 2, or 3.")
