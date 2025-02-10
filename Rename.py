import os

directory = ""

new_name_base = ""  # dog...
new_extension = ".jpg"

valid_extensions = ['.jpg', '.jpeg', '.png', '.jfif']

try:
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in valid_extensions]

    count = 1

    for file in files:
        old_path = os.path.join(directory, file)
        new_name = f"{new_name_base}-{count}{new_extension}"
        new_path = os.path.join(directory, new_name)

        try:
            os.rename(old_path, new_path)
            print(f"{file} -> {new_name} に変更しました。")
            count += 1
        except Exception as e:
            print(f"{file} の名前変更中にエラーが発生しました: {e}")

except Exception as e:
    print(f"フォルダ処理中にエラーが発生しました: {e}")
