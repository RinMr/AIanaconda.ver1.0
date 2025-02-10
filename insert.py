import os

folder_path = ""

for file_name in os.listdir(folder_path):
    full_path = os.path.join(folder_path, file_name)
    
    if os.path.isfile(full_path):
        new_file_name = "angry-dog-" + file_name
        new_full_path = os.path.join(folder_path, new_file_name)
        
        os.rename(full_path, new_full_path)
        print(f"{file_name} を {new_file_name} に変更しました。")

print("すべての画像ファイル名の変更が完了しました。")
